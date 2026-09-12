import argparse
import os
import pickle
import itertools
import numpy as np
import PcmPy as pcm
import pandas as pd
import inspect
from scipy.stats import spearmanr
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.G_matrix as G_matrix
import EFC_learningfMRI.util as util
import EFC_learningfMRI.betas as betas
import EFC_learningfMRI.behaviour as behaviour
import EFC_learningfMRI.pcm as pcm_

def _pair_index():
    """(row, col) indices of the chord pairs, split into the three pair groups.

    """
    mask                           = np.tri(8, k=-1, dtype=bool)
    mask_trained                   = mask.copy()
    mask_untrained                 = mask.copy()
    mask_trained[4:]               = False
    mask_untrained[:, :4]          = False
    mask_trained_untrained         = np.zeros((8, 8), dtype=bool)
    mask_trained_untrained[:4, 4:] = True

    # (classification, row indices, col indices) for the three chord-pair groups
    masks = {'trained'          : mask_trained,
             'untrained'        : mask_untrained,
             'trained_untrained': mask_trained_untrained}
    return {chord: np.where(m) for chord, m in masks.items()}


def _G_rows(G, sn, **labels):
    """One row per chord pair of a single 8x8 G: its crossnobis distance, cosine and angle.

    ``labels`` are copied onto every row, and are what identifies the G the pair
    came from — Hem/roi/session for a neural G, metric/session for a force one.
    The ``pair`` label is order-normalised so both Gs key on the same pair id.
    """
    chords = util.get_trained_and_untrained(sn)
    D      = pcm.G_to_dist(G)
    cos    = pcm.G_to_cosine(G)

    rows = []
    for chord, (r, c) in _pair_index().items():
        for ri, ci in zip(r, c):
            rows.append({'sn'        : sn,
                         **labels,
                         'chord'     : chord,
                         'pair'      : '-'.join(sorted([str(chords[ri]), str(chords[ci])])),
                         'crossnobis': D[ri, ci],
                         'cosine'    : 1 - cos[ri, ci],
                         #'theta'     : np.arccos(cos[ri, ci])
                         })
    return rows


def _add_group_reference(df, keys, ref_session=3, crossval=False):
    """Attach the reference-session group geometry.

    """

    # group-mean geometry: across-subject mean of each chord pair in the reference
    # session, pooled over trained/untrained/mixed (no 'chord'/'session' in the key,
    # so all subjects contribute). Merged onto every row, so the *_group columns hold
    # the ref-session reference for every session.
    s_ref = df[df.session == ref_session]
    if crossval:
        # leave-one-subject-out: subtract each subject's own value from the pair sum,
        # so their reference is the mean over the other subjects. Keyed by subject too,
        # then merged so it broadcasts across that subject's sessions.
        ref = s_ref[keys + ['sn']].copy()
        for metric in ('crossnobis', 'cosine'):
            g = s_ref.groupby(keys)[metric]
            ref[f'{metric}_group'] = (g.transform('sum') - s_ref[metric]) / (g.transform('size') - 1)
        df = df.merge(ref, on=keys + ['sn'], how='left')
    else:
        ref = (s_ref.groupby(keys, as_index=False)
                 .agg(crossnobis_group=('crossnobis', 'mean'),
                      cosine_group    =('cosine',     'mean')))
        df  = df.merge(ref, on=keys, how='left')

    df['theta_group'] = np.arccos(df.cosine_group)

    return df


def make_rois_distance_dataframe(glm=3, atlas_name='ROI', sns=None, ref_session=3, crossval=False):

    sns  = gl.participants if sns is None else sns
    rois = gl.rois[atlas_name]

    rows = []
    for H, roi, sess in itertools.product(gl.Hem, rois, gl.sessions):
        for sn in sns:

            print(f'doing participant {sn}, session {sess}, {H}, {roi}...')

            G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs_raw.within_session.{sess}.glm{glm}.{H}.{roi}.npy'))
            rows += _G_rows(G, sn, Hem=H, roi=roi, session=sess)

    df = pd.DataFrame(rows)
    df = _add_group_reference(df, ['Hem', 'roi', 'pair'], ref_session, crossval)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def make_force_distance_dataframe(metrics=('raw', 'abs', 'der'), sns=gl.participants, ref_session=3, crossval=False):
    """The neural dataframe's counterpart over the force Gs written by pattern_G_matrix.

    Same rows, same columns, with ``metric`` (the force measure) standing in for
    ``Hem``/``roi``: the force Gs have the identical 8x8 trained-first layout, so
    every chord pair lines up one-to-one with its neural row.
    """
    sns = gl.participants if sns is None else sns

    rows = []
    for metric, sess in itertools.product(metrics, gl.sessions):
        for sn in sns:

            print(f'doing participant {sn}, session {sess}, force {metric}...')

            G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs_raw.within_session.{sess}.force.{metric}.npy'))
            rows += _G_rows(G, sn, metric=metric, session=sess)

    df = pd.DataFrame(rows)
    df = _add_group_reference(df, ['metric', 'pair'], ref_session, crossval)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t', index=False)


def _ancova_beta(df, metric):
    """Trained - untrained difference in ``metric``, adjusted for the group geometry.

    Returns (slope_group, slope_chord, intercept).
    """
    y = df[metric].to_numpy()
    g = df[f'{metric}_group'].to_numpy()
    c = df.chord.map({'trained'  : 1, 'untrained': -1}).to_numpy()
    X = np.c_[g, c, np.ones(len(df))]
    B = np.linalg.pinv(X) @ y

    return B[0], B[1], B[2]


def make_ancova_dataframe(glm=3, atlas_name='ROI', rois=None, sns=gl.participants, metrics=('crossnobis', 'cosine')):

    """Trained vs untrained, with the reference-session geometry regressed out.

    Which chords a participant trained is counterbalanced, but the chord pairs
    themselves are not equally distinct to begin with -- pairs that share no fingers
    sit further apart in every participant -- so whichever four chords a participant
    happened to be given carries an intrinsic trained-untrained difference that has
    nothing to do with training. ``*_group`` (the across-participant mean of that pair
    in ``ref_session``, pooled over training status, so it is training-neutral) is that
    intrinsic geometry, and this is the ANCOVA that takes it out.

    One row per chord pair, as in the dissimilarity tsv it is built from: each metric
    column holds that pair's dissimilarity with the group geometry regressed out, so
    ``crossnobis``/``cosine`` here are the adjusted values, not the raw ones. The fit
    itself is one per participant x Hem x roi x session x metric, and its
    ``{metric}_slope_chord`` (the adjusted trained-untrained difference, to be tested
    against zero across participants), ``{metric}_slope_group`` and
    ``{metric}_intercept`` are repeated across that cell's rows.

    The reference is rebuilt from the tsv rather than taken from its ``*_group``
    columns, so ``crossval``/``ref_session`` can be changed without recomputing the Gs.
    ``crossval=True`` (leave-one-participant-out) is the default here: with the
    participant inside their own reference, the covariate shares noise with y, which
    inflates the slope in ``ref_session`` only -- exactly the session the across-session
    comparison hangs on.

    """
    if rois is None:
        rois = gl.rois[atlas_name]

    df = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t')
    df = df[df.chord.isin(['trained', 'untrained'])]

    cells = []
    for sn, H, roi, session in itertools.product(sns, gl.Hem, rois, gl.sessions):

        print(f'ancova, participant {sn}, {H}, {roi}, session {session}...')

        cell = df[(df.sn==sn) & (df.Hem == H) & (df.roi == roi) & (df.session == session)]

        adjusted = cell[['sn', 'Hem', 'roi', 'session', 'chord', 'pair']].copy()
        for metric in metrics:

            slope_group, slope_chord, intercept = _ancova_beta(cell, metric)

            # the pair's own dissimilarity with the group geometry taken out, centred on
            # the cell's mean covariate so the adjusted values stay on the scale of y
            g = cell[f'{metric}_group']
            adjusted[metric]                  = cell[metric] - slope_group * (g - g.mean())
            adjusted[f'{metric}_slope_chord'] = slope_chord
            adjusted[f'{metric}_slope_group'] = slope_group
            adjusted[f'{metric}_intercept']   = intercept

        cells.append(adjusted)

    df_ancova = pd.concat(cells, ignore_index=True)

    df_ancova.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_ancova.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def _make_fname(session, repetition):
    """The 'within_session.3.1' / 'across_session' part of a G filename."""
    fname = 'across_session' if session == 'all' else 'within_session'
    fname = fname + '.' + str(session) if session != 'all' else fname
    fname = fname + '.' + str(repetition) if repetition != 'all' else fname
    return fname  


def calc_G_rois(sns=gl.participants, glm=3, sessions=gl.sessions, repetitions=['all']):
    """G_rois: build the betas loader, then the ROI Gs."""
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm)
    for data in loader:
        for session in sessions:
            for repetition in repetitions:

                print(f'rois, doing participant {data.sn}...')

                G        = G_matrix.calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=False)
                cov      = G_matrix.calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=True)
                G_raw    = G_matrix.calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=False, fixed_effect=False)
                G_noxval = G_matrix.calc_G(data.betas, data.cond_vec, data.part_vec, session, repetition=repetition, centred=False, fixed_effect=False, crossval=False)

                save_path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
                os.makedirs(save_path, exist_ok=True)

                fname = _make_fname(session, repetition)

                np.save(os.path.join(save_path, f'G_obs.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G)
                np.save(os.path.join(save_path, f'cov.{fname}.glm{glm}.{data.Hem}.{data.roi}'), cov)
                np.save(os.path.join(save_path, f'G_obs_raw.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G_raw)
                np.save(os.path.join(save_path, f'G_obs_noxval.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G_noxval)


def calc_G_force(sns=gl.participants, metrics=('raw', 'abs', 'der'), sessions=('all', *gl.sessions), repetitions=('all', 1, 2)):
    """The same Gs as calc_G_rois, but over the five fingers' force instead of voxels.

    Fingers take the place of the voxels, so the matrices come out with the same
    layout as the ROI ones — 8x8 within a session, 24x24 across, trained chords
    first — and land next to them in the pcm directory as
    ``G_obs.<epoch>.force.<metric>.npy``.
    """
    force = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, 'force.run.wide.tsv'), sep='\t')

    for metric in metrics:
        for sn in sns:
            for session in sessions:
                for repetition in repetitions:

                    print(f'force {metric}, doing participant {sn}...')

                    data, cond_vec, part_vec = behaviour.force_patterns(force, sn, metric, session=session, repetition=repetition)

                    G        = G_matrix.calc_G(data, cond_vec, part_vec, centred=False)
                    cov      = G_matrix.calc_G(data, cond_vec, part_vec, centred=True)
                    G_raw    = G_matrix.calc_G(data, cond_vec, part_vec, centred=False, fixed_effect=False)
                    G_noxval = G_matrix.calc_G(data, cond_vec, part_vec, centred=False, fixed_effect=False, crossval=False)

                    save_path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}')
                    os.makedirs(save_path, exist_ok=True)

                    fname = _make_fname(session, repetition)

                    np.save(os.path.join(save_path, f'G_obs.{fname}.force.{metric}'), G)
                    np.save(os.path.join(save_path, f'cov.{fname}.force.{metric}'), cov)
                    np.save(os.path.join(save_path, f'G_obs_raw.{fname}.force.{metric}'), G_raw)
                    np.save(os.path.join(save_path, f'G_obs_noxval.{fname}.force.{metric}'), G_noxval)


def correlation_between_sessions(sns=gl.participants, glm=3, Hem=('L', 'R'), atlas_name='ROI', residual_fname='residual.dtseries.nii'):
    """Between-session correlation per (Hem, ROI, session-pair, chord set).

    Writes two tsvs to ``path_pcm``:
      - ``MLE_correlation`` : the PCM correlation-model fit (r_group/r_indiv + the
        model's own SNR).
      - ``xval_correlation``: the cross-validated cosine on voxel-centred betas
        (matched-chord diagonal of the across-session block -> Pearson-style r),
        plus the diag(G)/diag(Sig) SNR. The (n_subj, 8, 8) cov of each group is
        also saved as ``cov.corr_across_sess.*.npy``.
    """
    path_pcm = os.path.join(gl.baseDir, gl.pcmDir)
    Mflex    = pcm.CorrelationModel("flex", num_items=4, corr=None, cond_effect=True)
    corrs    = list(itertools.combinations(gl.sessions, 2))   # real session numbers, e.g. (3, 9)
    chords   = ['trained', 'untrained']

    correlation  = pcm_.CorrelationBetweenSessions(glm, sns=sns, atlas_name=atlas_name, Hem=Hem, residual_fname=residual_fname)
    Y, cov, snr  = correlation.group_datasets(corrs, chords)

    mle, xval = [], []
    for (H, roi, sessions, chord), Y_ in Y.items():

        spair = f'{sessions[0]}-{sessions[1]}'

        # PCM correlation model (MLE) -- keeps the model's own SNR
        r_indiv, r_group, SNR = pcm_.fit_correlation(Y_, Mflex)
        mle.append(pd.DataFrame({'sn'     : sns,
                                 'r_group': r_group,
                                 'r_indiv': r_indiv,
                                 'SNR'    : SNR,
                                 'chord'  : chord,
                                 'corr'   : spair,
                                 'roi'    : roi,
                                 'Hem'    : H,}))

        # cross-validated cosine -- matched-chord diagonal + diag(G)/diag(Sig) SNR
        G_group = cov[H, roi, sessions, chord]
        np.save(os.path.join(path_pcm, f'cov.corr_across_sess.glm{glm}.{spair}.{chord}.{H}.{roi}.npy'), G_group)

        r_xval = pcm.G_to_cosine(G_group)
        r_avg  = np.diagonal(r_xval[:, :4, 4:], axis1=1, axis2=2).mean(axis=1)
        xval.append(pd.DataFrame({'sn'     : sns,
                                  'r_indiv': r_avg,
                                  'r_group': r_avg.mean(),
                                  'SNR'    : snr[H, roi, sessions, chord],
                                  'corr'   : spair,
                                  'chord'  : chord,
                                  'roi'    : roi,
                                  'Hem'    : H}))

    pd.concat(mle, ignore_index=True).to_csv(os.path.join(path_pcm, f'MLE_correlation.{atlas_name}.glm{glm}.tsv'),  sep='\t', index=False)
    pd.concat(xval, ignore_index=True).to_csv(os.path.join(path_pcm, f'xval_correlation.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def _rdm_vectors(sns, H, roi, session, glm, prefix='G_obs_raw', order=None):
    """Vectorised dissimilarity matrix of every participant, on a common chord order.

    Each participant's G is stored in their own trained-first order, so it is put
    through ``G_sorted`` first: without that, entry k of one participant's vector
    is a different chord pair than entry k of the next, and the correlations below
    are meaningless. ``order`` defaults to ``gl.chordID`` (line participants up by
    chord identity); pass a trained/untrained framing to align by training status.

    Returns an (n_subj, n_pairs) array, the lower triangle of each participant's RDM.
    """
    mask = np.tri(len(gl.chordID), k=-1, dtype=bool)

    rdms = []
    for sn in sns:
        G   = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'{prefix}.within_session.{session}.glm{glm}.{H}.{roi}.npy'))
        G_s = G_matrix.G_sorted(G, sn, order=order)
        D   = pcm.G_to_dist(G_s)
        rdms.append(D[mask])

    return np.array(rdms)


def _corr(a, b, method='pearson'):
    """Correlation between two vectorised RDMs."""
    if method == 'spearman':
        return spearmanr(a, b)[0]
    return np.corrcoef(a, b)[0, 1]


def _noise_ceiling(rdms, method='pearson'):
    """Leave-one-participant-out lower bound and the upper bound of the RSA noise ceiling.

    lower: each participant's RDM against the mean of the *other* participants. The
           group mean is estimated without them, so this is what a model that
           generalises across participants can be expected to reach.
    upper: the same, but against the mean of *all* participants, that participant
           included. The reference is fitted to the participant it is scored on, so
           it overestimates, and no model should beat it.

    Returns lower, upper: (n_subj,) arrays, one correlation per participant.
    """
    n     = len(rdms)
    if n < 3:
        raise ValueError(f'need at least 3 participants for a leave-one-out ceiling, got {n}')
    total = rdms.sum(axis=0)

    lower, upper = [], []
    for rdm in rdms:
        others_mean = (total - rdm) / (n - 1)  # group mean leaving this participant out
        group_mean  = total / n                # group mean including this participant

        lower.append(_corr(rdm, others_mean, method))
        upper.append(_corr(rdm, group_mean, method))

    return np.array(lower), np.array(upper)


def make_noise_ceiling_dataframe(glm=3, atlas_name='ROI', sns=None, prefix='G_obs_raw',
                                 method='pearson', order=None):
    """Noise ceiling of the crossnobis RDM for every hemisphere, roi and session.

    One row per participant, so the ceiling of a given roi/Hem/session is the mean
    of its rows (``df.groupby(['Hem', 'roi', 'session'])[['lower', 'upper']].mean()``).
    Writes ``noise_ceiling.within_session.<atlas_name>.glm<glm>.tsv`` to the pcm dir.
    """
    sns  = gl.participants if sns is None else sns
    rois = gl.rois[atlas_name]

    rows = []
    for H, roi, session in itertools.product(gl.Hem, rois, gl.sessions):

        print(f'doing {H}, {roi}, session {session}...')

        rdms         = _rdm_vectors(sns, H, roi, session, glm, prefix=prefix, order=order)
        lower, upper = _noise_ceiling(rdms, method=method)

        for i, sn in enumerate(sns):
            rows.append({'sn'     : sn,
                         'Hem'    : H,
                         'roi'    : roi,
                         'session': session,
                         'lower'  : lower[i],
                         'upper'  : upper[i]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'noise_ceiling.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def _model_rdm_vectors(sn, glm=3, order=None, Hem=None, roi=None, force=False):
    """Vectorised RDM of every model in ``pcm.model_Gs``, for one participant.

    Built on the same chord ``order`` and put through the same ``G_to_dist`` and lower
    triangle as :func:`_rdm_vectors`, so entry k of a model vector is the chord pair
    that entry k of the data vector is. The models are per participant because
    ``type``, ``trained`` and ``untrained`` depend on which chords that participant
    trained -- on a common order those land in different slots for each of them.

    Returns a dict model name -> (n_pairs,) vector.
    """
    mask = np.tri(len(gl.chordID), k=-1, dtype=bool)
    Gs   = pcm_.model_Gs(sn, glm=glm, Hem=Hem, roi=roi, order=order, force=force)
    return {name: pcm.G_to_dist(G)[mask] for name, G in Gs.items()}


def make_model_correlation_dataframe(sns=gl.participants, glm=3, atlas_name='ROI', rois=None, prefix='G_obs_raw',
                                     method='pearson', order=None, include_base=False):
    """RSA: each participant's crossnobis RDM against every model RDM of ``pcm.model_Gs``.

    One row per participant x Hem x roi x session x model, carrying the correlation ``r``
    and that cell's noise ceiling (``lower``, ``upper``) in the same columns -- the same
    ``method`` scores model and ceiling, which is the only way the two are comparable, so
    a model can be plotted straight against the ceiling band without a second merge.

    ``include_base`` adds the group-mean observed G of session 3 (``pcm.base_model``) as
    a model. It is off by default: it is the data's own group mean, so it is close to the
    upper bound of the noise ceiling by construction rather than a prediction about the
    geometry -- in session 3 it *is* that bound.

    Note that ``trained`` and ``untrained`` are redundant here, and come out as exact
    mirror images of each other: their two Gs sum to the centring matrix, whose RDM is
    flat, so their RDMs sum to a constant and correlate -1. The component model tells
    them apart because it weights them non-negatively and separately; a correlation
    cannot. Read the pair as one axis, 'which set is the more spread out'.

    Writes ``model_correlation.within_session.<atlas_name>.glm<glm>.tsv`` to the pcm dir.
    """
    if rois is none:
        rois  = gl.rois[atlas_name]
    order = gl.chordID if order is None else order

    # roi-independent, so built once per participant rather than once per cell
    models = {sn: _model_rdm_vectors(sn, glm=glm, order=order) for sn in sns}

    rows = []
    for H, roi, session in itertools.product(gl.Hem, rois, gl.sessions):

        print(f'doing {H}, {roi}, session {session}...')

        # calculate noise ceiling (rdms in common order)
        rdms         = _rdm_vectors(sns, H, roi, session, glm, prefix=prefix, order=order)
        lower, upper = _noise_ceiling(rdms, method=method)

        for i, sn in enumerate(sns):
            for name, model_rdm in model_rdms.items():
                rows.append({'sn'     : sn,
                             'Hem'    : H,
                             'roi'    : roi,
                             'session': session,
                             'model'  : name,
                             'r'      : _corr(rdms[i], model_rdm, method),
                             'lower'  : lower[i],
                             'upper'  : upper[i]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_correlation.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def make_scaling_dataframe(sns=None, glm=3, atlas_name='ROI', ref_session=3, prefix='G_obs_raw'):
    """How much of each session's geometry is a pure rescaling of the reference session's.

    For every participant x hemisphere x roi x session, ``G_scaling`` predicts the
    session's mean dissimilarity from the reference session's, scaled by the ratio of
    the two sessions' mean activity. Its ``residual`` column (observed - predicted) is
    what says whether the geometry changed beyond that change in overall activity.

    Run three times per cell: over all 8 chords, and over the trained and the untrained
    4x4 block on their own (the Gs are stored trained-first, so the blocks are just the
    leading and trailing quarters). Writes one row per cell x chord set to
    ``scaling.between_session.glm<glm>.<atlas_name>.tsv`` in the pcm dir.
    """
    sns    = gl.participants if sns is None else sns
    rois   = gl.rois[atlas_name]
    blocks = {'all': slice(None), 'trained': slice(0, 4), 'untrained': slice(4, None)}

    rows = []
    for session, sn, H, roi in itertools.product(gl.sessions, sns, gl.Hem, rois):

        print(f'doing participant {sn}, session {session}, {H}, {roi}...')

        G_ref = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'{prefix}.within_session.{ref_session}.glm{glm}.{H}.{roi}.npy'))
        G_tar = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'{prefix}.within_session.{session}.glm{glm}.{H}.{roi}.npy'))

        for chord, b in blocks.items():
            G_scaled = G_matrix.G_scaling(G_ref[b, b], G_tar[b, b])
            rows.append(G_scaled.assign(chord=chord, session=session, Hem=H, roi=roi, sn=sn))

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'scaling.between_session.glm{glm}.{atlas_name}.tsv'), sep='\t', index=False)


def fit_component_model_rois(sns=gl.participants, glm=3, atlas_name='ROI', residual_fname='residual.dtseries.nii'):
    """Fit the PCM component model per (subject, Hem, roi, session).

    Prewhitens the betas once per (subject, roi) with BetasPrewithenedLoader, then
    fits the component model (which includes 'base', the group-mean observed G of
    session 3) and pickles each fit's ``theta_in`` next to the participant's Gs, as
    ``component_model.theta_in.<atlas_name>.glm<glm>.<session>.<Hem>.<roi>.p`` --
    which component_summary reads back.
    """
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    pcm_.fit_component_model(loader)


def make_component_weight_dataframe(sns=None, glm=3, atlas_name='ROI'):
    """Collect the component weights fitted by component_fit into one dataframe.

    Reads each pickled ``theta_in``, takes the component model's theta (the other
    models' thetas have different lengths, so the list cannot be arrayed as a whole),
    exponentiates it to weights -- one per component -- and writes one long-form row
    per component to ``component_model.<atlas_name>.glm<glm>.tsv`` in the pcm dir.

    The component names are the ones the fit itself stored next to its theta, so a
    component list that has changed since the fit cannot shift the weights onto the
    wrong names -- a pickle from before the names were stored raises instead.
    """
    sns = gl.participants if sns is None else sns

    df = pd.DataFrame()
    for sn, session, H, roi in itertools.product(sns, gl.sessions, gl.Hem, gl.rois[atlas_name]):

        path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'component_model.theta_in.{atlas_name}.glm{glm}.{session}.{H}.{roi}.p')
        with open(path, 'rb') as f:
            fit = pickle.load(f)

        if not isinstance(fit, dict):
            raise ValueError(f'{path} was fitted before the component names were stored with the theta, '
                             f'so which component each weight belongs to cannot be known -- rerun the '
                             f'fit_component pass')

        comp_names = fit['comp_names']

        # fit['theta'] holds one theta per model and they have different lengths, so it cannot
        # be arrayed as a whole -- take the component model's: one log weight per component,
        # followed by the log scale (only if the fit had fit_scale=True) and the log noise,
        # which is why the components are sliced by name count and not by dropping the tail
        weight = np.exp(np.array(fit['theta'][0][:len(comp_names)]))

        df_tmp = pd.DataFrame({'weight': weight.squeeze(), 'component': comp_names})
        df_tmp['sn']      = sn
        df_tmp['Hem']     = H
        df_tmp['roi']     = roi
        df_tmp['session'] = session
        df = pd.concat([df, df_tmp], ignore_index=True)

    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'component_model.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def make_likelihood_dataframe(sns=gl.participants, glm=3, atlas_name='ROI'):

    rois = gl.rois[atlas_name]

    LL = pd.DataFrame()
    for sn, session, H, roi in itertools.product(sns, gl.sessions, gl.Hem, rois):
        path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'component_model.T_in.{atlas_name}.glm{glm}.{session}.{H}.{roi}.p')
        with open(path, 'rb') as f:
            T_in = pickle.load(f)
        
        ll            = T_in.likelihood
        ll['sn']      = sn
        ll['session'] = session
        ll['Hem']     = H
        ll['roi']     = roi

        LL = pd.concat([LL, ll])

    LL.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'likelihood.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


# Step name -> function, grouped by what the step produces: the Gs, then the PCM fits
# over them, then the tsvs collected from either.
FUNC = {
    'G_rois'                        : calc_G_rois,
    'G_force'                       : calc_G_force,
    'fit_correlation'               : correlation_between_sessions,
    'fit_component'                 : fit_component_model_rois,
    'dataframe_distance_rois'       : make_rois_distance_dataframe,
    'dataframe_distance_force'      : make_force_distance_dataframe,
    'dataframe_ancova'              : make_ancova_dataframe,
    'dataframe_noise_ceiling'       : make_noise_ceiling_dataframe,
    'dataframe_model_correlation'   : make_model_correlation_dataframe,
    'dataframe_scaling'             : make_scaling_dataframe,
    'dataframe_component_weight'    : make_component_weight_dataframe,
    'dataframe_component_likelihood': make_likelihood_dataframe,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`glm=`, `metrics=`, `repetitions=`, ...), but only
    the ones it accepts.
    """

    if what is not None:
        func = FUNC[what] # select function
        accepted = inspect.signature(func).parameters # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted}) # run the function

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate the second moment matrices of the neural and force patterns.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: G_rois for subj116)')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas come from (default: the step default, 3)')
    parser.add_argument('--sns', nargs='+', type=int, default=gl.participants, help='participant ids to include in the analysis')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        calc_G_rois(sns=[116])

