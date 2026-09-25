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
import EFC_learningfMRI.betas as betas
import EFC_learningfMRI.pcm as pcm_

def make_rois_distance_dataframe(glm=3, atlas_name='ROI', sns=None, ref_session=3, crossval=False):

    sns  = gl.participants if sns is None else sns
    rois = gl.rois[atlas_name]

    rows = []
    for H, roi, sess in itertools.product(gl.Hem, rois, gl.sessions):
        for sn in sns:

            print(f'doing participant {sn}, session {sess}, {H}, {roi}...')

            G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs_raw.within_session.{sess}.glm{glm}.{H}.{roi}.npy'))
            rows += G_matrix.G_rows(G, sn, Hem=H, roi=roi, session=sess)

    df = pd.DataFrame(rows)
    df = G_matrix.add_group_reference(df, ['Hem', 'roi', 'pair'], ref_session, crossval)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


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

                fname = G_matrix.make_fname(session, repetition)

                np.save(os.path.join(save_path, f'G_obs.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G)
                np.save(os.path.join(save_path, f'cov.{fname}.glm{glm}.{data.Hem}.{data.roi}'), cov)
                np.save(os.path.join(save_path, f'G_obs_raw.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G_raw)
                np.save(os.path.join(save_path, f'G_obs_noxval.{fname}.glm{glm}.{data.Hem}.{data.roi}'), G_noxval)


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
    'fit_component'                 : fit_component_model_rois,
    'dataframe_distance_rois'       : make_rois_distance_dataframe,
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
    parser = argparse.ArgumentParser(description='Calculate the second moment matrices of the neural patterns and summarise their geometry.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: G_rois for subj116)')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas come from (default: the step default, 3)')
    parser.add_argument('--sns', nargs='+', type=int, default=gl.participants, help='participant ids to include in the analysis')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        calc_G_rois(sns=[116])

