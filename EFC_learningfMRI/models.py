import os
import functools
import numpy as np
import pandas as pd
import PcmPy as pcm
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.util as util
import EFC_learningfMRI.G_matrix as G_matrix
import EFC_learningfMRI.model_from_spencer as spencer
import EFC_learningfMRI.emg as emg

C = pcm.centering(8)


@functools.lru_cache(maxsize=None)
def _read_behav(fname):
    """A tsv in behavDir, read once per Python session and then served from memory.

    make_G_dict runs for every hemisphere x ROI x participant, and re-parsing the large
    trial tsvs from the drive each time made it ~20x slower. Callers must not modify the
    returned DataFrame in place (filtering with df[...] makes a copy, so that is fine).
    A tsv regenerated mid-session is not picked up until the kernel restarts, or
    ``_read_behav.cache_clear()``.
    """
    return pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, fname), sep='\t')


FINGER = {
     21911: np.array([1, 1, 0, 1, 1]),
     92122: np.array([0, 1, 1, 1, 1]),
     91211: np.array([0, 1, 1, 1, 1]),
     22911: np.array([1, 1, 0, 1, 1]),
     21291: np.array([1, 1, 1, 0, 1]),
     12129: np.array([1, 1, 1, 1, 0]),
     12291: np.array([1, 1, 1, 0, 1]),
     11911: np.array([1, 1, 0, 1, 1])
}

PATTERN = {
     21911: np.array([-1,  1,  0,  1,  1]),
     92122: np.array([ 0, -1,  1, -1, -1]),
     91211: np.array([ 0,  1, -1,  1,  1]),
     22911: np.array([-1, -1,  0,  1,  1]),
     21291: np.array([-1,  1, -1,  0,  1]),
     12129: np.array([ 1, -1,  1, -1,  0]),
     12291: np.array([ 1, -1, -1,  0,  1]),
     11911: np.array([ 1,  1,  0,  1,  1])
}

FLEXION = {
    21911: 1,
    92122: 3,
    91211: 1,
    22911: 2,
    21291: 2,
    12129: 2,
    12291: 2,
    11911: 0
}



def normalise(G):
    """Symmetrise, centre and trace-normalise a model G, so every model is on the same scale.

    Crossvalidated Gs are only symmetric on average, hence the symmetrisation; on an
    already symmetric, centred G both steps do nothing. ``what`` names the model in the
    error raised when the trace is not positive -- dividing by a negative trace would
    silently flip the model's sign.
    """
    G  = C @ ((G + G.T) / 2) @ C
    tr = np.trace(G)
    if tr <= 0:
        raise ValueError(f'G trace {tr:.3g}, so it cannot be trace-normalised')
    return G / tr



def fixed(is_trained):
    """
    is_trained: (8,) bool, True in the slots of the subject's trained chords
    """
    tr   = np.asarray(is_trained).astype(float)
    untr = 1 - tr

    # trained untrained
    v_type = untr - tr
    G_type = normalise(np.outer(v_type, v_type))

    # trained
    G_tr = normalise(np.diag(tr))

    # untrained
    G_untr = normalise(np.diag(untr))

    return G_type, G_tr, G_untr


# def subj_spec_models(order=None, glm=3):

#     """
#     order: list of chords in the order they appear in the G matrices
#     """

#     finger  = np.zeros((8, 5))
#     pattern = np.zeros_like(finger)
#     flexion = np.zeros(8)
#     for i, ch in enumerate(order):
#         flexion[i] = FLEXION[ch]
#         finger[i]  = FINGER[ch]
#         pattern[i] = PATTERN[ch]

#     G_finger  = normalise(finger @ finger.T, 'finger')
#     G_pattern = normalise(pattern @ pattern.T, 'pattern')
#     G_flexion = normalise(np.outer(flexion, flexion), 'flexion')

#     return G_finger, G_pattern, G_flexion


def group_geometry(glm, Hem, roi, order=None, sns=gl.participants, session=3, crossval=True, sn=None):
    """Group-mean observed G of one ROI: the empirical baseline geometry.

    The crossvalidated Gs that :func:`scripts.pattern.calc_G_rois` writes per subject,
    averaged over ``sns``. Each is stored in *that* subject's chord order (trained
    first), so ``G_sorted`` puts every one of them on ``order`` **before** they are
    averaged: slot *i* is then one physical chord, and what survives the average is the
    geometry of the chords themselves. Averaged as stored, a different chord would sit
    in slot *i* for every subject and, the trained sets being counterbalanced, that
    geometry would wash out -- the mean of the raw slots comes out 0.93 cosine with a
    uniform G, i.e. little more than overall dissimilarity.

    Pass the chord order of the subject being fitted (``order``), which is the order its
    data and its other components are in; it defaults to the common ``gl.chordID``.

    As a component this asks how much of a session's geometry is already there in the
    average geometry of ``session``, so the structural components are fitted on what
    that baseline leaves over. With ``crossval`` (the default) the baseline is
    leave-one-subject-out: ``sn``, the subject being fitted, is dropped from ``sns``, so
    the baseline is independent of its data -- as ``G_matrix.add_group_reference`` does
    for the distances. Without it the subject is *in* the average, which favours
    ``session`` (its own data are in the baseline) over the other sessions.

    Reads 16 small .npy per call, ~46 ms, so ~30 s over a whole fit run: not worth
    caching, and a cache would have to hash ``order``.
    """
    order = gl.chordID if order is None else order

    if crossval:
        if sn is None:
            raise ValueError('crossval=True needs sn, the subject to leave out')
        sns = [s for s in sns if s != sn]

    Gs = []
    for s in sns:
        fname = f'G_obs.within_session.{session}.glm{glm}.{Hem}.{roi}.npy'
        G     = np.load(os.path.join(gl.baseDir, gl.geomDir, f'subj{s}', fname))
        G     = G_matrix.G_sorted(G, s, order=order)      # off this subject's own chord order, onto `order`
        Gs.append(G)

    Gs = np.array(Gs)

    return normalise(np.mean(Gs, axis=0))


def single_finger(Hem, roi, order=None, glm=3, sns=spencer.participants):
    """Group-mean G of the chord patterns predicted from Arbuckle's single-finger data, in one ROI.

    The crossvalidated Gs that scripts/model_from_spencer.calc_G_rois writes per Arbuckle participant,
    averaged over ``sns``, centred, then divided by its trace (so the centred G has trace 1).
    Those Gs are in sorted chordID order (``gl.chordID``), so they are put on ``order``, the chord
    order of the subject being fitted (default ``gl.chordID``), without ``G_sorted``, which reads
    the input order off a subject's trained set.
    """
    order = gl.chordID if order is None else order

    Gs = [np.load(os.path.join(gl.baseDir, gl.arbuckleDir, 'pcm', s, f'G_obs_raw.glm{glm}.{Hem}.{roi}.npy')) for s in sns]
    Gs = np.array([pcm.G_to_cosine(G) for G in Gs])
    G  = normalise(np.mean(Gs, axis=0))

    perm = [list(gl.chordID).index(chord) for chord in order]       # source index for each target slot
    return G[np.ix_(perm, perm)]


def hand_use(order=None):
    """G of the chords predicted from the natural statistics of hand use (glove data).

    The G that scripts/spencer.calc_G_natural writes in sorted chordID order (``gl.chordID``),
    normalised and put on ``order``, the chord order of the subject being fitted (default
    ``gl.chordID``), as :func:`single_finger`.
    """
    order = gl.chordID if order is None else order

    G = np.load(os.path.join(gl.baseDir, gl.arbuckleDir, gl.naturalDir, 'G_natural.npy'))
    G = normalise(G)

    perm = [list(gl.chordID).index(chord) for chord in order]       # source index for each target slot
    return G[np.ix_(perm, perm)]


def muscle(order=None, session=1, sns=emg.participants):
    """Group-mean G of the chords' muscle (EMG) patterns, trace-normalised.

    The Gs that scripts/emg.calc_G_emg writes per EFC_learningEMG participant (the ten EMG
    channels as voxels) on day ``session``, put on ``order`` (default ``gl.chordID``) and
    averaged over ``sns``, then centred and divided by the trace (so the centred G has trace 1),
    as :func:`single_finger`. Each participant's G is in its own trained-first order, read from
    EFC_learningEMG/participants.tsv, so ``G_sorted`` puts it on ``order`` before averaging.
    """
    order = gl.chordID if order is None else order

    fname = f'G_obs_raw.within_session.{session}.emg.npy'
    Gs    = np.array([np.load(os.path.join(gl.baseDir, gl.emgDir, 'pcm', f'subj{s}', fname)) for s in sns])
    Gs    = G_matrix.G_sorted(Gs, sns, order=order, experiment=gl.emgDir) # off each participant's trained-first order, onto `order`
    Gs    = np.array([pcm.G_to_cosine(G) for G in Gs])   # unit-length pattern per chord, per participant

    return normalise(Gs.mean(axis=0))


def force(sn, metric, order, ref_session=3, sns=gl.participants):
    """Group-mean G of the force patterns (fingers as voxels) in ``ref_session``, leaving out ``sn``.

    The Gs that scripts/behaviour.calc_G_force writes per participant, each in that
    participant's trained-first order, so ``G_sorted`` puts every one on ``order``
    **before** they are averaged (see :func:`group_geometry`). The average is over
    ``sns`` without ``sn``, the subject being fitted, so the model doesn't contain that
    subject's own behaviour. A fixed ``ref_session`` keeps the model the same across the
    sessions it is compared with.
    """
    others = [s for s in sns if s != sn]
    fname  = f'G_obs_raw.within_session.{ref_session}.force.{metric}.npy'
    Gs     = np.array([np.load(os.path.join(gl.baseDir, gl.geomDir, f'subj{s}', fname)) for s in others])
    Gs     = G_matrix.G_sorted(Gs, others, order=order)    # off each participant's trained-first order, onto `order`
    Gs     = np.array([pcm.G_to_cosine(G) for G in Gs])   # unit-length pattern per chord, per participant

    return normalise(Gs.mean(axis=0))


def force_flexion_model(sn, order, ref_session=3):
    """Total flexion force per chord in ``ref_session``, as a one-dimensional G.

    Flexion is negative in the raw force, so v is the summed negative part over fingers.
    Every chord has four instructed fingers, so this is close to minus the total
    extension force and the two give nearly the same G.
    """
    force = _read_behav('force.run.wide.tsv')
    cols  = [f'{finger}_raw' for finger in gl.fingers]

    df = force[(force.subNum == sn) & (force.session == ref_session)]
    grouped = df.groupby('chordID')         # split the rows by chord (20 per chord: one per block and repetition)
    grouped = grouped[cols]                 # keep only the five fingers' raw force
    F       = grouped.mean()                # average each finger over a chord's rows -> (8 chords x 5 fingers), index = chordID, sorted
    F       = F.loc[order]                  # reorder the rows to `order`, picking them by chordID (KeyError if a chord is missing)
    F       = F.to_numpy()                  # DataFrame -> plain (8, 5) array, row i = chord order[i]
    v       = np.clip(-F, 0, None).sum(axis=1)

    return normalise(np.outer(v, v))


def performance_model(sn, order, measure='MD', fname='behaviour.trial.tsv'):
    """One-dimensional G of a behavioural feature per chord, from the other participants.

    ``measure`` is averaged per chord over the session-4 trials in ``fname`` of every
    participant except ``sn``, the subject being fitted, so the model doesn't contain
    that subject's own behaviour (as :func:`force`).
    """

    df = _read_behav(fname)
    df = df[(df.session == 4) & (df.subNum != sn)]

    values    = pd.to_numeric(df[measure], errors='coerce')
    per_chord = values.groupby(df['chordID']).mean()
    v         = per_chord.loc[order].to_numpy()

    return normalise(np.outer(v, v))


def make_G_dict(sn, order=None, Hem='L', roi='M1', glm=3, ref_session=3, emg_session=1):
    """Second moment matrix of every model, keyed by name, for one subject.

    The one place the model set is defined: :func:`make_models` fits these, and
    ``scripts.pattern.make_model_correlation_dataframe`` correlates the same matrices
    with the observed RDMs, so the two cannot drift apart.

    ``order`` is the chord order of the matrices. It defaults to the subject's own
    trained-first order, the order its data are in; pass ``gl.chordID`` to line every
    subject up slot by slot, as the RSA does. ``type``, ``trained`` and ``untrained`` are
    built from the subject's trained set, so they are right in either order.

    The models (keys of the returned dict):
        identity                         every chord equally distinct from every other (``np.eye(8)``)
        type, trained, untrained         which chords this subject trained (:func:`fixed`)
        mean deviation, execution time   the other subjects' mean MD and ET per chord in session 4, before
                                         training (:func:`performance_model`)
        single finger                    Arbuckle's single-finger patterns in ``Hem`` ``roi``, summed per
                                         chord (:func:`single_finger`)
        muscle pattern                   EMG patterns on day ``emg_session`` (:func:`muscle`)
        hand use                         natural statistics of hand use, from glove data (:func:`hand_use`)
        signed force pattern,            the other subjects' mean signed, absolute and absolute-derivative
        absolute force pattern,          force patterns (fingers as voxels) in ``ref_session``
        force derivative pattern         (:func:`force`)
        flexion-extension bias           the other subjects' mean signed force per chord in session 4, < 0
                                         flexion bias, > 0 extension bias (:func:`performance_model`)
        mean force magnitude,            the other subjects' mean absolute force and absolute force
        mean force derivative            derivative per chord in session 4 (:func:`performance_model`)

    Returns a dict name -> (8, 8) G, in component order.
    """
    chords = np.array(util.get_trained_and_untrained(sn)).astype(int)
    order  = chords if order is None else np.asarray(order).astype(int)

    G_type, G_trained, G_untrained = fixed(np.isin(order, chords[:4]))

    return {
        'identity'                : normalise(np.eye(8)),
        'type'                    : G_type,
        'trained'                 : G_trained,
        'untrained'               : G_untrained,
        'mean deviation'          : performance_model(sn, order, measure='MD', fname='behaviour.trial.tsv'),
        'execution time'          : performance_model(sn, order, measure='ET', fname='behaviour.trial.tsv'),
        'single finger'           : single_finger(Hem, roi, order=order, glm=glm),
        'muscle pattern'          : muscle(order=order, session=emg_session),
        'hand use'                : hand_use(order=order),
        'signed force pattern'    : force(sn, 'raw', order, ref_session=ref_session),
        'absolute force pattern'  : force(sn, 'abs', order, ref_session=ref_session),
        'force derivative pattern': force(sn, 'der', order, ref_session=ref_session),
        'flexion-extension bias'  : performance_model(sn, order, measure='force_raw', fname='force.trial.long.tsv'),
        'mean force magnitude'    : performance_model(sn, order, measure='force_abs', fname='force.trial.long.tsv'),
        'mean force derivative'   : performance_model(sn, order, measure='force_der', fname='force.trial.long.tsv'),
    }


def make_models(sn, glm=3, Hem=None, roi=None, comp_names=None):
    """The model list and the component names for one subject.

    The matrices come from :func:`model_Gs`, on the subject's own trained-first chord
    order -- the order its data are in. Pass ``Hem`` and ``roi`` to add the ROI's
    ``group_geometry`` component (see :func:`group_geometry`); without them the
    component model holds the structural components only.

    ``comp_names`` picks the models to use, by name, in order; the default is every
    model of :func:`model_Gs`. ``session`` is the session whose force patterns the
    force models are built from.
    """

    G_DICT = make_G_dict(sn, glm=glm, Hem=Hem, roi=roi)

    if comp_names is None:
        comp_names = list(G_DICT)

    M = [pcm.FixedModel('null', np.zeros((8, 8)))]
    M += [pcm.FixedModel(name, G_DICT[name]) for name in comp_names]  # null is already first

    Gc = [G_DICT[name] for name in comp_names]  # already trace-normalised by the model functions, so the weights are comparable

    M.append(pcm.ComponentModel('component', np.array(Gc)))
    M.append(pcm.FreeModel('ceil', 8))

    return M