import PcmPy as pcm
import os
import itertools
import functools
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from EFC_learningfMRI.util import runs_to_keep
import EFC_learningfMRI.util as util
from EFC_learningfMRI.betas import BetasPrewithenedLoader
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.models as models
from imaging_pipelines.util import calc_mle_corr
import nitools as nt
import nibabel as nb
import pandas as pd
import pickle
import zlib


def _dump(obj, fname, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, fname), 'wb') as f:
        pickle.dump(obj, f)


def fit_model_indiv(loader, sessions=None, comp_names=None):
    """Fit the component model and the model family to every (subject, Hem, roi, session).

    Three individual fits per cell (:func:`PcmPy.fit_model_individ`):
        1. every model of :func:`models.make_models` (null, each component alone, the
           component model, the ceiling), with ``fit_scale=True`` so a fixed model's G
           can be rescaled to the subject's signal
        2. the component model alone, without a scale, for its component weights
        3. the model family: every combination of ``comp_names``, starting from the null

    Writes four pickles per cell to ``pcm/subj<sn>/``, with
    ``<stem> = <atlas>.glm<glm>.<session>.<Hem>.<roi>.p``:
        component_model.T_in.<stem>       likelihoods of fit 1
        component_model.theta_in.<stem>   {'theta', 'comp_names'} of fit 2
        model_family.T.<stem>             likelihoods of fit 3
        model_family.theta.<stem>         {'theta', 'model_names', 'comp_names'} of fit 3

    The component names are saved with each theta, so a reader can check which
    component each weight belongs to.
    """
    sessions = gl.sessions if sessions is None else sessions

    for data in loader:
        path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')

        # the models don't depend on the session, so build them once per subject and ROI
        M  = models.make_models(data.sn, glm=loader.glm, Hem=data.Hem, roi=data.roi, comp_names=comp_names)
        MF = pcm.model.ModelFamily(M[-2].Gc, comp_names=comp_names)   # M[-2] is the component model
        # MF.models[0].name = 'null'                                     # PcmPy calls the empty model 'base'

        for session in sessions:
            keep = runs_to_keep(data.cond_vec, session=session)
            Y    = pcm.dataset.Dataset(data.betas[keep], obs_descriptors={'cond_vec': data.cond_vec[keep],
                                                                          'part_vec': data.part_vec[keep]})

            T_in, _        = pcm.fit_model_individ(Y, M,     fit_scale=True,  fixed_effect='block', verbose=True)
            _,    theta_in = pcm.fit_model_individ(Y, M[-2], fit_scale=False, fixed_effect='block', verbose=True)
            T_mf, theta_mf = pcm.fit_model_individ(Y, MF,    fit_scale=False, fixed_effect='block', verbose=False)

            stem = f'{loader.atlas_name}.glm{loader.glm}.{session}.{data.Hem}.{data.roi}.p'
            
            _dump(T_in, f'component_model.T_in.{stem}', path)
            _dump({'theta': theta_in, 'comp_names': comp_names}, f'component_model.theta_in.{stem}', path)
            _dump(T_mf, f'model_family.T_in.{stem}', path)
            _dump({'theta': theta_mf, 'model_names': [m.name for m in MF.models], 'comp_names': comp_names}, f'model_family.theta_in.{stem}', path)
    


def family_weights(sn, stem, comp_names=None):
    """Component weights of one model of a subject's model family fit.

    ``stem`` is ``<atlas>.glm<glm>.<session>.<Hem>.<roi>.p``, the cell of
    ``model_family.theta_in.<stem>`` (:func:`fit_model_indiv`). ``comp_names`` names the
    model's components, in any order; the default is the family's full model.

    Returns ``(comps, weights)``: the components in the order the family stores them
    (the order in the model's name) and their exponentiated log-weights, the first
    ``len(comps)`` entries of the model's theta (the family is fitted without a scale).
    """
    with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.theta_in.{stem}'), 'rb') as f:
        fit =  pickle.load(f)

    target = set(fit['comp_names'] if comp_names is None else comp_names)
    idx    = [i for i, m in enumerate(fit['model_names']) if set(m.split('+')) == target]
    if len(idx) != 1:
        raise ValueError(f'subj{sn} {stem}: no model of the family is built from exactly {sorted(target)}')

    comps = fit['model_names'][idx[0]].split('+')
    return comps, np.exp(fit['theta'][idx[0]][:len(comps), 0])


def transfer_stem(atlas_name, glm, ref_session, comps, learn_comps):
    """``<atlas>.glm<glm>.ref<ref_session>.<comps>.learn_<learn>``: the part of the
    :func:`fit_model_transfer` file names that says which fit it is. ``comps`` must be in
    the family's stored order (:func:`family_weights`); the learning components are
    sorted, so their order does not matter."""
    return f'{atlas_name}.glm{glm}.ref{ref_session}.{"+".join(comps)}.learn_{"+".join(sorted(learn_comps))}'


def fit_model_transfer(loader, comp_names, learn_comps=('type', 'trained', 'untrained'), ref_session=3, sessions=None):
    """Likelihood of every session's data under the weights fitted in ``ref_session``.

    Tests whether the changes of the geometry across sessions can be accounted for by
    the models that capture learning (``learn_comps``), with the mix of the other
    components (``comp_names``) kept as it was in ``ref_session``. For each subject, Hem
    and ROI, the weights of the family model built from ``comp_names``, fitted in
    ``ref_session`` (:func:`family_weights`), times the subject's components
    (:func:`models.make_G_dict`) give a fixed G_ref (trace-normalised). Three models are
    fitted to each session's data, all as component models without a scale, like the
    family:

        transfer        G_ref alone: its one weight is a scale, so only the shape of the
                        reference G carries over and a session can have more or less signal
        transfer_learn  G_ref + learn_comps: the reference mix, with the learning
                        components free (1 + len(learn_comps) parameters)
        free_learn      comp_names + learn_comps, every weight fitted to this session
                        (len(comps) + len(learn_comps) parameters)

    ``free_learn - transfer_learn`` is the change of the geometry that the learning
    components cannot account for: about 0 when they absorb it and the mix of
    ``comp_names`` is stable, > 0 when that mix has to change as well. ``free`` (the same
    member without the learning components) is in the family fit of the same session
    (``model_family.T_in``, same data), so it is not refitted here; ``type`` is not in the
    family, so ``free_learn`` is.

    trained + untrained add up to identity, so with identity in ``comp_names`` and both of
    them in ``learn_comps``, free_learn has one redundant direction: its likelihood is
    valid, its weights are not identifiable, and counting every weight as a parameter
    overstates its effective number by one. ``comp_names`` and ``learn_comps`` must not
    overlap. In ``ref_session`` itself the transfer weights were fitted to the same data,
    so transfer and the family member agree there (a check of the fit).

    Run :func:`fit_model_indiv` first, for ``ref_session`` at least. Writes one pickle per
    cell to ``pcm/subj<sn>/``:
        model_transfer.T.<transfer_stem>.<session>.<Hem>.<roi>.p
    (:func:`transfer_stem`), with the components in the family's stored order.
    """
    sessions = gl.sessions if sessions is None else sessions
    learn    = sorted(learn_comps)

    for data in loader:
        path     = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
        stem_ref = f'{loader.atlas_name}.glm{loader.glm}.{ref_session}.{data.Hem}.{data.roi}.p'
        comps, w = family_weights(data.sn, stem_ref, comp_names)
        if set(learn) & set(comps):
            raise ValueError(f'comp_names {comps} and learn_comps {learn} overlap')

        G_dict = models.make_G_dict(data.sn, glm=loader.glm, Hem=data.Hem, roi=data.roi)   # on this subject's chord order, the order of its data
        G_ref  = models.normalise(sum((wk * G_dict[c] for wk, c in zip(w, comps)), np.zeros((8, 8))))
        G_lrn  = [G_dict[c] for c in learn]
        M      = [pcm.ComponentModel('transfer',       np.array([G_ref])),
                  pcm.ComponentModel('transfer_learn', np.array([G_ref] + G_lrn)),
                  pcm.ComponentModel('free_learn',     np.array([G_dict[c] for c in comps] + G_lrn))]

        stem = f'{loader.atlas_name}.glm{loader.glm}.ref{ref_session}.{"+".join(comps)}.learn_{"+".join(learn)}'
        for session in sessions:
            print(f'model transfer: subj{data.sn}, ref session {ref_session} -> session {session}, {data.Hem}, {data.roi}')

            keep = runs_to_keep(data.cond_vec, session=session)
            Y    = pcm.dataset.Dataset(data.betas[keep], obs_descriptors={'cond_vec': data.cond_vec[keep],
                                                                          'part_vec': data.part_vec[keep]})

            T, _ = pcm.fit_model_individ(Y, M, fit_scale=False, fixed_effect='block', verbose=False)

            _dump(T, f'model_transfer.T.{stem}.{session}.{data.Hem}.{data.roi}.p', path)


def _simulate(model, theta, noise, cond_vec, part_vec, n_channel, rng):
    """One simulated pcm Dataset from ``model`` at ``theta``, with i.i.d. noise of variance
    ``noise`` and the design (``cond_vec``, ``part_vec``) and voxel count of a real session."""
    return pcm.sim.make_dataset(model, theta, cond_vec, n_channel=n_channel, n_sim=1, noise=noise,
                                part_vec=part_vec, rng=rng)[0]


def fit_model_transfer_null(loader, comp_names, learn_comps=('type', 'trained', 'untrained'), ref_session=3, sessions=None, n_sim=100, seed=0):
    """Parametric bootstrap of :func:`fit_model_transfer` under a geometry that does not change.

    Even with a stable geometry, ``free - transfer`` is positive outside ``ref_session``:
    free overfits the session it is fitted to, and transfer inherits the estimation error
    of the ``ref_session`` weights. The AIC penalty corrects for both only asymptotically
    and only when the sessions carry the same information. This function measures the
    null directly: data simulated with the same G in every session, run through the
    same pipeline.

    For each subject, Hem and ROI:

    1. Generating model: G_ref (the ``ref_session`` mix of ``comp_names``, as in
       :func:`fit_model_transfer`) fitted with one scale to each session's real data,
       which gives that session's scale (signal) and noise variance. Every session thus
       keeps its own SNR but has the same geometry.
    2. ``n_sim`` times: simulate ``ref_session`` and every other session from the
       generating model, with the real design (conditions, runs) and voxel count; fit
       ``comp_names`` to the simulated ``ref_session`` to get the weights and the
       simulated G_ref; fit transfer, transfer_learn, free and free_learn to each
       simulated session, exactly as for the real data.

    The metrics computed from these likelihoods (``free - transfer``, ``free_learn -
    transfer_learn``, ...) are their distribution under "nothing changed"; compare the
    real metric of a session to the simulated ones of the same cell and session. The
    penalty is the same on both sides, so it drops out of that comparison.

    The simulation assumes the PCM noise model: voxels independent after prewhitening.
    Real voxels are spatially correlated, so they carry less independent information
    than the simulated ones, the simulated metrics vary less than the real ones would
    under the null, and the test leans towards finding a change. The true patterns are
    drawn independently in each simulated session (stable geometry, not the same
    voxel patterns).

    ``ref_session`` itself is skipped among ``sessions``: its real transfer is in-sample,
    so there is nothing to compare. The generator is seeded by ``seed``, the subject,
    Hem and ROI, so a cell gives the same draws whatever else is run with it.

    Writes one pickle per cell to ``pcm/subj<sn>/``:
        model_transfer_null.<transfer_stem>.<Hem>.<roi>.p
    a dict with ``likelihood`` (DataFrame: sim, session, transfer, transfer_learn,
    free, free_learn), ``generating`` (DataFrame: session, scale, noise), ``n_channel``
    and ``seed``.
    """
    sessions = [s for s in (gl.sessions if sessions is None else sessions) if s != ref_session]
    learn    = sorted(learn_comps)
    names    = ['transfer', 'transfer_learn', 'free', 'free_learn']

    for data in loader:
        path     = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
        stem_ref = f'{loader.atlas_name}.glm{loader.glm}.{ref_session}.{data.Hem}.{data.roi}.p'
        comps, w = family_weights(data.sn, stem_ref, comp_names)
        if set(learn) & set(comps):
            raise ValueError(f'comp_names {comps} and learn_comps {learn} overlap')

        G_dict     = models.make_G_dict(data.sn, glm=loader.glm, Hem=data.Hem, roi=data.roi)
        G_comp     = np.array([G_dict[c] for c in comps])
        G_lrn      = [G_dict[c] for c in learn]
        G_ref      = models.normalise(np.einsum('k,kij->ij', w, G_comp))
        M_gen      = pcm.ComponentModel('transfer', np.array([G_ref]))
        M_free     = pcm.ComponentModel('free', G_comp)
        M_free_lrn = pcm.ComponentModel('free_learn', np.array(list(G_comp) + G_lrn))
        stem       = transfer_stem(loader.atlas_name, loader.glm, ref_session, comps, learn)
        n_channel  = data.betas.shape[1]
        rng        = np.random.default_rng([seed, data.sn, zlib.crc32(f'{data.Hem}.{data.roi}'.encode())])

        # 1. generating model: G_ref with each session's own scale and noise, fitted to the real data
        gen = {}
        for session in [ref_session] + sessions:
            keep     = runs_to_keep(data.cond_vec, session=session)
            Y        = pcm.dataset.Dataset(data.betas[keep], obs_descriptors={'cond_vec': data.cond_vec[keep],
                                                                              'part_vec': data.part_vec[keep]})
            T, theta = pcm.fit_model_individ(Y, M_gen, fit_scale=False, fixed_effect='block', verbose=False)
            gen[session] = dict(theta=theta[0][:M_gen.n_param, 0], noise=T.noise['transfer'].iloc[0],
                                cond_vec=data.cond_vec[keep], part_vec=data.part_vec[keep])

        # 2. the transfer pipeline on data simulated from it
        rows = []
        for i in range(n_sim):
            print(f'model transfer null: subj{data.sn}, {data.Hem}, {data.roi}, sim {i + 1}/{n_sim}')

            Y_ref       = _simulate(M_gen, n_channel=n_channel, rng=rng, **gen[ref_session])
            _, theta    = pcm.fit_model_individ(Y_ref, M_free, fit_scale=False, fixed_effect='block', verbose=False)
            G_ref_sim   = models.normalise(np.einsum('k,kij->ij', np.exp(theta[0][:len(comps), 0]), G_comp))
            M           = [pcm.ComponentModel('transfer',       np.array([G_ref_sim])),
                           pcm.ComponentModel('transfer_learn', np.array([G_ref_sim] + G_lrn)),
                           M_free, M_free_lrn]

            Y_sim       = [_simulate(M_gen, n_channel=n_channel, rng=rng, **gen[s]) for s in sessions]
            T, _        = pcm.fit_model_individ(Y_sim, M, fit_scale=False, fixed_effect='block', verbose=False)
            rows       += [{'sim': i, 'session': s, **{m: T.likelihood[m].iloc[j] for m in names}} for j, s in enumerate(sessions)]

        generating = pd.DataFrame([{'session': s, 'scale': np.exp(g['theta'][0]), 'noise': g['noise']} for s, g in gen.items()])
        _dump({'likelihood': pd.DataFrame(rows), 'generating': generating, 'n_channel': n_channel, 'seed': seed},
              f'model_transfer_null.{stem}.{data.Hem}.{data.roi}.p', path)


# def _normalise_block(G, idx, name):
#     """The ``idx`` x ``idx`` block of ``G``, centred over those conditions and trace-normalised."""
#     C  = pcm.centering(len(idx))
#     G  = C @ G[np.ix_(idx, idx)] @ C
#     tr = np.trace(G)
#     if tr <= 1e-12:
#         raise ValueError(f'{name} has no variance within the chord set (trace {tr:.3g}), so it cannot be a component there')
#     return G / tr


# def fit_model_transfer_chord(loader, comp_names, ref_session=3, sessions=None):
#     """:func:`fit_model_transfer` separately for the trained and the untrained chords.

#     If what does not transfer across sessions is due to training, the weights should
#     carry over for the untrained chords and fail for the trained ones. For each chord set
#     (slots 0-3 trained, 4-7 untrained, :func:`util.runs_to_keep`), every component is
#     cut to that set's 4 x 4 block, centred and trace-normalised (:func:`_normalise_block`),
#     and three component models without a scale are fitted to the set's data:

#         identity  the four chords equally distinct
#         free      the components with weights fitted to this session
#         transfer  the components with the weights ``free`` got in ``ref_session`` on the
#                   same chord set, as one component: only its scale is fitted

#     The reference weights are fitted on the chord set itself, so in ``ref_session``
#     transfer and free agree (a check of the fit), and both sets get the same number of
#     parameters: identity 1, transfer 1, free ``len(comps)``. Compare the sets on
#     ``free - transfer`` (what does not transfer) and ``transfer - identity`` (negative
#     when the reference structure fits worse than no structure at all). The share
#     ``(transfer - identity) / (free - identity)`` is unstable: free holds identity, so
#     when a set has no structure left beyond identity the denominator goes to 0. The
#     trained/untrained components are constant within a set, so ``comp_names`` cannot
#     hold them.

#     The components are named in the order the family stores them (read off the
#     ``ref_session`` family fit, :func:`family_weights`), so the pickles match those of
#     :func:`fit_model_transfer`; the family's weights are not used. Writes one pickle per
#     cell to ``pcm/subj<sn>/``:
#         model_transfer_chord.T.<atlas>.glm<glm>.ref<ref_session>.<comps>.<chord>.<session>.<Hem>.<roi>.p
#     """
#     sessions = gl.sessions if sessions is None else sessions

#     for data in loader:
#         path     = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
#         stem_ref = f'{loader.atlas_name}.glm{loader.glm}.{ref_session}.{data.Hem}.{data.roi}.p'
#         comps, _ = family_weights(data.sn, stem_ref, comp_names)
#         if {'trained', 'untrained'} & set(comps):
#             raise ValueError(f'comp_names {comps} hold trained or untrained, which are constant within a chord set')

#         G_dict = models.make_G_dict(data.sn, glm=loader.glm, Hem=data.Hem, roi=data.roi)   # on this subject's chord order, slots 0-3 trained

#         for chord, idx in (('trained', np.arange(4)), ('untrained', np.arange(4, 8))):
#             Gc     = np.array([_normalise_block(G_dict[c], idx, c) for c in comps])
#             M_free = pcm.ComponentModel('free', Gc)

#             # the reference weights: free fitted to this chord set in ref_session
#             _, theta = pcm.fit_model_individ(_dataset(data, ref_session, chord), M_free, fit_scale=False, fixed_effect='block', verbose=False)
#             G_ref    = np.einsum('k,kij->ij', np.exp(theta[0][:len(comps), 0]), Gc)
#             if np.trace(G_ref) <= 0:
#                 raise ValueError(f'subj{data.sn} {data.Hem} {data.roi} {chord}: every reference weight is zero')

#             M = [pcm.ComponentModel('identity', np.array([_normalise_block(np.eye(8), idx, 'identity')])),
#                  pcm.ComponentModel('transfer', np.array([G_ref / np.trace(G_ref)])),
#                  M_free]

#             for session in sessions:
#                 print(f'model transfer {chord}: subj{data.sn}, ref session {ref_session} -> session {session}, {data.Hem}, {data.roi}')

#                 T, _ = pcm.fit_model_individ(_dataset(data, session, chord), M, fit_scale=False, fixed_effect='block', verbose=False)

#                 stem = f'{loader.atlas_name}.glm{loader.glm}.ref{ref_session}.{"+".join(comps)}.{chord}.{session}.{data.Hem}.{data.roi}.p'
#                 _dump(T, f'model_transfer_chord.T.{stem}', path)


# def _average_weights(thetas):
#     """Group-average component weights, as the mean of each subject's normalised weights.

#     ``thetas`` is (n_subj, n_comp) of log-weights. Each subject's weights are divided
#     by their sum before averaging: the components are trace-normalised, so this
#     averages the shape of each subject's predicted G and leaves its size -- the
#     subject's signal level -- to the scale fitted on the left-out subject. Averaging
#     in weight space, not log space, keeps the switched-off components (log-weights of
#     -30 to -1000) from swamping the mean. Subjects whose weights all underflow to
#     zero carry no shape and are skipped.
#     """
#     w = np.exp(np.asarray(thetas, dtype=float))
#     w = w[w.sum(axis=1) > 0]
#     if len(w) == 0:
#         raise ValueError('every training subject has all component weights at zero')
#     return (w / w.sum(axis=1, keepdims=True)).mean(axis=0)


# def _nearest_psd(G):
#     """``G`` with its negative eigenvalues set to zero, so a scaled G is a valid covariance."""
#     ev, U = np.linalg.eigh((G + G.T) / 2)
#     return (U * np.maximum(ev, 0)) @ U.T


# def fit_model_crossval(loader, sessions=None, group=None):
#     """Leave-one-subject-out likelihood of every model of the model family.

#     The cross-validated counterpart of the family fit of :func:`fit_model_indiv`, for
#     model comparison without an AIC/BIC penalty. ``PcmPy.fit_model_group_crossval``
#     cannot be used, because each subject's conditions are in its own (trained-first)
#     order, so its component matrices differ. Instead, for each subject, session, Hem
#     and ROI, every model of the family becomes a fixed model:

#         1. the weights that :func:`fit_model_indiv` saved for the model in every other
#            subject of ``group`` are averaged (:func:`_average_weights`)
#         2. the model's G is those weights times the subject's own components, on its
#            own chord order (:func:`models.make_G_dict`)
#         3. :func:`PcmPy.fit_model_individ` fits only a scale and the noise to the data

#     So a model's likelihood is how well the others' G, scaled, predicts the subject's
#     data; the weights are never fitted to the subject being scored. A model's
#     components are read off its name, in the order PcmPy stores its weights.

#     The last model, ``ceil``, is the cross-validated noise ceiling: the others' observed
#     Gs (:func:`models.group_geometry_model`, leave-one-out over ``group``), each put on
#     the subject's chord order and averaged, then projected to positive semi-definite,
#     since an average of crossvalidated Gs can have small negative eigenvalues. This is
#     the lower noise ceiling, not a bound: the others' average G carries their
#     measurement noise, so a smooth model of the right structure can predict the
#     subject better and exceed it. The upper ceiling is the individual free model
#     (``ceil`` in ``component_model.T_in``).

#     Run :func:`fit_model_indiv` for every subject in ``group`` first. The subjects can
#     then be run one at a time (``loader`` over one subject), since ``group``, not the
#     loader, sets the others.

#     Writes two pickles per cell to ``pcm/subj<sn>/``, with
#     ``<stem> = <atlas>.glm<glm>.<session>.<Hem>.<roi>.p``:
#         model_family.T_cv.<stem>       cross-validated likelihoods, the family's models
#                                        then ``ceil``
#         model_family.theta_cv.<stem>   {'theta', 'weights', 'model_names', 'group'}: per
#                                        model, theta is [log-scale, log-noise] and weights
#                                        the others' average weights of its components
#     """
#     sessions = gl.sessions if sessions is None else sessions
#     group    = gl.participants if group is None else list(group)

#     for data in loader:
#         path   = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
#         others = [sn for sn in group if sn != data.sn]
#         G_dict = models.make_G_dict(data.sn, glm=loader.glm, Hem=data.Hem, roi=data.roi)   # components on this subject's chord order
#         order  = np.asarray(util.get_trained_and_untrained(data.sn), dtype=int)

#         for session in sessions:
#             print(f'model family crossval: subj{data.sn}, session {session}, {data.Hem}, {data.roi}')

#             stem = f'{loader.atlas_name}.glm{loader.glm}.{session}.{data.Hem}.{data.roi}.p'
#             fits = [_load_family_theta(sn, stem) for sn in others]

#             model_names = fits[0]['model_names']
#             if any(fit['model_names'] != model_names for fit in fits):
#                 raise ValueError(f'the other subjects\' model families ({stem}) hold different models')

#             # every model as a fixed G: the others' average weights on this subject's components
#             M, weights = [], []
#             for i, name in enumerate(model_names):
#                 comps = [] if name == 'base' else name.split('+')
#                 w     = _average_weights([fit['theta'][i][:len(comps), 0] for fit in fits]) if comps else np.zeros(0)
#                 M.append(pcm.FixedModel(name, sum((wk * G_dict[c] for wk, c in zip(w, comps)), np.zeros((8, 8)))))
#                 weights.append(w)

#             G_ceil = models.group_geometry_model(loader.glm, data.Hem, data.roi, order=order, sns=group, session=session, crossval=True, sn=data.sn)
#             M.append(pcm.FixedModel('ceil', _nearest_psd(G_ceil)))

#             keep = runs_to_keep(data.cond_vec, session=session)
#             Y    = pcm.dataset.Dataset(data.betas[keep], obs_descriptors={'cond_vec': data.cond_vec[keep],
#                                                                           'part_vec': data.part_vec[keep]})

#             with warnings.catch_warnings():   # PcmPy adds the scale column one model at a time: 512 PerformanceWarnings
#                 warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
#                 T_cv, theta_cv = pcm.fit_model_individ(Y, M, fit_scale=True, fixed_effect='block', verbose=False)

#             _dump(T_cv, f'model_family.T_cv.{stem}', path)
#             _dump({'theta': theta_cv, 'weights': weights, 'model_names': [m.name for m in M], 'group': others}, f'model_family.theta_cv.{stem}', path)


@dataclass
class CorrelationBetweenSessions():
    """Per-subject Datasets for between-session correlation.

    Betas are prewhitened once per (subject, ROI) by BetasPrewithenedLoader and
    reused for every session-pair and chord set, so the expensive step runs once.
    """

    glm: int
    sns: Sequence[int]  = field(default_factory=lambda: gl.participants)
    atlas_name: str     = 'ROI'
    residual_fname: str = 'residual.dtseries.nii'
    Hem: Sequence[str]  = ('L', 'R')

    def group_datasets(self, session_pairs, chords):
        """Per-subject Datasets, cross-validated G and SNR for every group.

        Drives the loader once (one prewhitening per subject/ROI) and slices each
        subject's betas into every (session-pair, chord) group with ``runs_to_keep``,
        which is the only reader of the cond_vec format. ``session_pairs`` are pairs of
        real session numbers (3, 9, 23), not label indices. Betas are centred
        across voxels (``axis=1``) so ``G_to_cosine`` reads out a Pearson-style
        correlation. Returns three dicts keyed by (Hem, roi, session_pair, chord):
        ``Y[key]`` a list of one pcm Dataset per subject, ``cov[key]`` a
        (n_subj, 8, 8) array, and ``snr[key]`` a (n_subj,) signal/noise ratio.
        Subjects follow loader order (``self.sns``).

        SNR is ``mean(diag(G)) / mean(diag(Sig))`` where ``Sig`` is the second
        output of ``est_G_crossval`` (the noise covariance of the single-run
        condition estimates): cross-validated signal variance over noise variance.
        """
        loader = BetasPrewithenedLoader(self.glm, sns=self.sns, atlas_name=self.atlas_name, residual_fname=self.residual_fname, Hem=self.Hem)
        Y, cov, snr = defaultdict(list), defaultdict(list), defaultdict(list)
        for data in loader:
            for sessions, chord in itertools.product(session_pairs, chords):
                keep = runs_to_keep(data.cond_vec, session=sessions, chord=chord)

                betas     = data.betas[keep]
                betas     = betas - betas.mean(axis=1, keepdims=True)   # centre across voxels -> cosine == Pearson r
                cond_vec  = data.cond_vec[keep]
                part_vec  = data.part_vec[keep]

                dataset = pcm.dataset.Dataset(betas, obs_descriptors={'cond_vec': cond_vec, 'part_vec': part_vec})

                cov_, Sig_ = pcm.est_G_crossval(dataset.measurements, cond_vec, part_vec, X=pcm.indicator(part_vec))

                Y[data.Hem, data.roi, sessions, chord].append(dataset)
                cov[data.Hem, data.roi, sessions, chord].append(np.asarray(cov_))
                snr[data.Hem, data.roi, sessions, chord].append(np.diagonal(cov_).mean() / np.diagonal(Sig_).mean())

        return (Y,
                {key: np.array(v) for key, v in cov.items()},
                {key: np.array(v) for key, v in snr.items()})


def fit_correlation(Y, model):
    """Fit the correlation model (individual + group) to a list of Datasets and
    return per-subject individual r, group r and SNR.

    Note on `cond_effect`: with one block regressor per run, the condition mean is
    already absorbed by the fixed effect ([Z X] is rank deficient by one per
    session), so the cond_effect thetas are unidentifiable and sit at their
    est_theta0 floor. r is identical either way; cond_effect=False just drops two
    dead parameters. The flag is read off the model so the theta indices used by
    calc_mle_corr cannot desync from it.
    """
    _, theta    = pcm.fit_model_individ(Y, model, fixed_effect='block', fit_scale=False, verbose=True)
    _, theta_gr = pcm.fit_model_group(Y, model, fixed_effect='block', fit_scale=False, verbose=True)

    r_indiv, r_group, SNR, _, _, _ = calc_mle_corr(model, theta[0], theta_gr[0], cond_effect=model.cond_effect)

    return r_indiv, r_group, SNR
    



    