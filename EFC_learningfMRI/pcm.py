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


def _load_family_theta(sn, stem):
    with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.theta_in.{stem}'), 'rb') as f:
        return pickle.load(f)


def _average_weights(thetas):
    """Group-average component weights, as the mean of each subject's normalised weights.

    ``thetas`` is (n_subj, n_comp) of log-weights. Each subject's weights are divided
    by their sum before averaging: the components are trace-normalised, so this
    averages the shape of each subject's predicted G and leaves its size -- the
    subject's signal level -- to the scale fitted on the left-out subject. Averaging
    in weight space, not log space, keeps the switched-off components (log-weights of
    -30 to -1000) from swamping the mean. Subjects whose weights all underflow to
    zero carry no shape and are skipped.
    """
    w = np.exp(np.asarray(thetas, dtype=float))
    w = w[w.sum(axis=1) > 0]
    if len(w) == 0:
        raise ValueError('every training subject has all component weights at zero')
    return (w / w.sum(axis=1, keepdims=True)).mean(axis=0)


def _nearest_psd(G):
    """``G`` with its negative eigenvalues set to zero, so a scaled G is a valid covariance."""
    ev, U = np.linalg.eigh((G + G.T) / 2)
    return (U * np.maximum(ev, 0)) @ U.T


def fit_model_crossval(loader, sessions=None, group=None):
    """Leave-one-subject-out likelihood of every model of the model family.

    The cross-validated counterpart of the family fit of :func:`fit_model_indiv`, for
    model comparison without an AIC/BIC penalty. ``PcmPy.fit_model_group_crossval``
    cannot be used, because each subject's conditions are in its own (trained-first)
    order, so its component matrices differ. Instead, for each subject, session, Hem
    and ROI, every model of the family becomes a fixed model:

        1. the weights that :func:`fit_model_indiv` saved for the model in every other
           subject of ``group`` are averaged (:func:`_average_weights`)
        2. the model's G is those weights times the subject's own components, on its
           own chord order (:func:`models.make_G_dict`)
        3. :func:`PcmPy.fit_model_individ` fits only a scale and the noise to the data

    So a model's likelihood is how well the others' G, scaled, predicts the subject's
    data; the weights are never fitted to the subject being scored. A model's
    components are read off its name, in the order PcmPy stores its weights.

    The last model, ``ceil``, is the cross-validated noise ceiling: the others' observed
    Gs (:func:`models.group_geometry_model`, leave-one-out over ``group``), each put on
    the subject's chord order and averaged, then projected to positive semi-definite,
    since an average of crossvalidated Gs can have small negative eigenvalues. This is
    the lower noise ceiling, not a bound: the others' average G carries their
    measurement noise, so a smooth model of the right structure can predict the
    subject better and exceed it. The upper ceiling is the individual free model
    (``ceil`` in ``component_model.T_in``).

    Run :func:`fit_model_indiv` for every subject in ``group`` first. The subjects can
    then be run one at a time (``loader`` over one subject), since ``group``, not the
    loader, sets the others.

    Writes two pickles per cell to ``pcm/subj<sn>/``, with
    ``<stem> = <atlas>.glm<glm>.<session>.<Hem>.<roi>.p``:
        model_family.T_cv.<stem>       cross-validated likelihoods, the family's models
                                       then ``ceil``
        model_family.theta_cv.<stem>   {'theta', 'weights', 'model_names', 'group'}: per
                                       model, theta is [log-scale, log-noise] and weights
                                       the others' average weights of its components
    """
    sessions = gl.sessions if sessions is None else sessions
    group    = gl.participants if group is None else list(group)

    for data in loader:
        path   = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
        others = [sn for sn in group if sn != data.sn]
        G_dict = models.make_G_dict(data.sn, glm=loader.glm, Hem=data.Hem, roi=data.roi)   # components on this subject's chord order
        order  = np.asarray(util.get_trained_and_untrained(data.sn), dtype=int)

        for session in sessions:
            print(f'model family crossval: subj{data.sn}, session {session}, {data.Hem}, {data.roi}')

            stem = f'{loader.atlas_name}.glm{loader.glm}.{session}.{data.Hem}.{data.roi}.p'
            fits = [_load_family_theta(sn, stem) for sn in others]

            model_names = fits[0]['model_names']
            if any(fit['model_names'] != model_names for fit in fits):
                raise ValueError(f'the other subjects\' model families ({stem}) hold different models')

            # every model as a fixed G: the others' average weights on this subject's components
            M, weights = [], []
            for i, name in enumerate(model_names):
                comps = [] if name == 'base' else name.split('+')
                w     = _average_weights([fit['theta'][i][:len(comps), 0] for fit in fits]) if comps else np.zeros(0)
                M.append(pcm.FixedModel(name, sum((wk * G_dict[c] for wk, c in zip(w, comps)), np.zeros((8, 8)))))
                weights.append(w)

            G_ceil = models.group_geometry_model(loader.glm, data.Hem, data.roi, order=order, sns=group, session=session, crossval=True, sn=data.sn)
            M.append(pcm.FixedModel('ceil', _nearest_psd(G_ceil)))

            keep = runs_to_keep(data.cond_vec, session=session)
            Y    = pcm.dataset.Dataset(data.betas[keep], obs_descriptors={'cond_vec': data.cond_vec[keep],
                                                                          'part_vec': data.part_vec[keep]})

            with warnings.catch_warnings():   # PcmPy adds the scale column one model at a time: 512 PerformanceWarnings
                warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
                T_cv, theta_cv = pcm.fit_model_individ(Y, M, fit_scale=True, fixed_effect='block', verbose=False)

            _dump(T_cv, f'model_family.T_cv.{stem}', path)
            _dump({'theta': theta_cv, 'weights': weights, 'model_names': [m.name for m in M], 'group': others}, f'model_family.theta_cv.{stem}', path)


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
    



    