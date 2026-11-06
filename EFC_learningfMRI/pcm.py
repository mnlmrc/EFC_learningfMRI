import PcmPy as pcm
import os
import itertools
import functools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from EFC_learningfMRI.util import runs_to_keep
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


def fit_component_model(loader, sessions=None, comp_names=None):
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
            _dump(T_mf, f'model_family.T.{stem}', path)
            _dump({'theta': theta_mf, 'model_names': [m.name for m in MF.models], 'comp_names': comp_names}, f'model_family.theta.{stem}', path)


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
    



    