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
    """Fit the model family (every combination of ``comp_names``) to each subject, Hem,
    ROI and session, without a scale.

    Writes to ``pcm/subj<sn>/``, with ``<stem> = <atlas>.glm<glm>.<session>.<Hem>.<roi>.p``:
        model_family.T_in.<stem>       likelihoods
        model_family.theta_in.<stem>   {'theta', 'model_names', 'comp_names'}
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

            T_mf, theta_mf = pcm.fit_model_individ(Y, MF,    fit_scale=False, fixed_effect='block', verbose=False)

            stem = f'{loader.atlas_name}.glm{loader.glm}.{session}.{data.Hem}.{data.roi}.p'
            
            _dump(T_mf, f'model_family.T_in.{stem}', path)
            _dump({'theta': theta_mf, 'model_names': [m.name for m in MF.models], 'comp_names': comp_names}, f'model_family.theta_in.{stem}', path)



def fit_model_transfer(loader, ref_session=3, sessions=None):
    """Fit each session's data with the subject's own G from ``ref_session``.

    G_ref is the subject's ``G_obs`` in ``ref_session`` (negative eigenvalues set to 0,
    trace-normalised). Models:

        transfer            G_ref
        transfer_identity   G_ref + identity               (noise control)
        transfer_trained    G_ref + type + trained
        transfer_untrained  G_ref + type + untrained

    transfer_trained and transfer_untrained have the same number of parameters, so their
    difference tests whether the trained chords changed more than the untrained ones.
    In ``ref_session`` the models are fitted to the data G_ref came from, so they are
    inflated there. The noise ceilings are fitted separately (:func:`fit_model_ceil`).

    Writes to ``pcm/subj<sn>/``:
        model_transfer.T.<atlas>.glm<glm>.ref<ref_session>_obs.<session>.<Hem>.<roi>.p
    """
    sessions = gl.sessions if sessions is None else sessions
    stem     = f'{loader.atlas_name}.glm{loader.glm}.ref{ref_session}_obs'

    for data in loader:
        # the subject's own observed G in ref_session
        path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
        fname  = f'G_obs.within_session.{ref_session}.glm{loader.glm}.{data.Hem}.{data.roi}.npy'
        G_obs  = np.load(os.path.join(gl.baseDir, gl.geomDir, f'subj{data.sn}', fname))

        # make G positive definite and normalise to tr=1
        G_obs  = (G_obs + G_obs.T) / 2
        lam, U = np.linalg.eigh(G_obs)
        lam    = np.clip(lam, 0, None)      # negative eigenvalues to 0
        G_ref  = U @ np.diag(lam) @ U.T
        G_ref  = models.normalise(G_ref)

        # on this subject's chord order, the order of its data
        G_dict = models.make_G_dict(data.sn, glm=loader.glm, Hem=data.Hem, roi=data.roi)
        G_id   = G_dict['identity']
        G_type = G_dict['type']
        G_tr   = G_dict['trained']
        G_untr = G_dict['untrained']

        # model list
        Gc_transfer  = np.array([G_ref])
        Gc_identity  = np.array([G_ref, G_id])
        Gc_trained   = np.array([G_ref, G_type, G_tr])
        Gc_untrained = np.array([G_ref, G_type, G_untr])
        M = [pcm.ComponentModel('transfer',           Gc_transfer),
             pcm.ComponentModel('transfer_identity',  Gc_identity),
             pcm.ComponentModel('transfer_trained',   Gc_trained),
             pcm.ComponentModel('transfer_untrained', Gc_untrained)]

        for session in sessions:
            print(f'model transfer: subj{data.sn}, ref session {ref_session} -> session {session}, {data.Hem}, {data.roi}')

            keep = runs_to_keep(data.cond_vec, session=session)
            Y    = pcm.dataset.Dataset(data.betas[keep], obs_descriptors={'cond_vec': data.cond_vec[keep],
                                                                          'part_vec': data.part_vec[keep]})

            T, _ = pcm.fit_model_individ(Y, M, fit_scale=False, fixed_effect='block', verbose=False)

            _dump(T, f'model_transfer.T.{stem}.{session}.{data.Hem}.{data.roi}.p', path)


def fit_model_ceil(loader, sessions=None):
    """Upper and lower noise ceiling of each session's data.

        upper_ceil  free G fitted to the subject
        lower_ceil  free G fitted to the other subjects, evaluated on this one with its own
                    scale and noise

    The conditions are relabelled onto ``gl.chordID``, so the free G has the same chord
    order for every subject. The ceilings only depend on the data, so they hold for any
    model fitted to the same session with ``fixed_effect='block'``. The crossvalidated
    fit needs every subject at once, so all betas are held in memory until the end.

    Writes to ``pcm/subj<sn>/``:
        model_ceil.T_in.<atlas>.glm<glm>.<session>.<Hem>.<roi>.p   upper_ceil
        model_ceil.T_cv.<atlas>.glm<glm>.<session>.<Hem>.<roi>.p   lower_ceil
    """
    sessions = gl.sessions if sessions is None else sessions
    position = {chord: k for k, chord in enumerate(gl.chordID)}   # the common chord order
    M_in     = pcm.FreeModel('upper_ceil', len(gl.chordID))
    M_cv     = pcm.FreeModel('lower_ceil', len(gl.chordID))

    Y_common, sns = defaultdict(list), defaultdict(list)
    for data in loader:
        path   = os.path.join(gl.baseDir, gl.pcmDir, f'subj{data.sn}')
        chords = util.get_trained_and_untrained(data.sn)   # chord ID of each slot of this subject's data
        chords = np.array(chords).astype(int)
        sns[data.Hem, data.roi].append(data.sn)

        for session in sessions:
            print(f'model ceil: subj{data.sn}, session {session}, {data.Hem}, {data.roi}')

            # relabel the conditions onto gl.chordID
            keep      = runs_to_keep(data.cond_vec, session=session)
            cond_info = util.parse_cond_vec(data.cond_vec[keep])
            slot      = cond_info.chord.to_numpy()          # slot 0-7 in this subject's order
            chord_id  = chords[slot]                        # chord ID of each regressor
            cond      = np.array([position[c] for c in chord_id])
            Y         = pcm.dataset.Dataset(data.betas[keep], obs_descriptors={'cond_vec': cond,
                                                                               'part_vec': data.part_vec[keep]})

            # upper noise ceiling
            T, _  = pcm.fit_model_individ(Y, M_in, fit_scale=False, fixed_effect='block', verbose=False)
            fname = f'model_ceil.T_in.{loader.atlas_name}.glm{loader.glm}.{session}.{data.Hem}.{data.roi}.p'
            _dump(T, fname, path)

            Y_common[data.Hem, data.roi, session].append(Y)

    # lower noise ceiling
    for (H, roi, session), Ys in Y_common.items():
        print(f'model ceil, crossvalidated: session {session}, {H}, {roi}')

        T, _ = pcm.fit_model_group_crossval(Ys, M_cv, fixed_effect='block', fit_scale=True, verbose=False)

        for s, sn in enumerate(sns[H, roi]):   # rows of T follow the order of Ys, the order of sns
            T_s   = T.iloc[[s]]
            fname = f'model_ceil.T_cv.{loader.atlas_name}.glm{loader.glm}.{session}.{H}.{roi}.p'
            path  = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}')
            _dump(T_s, fname, path)


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
    



    