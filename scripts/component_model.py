import argparse
import os
import pickle
import itertools
import functools
import numpy as np
import PcmPy as pcm
import pandas as pd
import inspect
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.betas as betas
import EFC_learningfMRI.pcm as pcm_
import EFC_learningfMRI.models as models


def fit_model_indiv(sns            = gl.participants, 
                    glm            = 3,
                    atlas_name     = 'ROI',
                    residual_fname = 'residual.dtseries.nii',
                    comp_names     = ['type', 
                                      'trained', 
                                      'untrained',                    
                                      'mean deviation', 
                                      'single finger', 
                                      'muscle pattern', 
                                      'hand use',
                                      'flexion-extension bias', 
                                      'mean force magnitude'],
                    sessions       = (3, 9, 23)):

    """Fit the PCM component model per (subject, Hem, roi, session).

    Prewhitens the betas once per (subject, roi) with BetasPrewithenedLoader, then
    fits the component model (which includes 'base', the group-mean observed G of
    session 3) and pickles each fit's ``theta_in`` next to the participant's Gs, as
    ``component_model.theta_in.<atlas_name>.glm<glm>.<session>.<Hem>.<roi>.p`` --
    which component_summary reads back.
    """
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    pcm_.fit_model_indiv(loader, comp_names=comp_names, sessions=sessions)


def _calc_weights(sn, stem, comp_names=None):
    """Component weights of one model of a subject's model family fit.

    ``stem`` is ``<atlas>.glm<glm>.<session>.<Hem>.<roi>.p``, the cell of
    ``model_family.theta_in.<stem>`` (:func:`fit_model_indiv`). ``comp_names`` names the
    model's components, in any order; the default is the family's full model.

    Returns ``(comps, weights)``: the components in the order the family stores them
    (the order in the model's name) and their exponentiated log-weights, the first
    ``len(comps)`` entries of the model's theta (the family is fitted without a scale).
    """
    with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.theta_in.{stem}'), 'rb') as f:
        theta =  pickle.load(f)

    comp_names = set(theta['comp_names'] if comp_names is None else comp_names)
    idx    = [i for i, m in enumerate(theta['model_names']) if set(m.split('+')) == comp_names]
    if len(idx) != 1:
        raise ValueError(f'subj{sn} {stem}: no model of the family is built from exactly {sorted(comp_names)}')

    comps   = theta['model_names'][idx[0]].split('+')
    theta_m = theta['theta'][idx[0]][:, 0]       # this model's parameters
    weights = np.exp(theta_m[:len(comps)])       # component weights
    noise   = np.exp(theta_m[-1])                # noise variance (last parameter)
    return comps, weights, noise



def make_weight_dataframe(sns=gl.participants, glm=3, atlas_name='ROI', comp_names=None):
    """Component weights of one model of the family, one row per (sn, session, Hem, roi, component).

    ``comp_names`` names the components of the model to read, in any order; the default is
    the family's full model (from the command line, the ``--comp_names`` default is the
    fitted set, so the full model too). The model is looked up by its set of components, and its
    weights are the exponentiated first ``len(comp_names)`` entries of its theta (the family
    is fitted without a scale, so these come before the noise parameter), named in the
    order the family stores them -- the order of the components in the model's name.
    Written to ``model_family.weight.<atlas_name>.glm<glm>.<components>.tsv`` in the pcm dir,
    with the components joined by '_' in the stored order.
    """
    rois = gl.rois[atlas_name]

    df = []
    for sn, session, H, roi in itertools.product(sns, gl.sessions, gl.Hem, rois):
        stem             = f'{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'
        comps, w, noise  = _calc_weights(sn, stem, comp_names)
        df.append(pd.DataFrame({'component': comps,
                                'weight'   : w,
                                'noise'    : noise,
                                'sn'       : sn,
                                'session'  : session,
                                'Hem'      : H,
                                'roi'      : roi}))

    df = pd.concat(df, ignore_index=True)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_family.weight.{atlas_name}.glm{glm}.{"+".join(comps)}.tsv'), sep='\t', index=False)


def make_likelihood_dataframe(sns=gl.participants, glm=3, atlas_name='ROI'):
    """Likelihood of every model of the model family, one row per (sn, session, Hem, roi).

    The columns are the family's model names, in the family's order, so the mean over
    participants of a cell can go straight into ``PcmPy.vis.plot_tree``. The tsv is
    named after the component set the family fit stored with its theta.
    """
    rois = gl.rois[atlas_name]

    LL = []
    for sn, session, H, roi in itertools.product(sns, gl.sessions, gl.Hem, rois):

        print(f'model family: subj{sn}, session {session}, {H}, {roi}')

        stem = f'{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'
        with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.T_in.{stem}'), 'rb') as f:
            T = pickle.load(f)
        with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.theta_in.{stem}'), 'rb') as f:
            fit = pickle.load(f)

        ll            = T.likelihood.copy()
        ll['sn']      = sn
        ll['session'] = session
        ll['Hem']     = H
        ll['roi']     = roi
        LL.append(ll)

    LL = pd.concat(LL, ignore_index=True)
    LL.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_family.likelihood.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def fit_model_transfer(sns            = gl.participants,
                       glm            = 3,
                       atlas_name     = 'ROI',
                       residual_fname = 'residual.dtseries.nii',
                       ref_session    = 3,
                       sessions       = (3, 9, 23)):
    """Likelihood of every session's data under the subject's own G of ``ref_session``,
    alone and with identity, type + trained or type + untrained added (:func:`pcm.fit_model_transfer`).

    Needs the ``G_obs`` of ``ref_session`` (``pattern.py``, step ``calc_G_rois``). Collect
    the pickles with step ``make_transfer_dataframe``, after step ``fit_ceil``.
    """
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    pcm_.fit_model_transfer(loader, ref_session=ref_session, sessions=sessions)


def fit_model_ceil(sns            = gl.participants,
                   glm            = 3,
                   atlas_name     = 'ROI',
                   residual_fname = 'residual.dtseries.nii',
                   sessions       = (3, 9, 23)):
    """Upper and crossvalidated lower noise ceiling of each session (:func:`pcm.fit_model_ceil`)."""
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    pcm_.fit_model_ceil(loader, sessions=sessions)


def make_ceil_dataframe(sns=gl.participants, glm=3, atlas_name='ROI', sessions=(3, 9, 23)):
    """Upper and lower noise ceiling of step ``fit_ceil``, one row per (sn, session, Hem, roi).

        upper_ceil  free G fitted to the subject              (``model_ceil.T_in``)
        lower_ceil  free G fitted to the other subjects       (``model_ceil.T_cv``)

    Written to ``model_ceil.likelihood.<atlas>.glm<glm>.tsv`` in the pcm dir.
    """
    rois = gl.rois[atlas_name]

    df = []
    for sn, session, H, roi in itertools.product(sns, sessions, gl.Hem, rois):
        path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}')
        stem = f'{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'

        with open(os.path.join(path, f'model_ceil.T_in.{stem}'), 'rb') as f:
            T_in = pickle.load(f)
        with open(os.path.join(path, f'model_ceil.T_cv.{stem}'), 'rb') as f:
            T_cv = pickle.load(f)

        df.append({'upper_ceil': T_in.likelihood['upper_ceil'].iloc[0],
                   'lower_ceil': T_cv.likelihood['lower_ceil'].iloc[0],
                   'sn'        : sn,
                   'session'   : session,
                   'Hem'       : H,
                   'roi'       : roi})

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_ceil.likelihood.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def fit_model_split(sns            = gl.participants,
                    glm            = 3,
                    atlas_name     = 'ROI',
                    residual_fname = 'residual.dtseries.nii',
                    comp_names     = None,
                    sessions       = (3, 9, 23)):
    """Component weights of each half of each session's runs (:func:`pcm.fit_model_split`).
    Then step ``fit_transfer_split``."""
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    pcm_.fit_model_split(loader, comp_names=comp_names, sessions=sessions)



def make_transfer_dataframe(sns=gl.participants, glm=3, atlas_name='ROI', ref_session=3, sessions=(3, 9, 23)):
    """Likelihoods of step ``fit_transfer`` next to the noise ceilings and the null model,
    one row per (sn, session, Hem, roi).

    Columns, all from fits to the same session's data:
        transfer            the subject's ``ref_session`` G                    (``model_transfer.T``)
        transfer_identity   transfer + identity                                (``model_transfer.T``)
        transfer_trained    transfer + type + trained                          (``model_transfer.T``)
        transfer_untrained  transfer + type + untrained                        (``model_transfer.T``)
        upper_ceil          free G fitted to the subject                       (``model_ceil.T_in``)
        lower_ceil          free G of the other subjects                       (``model_ceil.T_cv``)
        null                the family's empty model, ``base``                 (``model_family.T_in``)
        identity            identity alone                                     (``model_family.T_in``)

    Written to ``model_transfer.<atlas>.glm<glm>.ref<ref_session>_obs.tsv`` in the pcm dir.
    """
    rois = gl.rois[atlas_name]
    stem = f'{atlas_name}.glm{glm}.ref{ref_session}_obs'   # as in pcm.fit_model_transfer

    df = []
    for sn, H, roi in itertools.product(sns, gl.Hem, rois):
        path     = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}')

        for session in sessions:
            with open(os.path.join(path, f'model_transfer.T.{stem}.{session}.{H}.{roi}.p'), 'rb') as f:
                T = pickle.load(f)
            with open(os.path.join(path, f'model_ceil.T_in.{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'), 'rb') as f:
                T_in = pickle.load(f)
            with open(os.path.join(path, f'model_ceil.T_cv.{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'), 'rb') as f:
                T_cv = pickle.load(f)
            with open(os.path.join(path, f'model_family.T_in.{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'), 'rb') as f:
                T_mf = pickle.load(f)

            df.append({'transfer'           : T.likelihood['transfer'].iloc[0],
                       'transfer_identity'  : T.likelihood['transfer_identity'].iloc[0],
                       'transfer_trained'   : T.likelihood['transfer_trained'].iloc[0],
                       'transfer_untrained' : T.likelihood['transfer_untrained'].iloc[0],
                       'upper_ceil'         : T_in.likelihood['upper_ceil'].iloc[0],
                       'lower_ceil'         : T_cv.likelihood['lower_ceil'].iloc[0],
                       'null'               : T_mf.likelihood['base'].iloc[0],
                       'identity'           : T_mf.likelihood['identity'].iloc[0],
                       'ref_session'        : ref_session,
                       'sn'                 : sn,
                       'session'            : session,
                       'Hem'                : H,
                       'roi'                : roi})

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_transfer.{stem}.tsv'), sep='\t', index=False)


# Step name -> function: the PCM fits, then the tsvs collected from them.
FUNC = {'fit_indiv'                : fit_model_indiv,
        'make_weight_dataframe'    : make_weight_dataframe,
        'make_likelihood_dataframe': make_likelihood_dataframe,
        'fit_transfer'             : fit_model_transfer,
        'fit_ceil'                 : fit_model_ceil,
        'make_ceil_dataframe'      : make_ceil_dataframe,
        'make_transfer_dataframe'  : make_transfer_dataframe}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`glm=`, `atlas_name=`, ...), but only
    the ones it accepts.
    """

    if what is not None:
        func = FUNC[what] # select function
        accepted = inspect.signature(func).parameters # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted}) # run the function

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Fit the PCM component model and the model family to the neural patterns and collect the fits into tsvs.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas come from (default: the step default, 3)')
    parser.add_argument('--sns', nargs='+', type=int, default=gl.participants, help='participant ids to include in the analysis')
    parser.add_argument('--sessions', nargs='+', type=int, default=(3, 9, 23), help='participant ids to include in the analysis')
    parser.add_argument('--comp_names', nargs='+', default=['identity', 'trained', 'untrained', 'single finger', 'muscle pattern', 'hand use', 'signed force pattern', 'mean deviation', 'flexion-extension bias', 'mean force magnitude'], type=str, help='components to include, keys of models.make_G_dict (quote names with spaces)')
    parser.add_argument('--ref_session', type=int, default=None, help='session whose component weights carry over to the others (steps fit_transfer*, transfer*; default 3)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)
