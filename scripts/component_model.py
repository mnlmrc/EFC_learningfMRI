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


def fit_model_indiv(sns            = gl.participants, 
                    glm            = 3,
                    atlas_name     = 'ROI',
                    residual_fname = 'residual.dtseries.nii',
                    comp_names     = ['identity', 'type', 'trained', 'untrained',                       # keys of models.make_G_dict
                                    'mean deviation', 'execution time',
                                    'single finger', 'muscle pattern', 'hand use',
                                    'signed force pattern', 'absolute force pattern', 'force derivative pattern',
                                    'flexion-extension bias', 'mean force magnitude', 'mean force derivative'],
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



def fit_model_crossval(sns            = gl.participants,
                       glm            = 3,
                       atlas_name     = 'ROI',
                       residual_fname = 'residual.dtseries.nii',
                       sessions       = (3, 9, 23)):
    """Leave-one-subject-out likelihoods of the model family (:func:`pcm.fit_model_family_crossval`).

    Needs the family fits of every participant (step ``fit``) on the drive. The weights
    are always averaged over all of ``gl.participants`` but the subject scored, so
    ``--sns`` can split the subjects across jobs.
    """
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    pcm_.fit_model_crossval(loader, sessions=sessions)


def make_component_weight_dataframe(sns=None, glm=3, atlas_name='ROI'):
    """Collect the component weights fitted by component_fit into one dataframe.

    Reads each pickled ``theta_in``, takes the component model's theta (the other
    models' thetas have different lengths, so the list cannot be arrayed as a whole),
    exponentiates it to weights -- one per component -- and writes one long-form row
    per component to ``component_model.<atlas_name>.glm<glm>.tsv`` in the pcm dir.

    ``log_bf`` is each component's log Bayes factor from the model family fit of the
    same cell (``PcmPy`` ``ModelFamily.component_bayesfactor``, AIC-corrected): the
    evidence for the models with the component against those without, averaged over
    every combination of the other components. It is NaN for the family's base
    components, which are in every model.

    The component names are the ones the fit itself stored next to its theta, so a
    component list that has changed since the fit cannot shift the weights onto the
    wrong names -- a pickle from before the names were stored raises instead.

    The fit files do not carry the component set in their name, so the tsv is named
    after the set the fits themselves stored.
    """
    sns = gl.participants if sns is None else sns

    df = pd.DataFrame()
    for sn, session, H, roi in itertools.product(sns, gl.sessions, gl.Hem, gl.rois[atlas_name]):

        path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'component_model.theta_in.{atlas_name}.glm{glm}.{session}.{H}.{roi}.p')
        with open(path, 'rb') as f:
            theta = pickle.load(f)

        comp_names = theta['comp_names']

        weight = np.exp(np.array(theta['theta'][0][:len(comp_names)]))

        df_tmp = pd.DataFrame({'weight'   : weight.squeeze(), 
                               'component': comp_names,})
        df_tmp['sn']      = sn
        df_tmp['Hem']     = H
        df_tmp['roi']     = roi
        df_tmp['session'] = session
        df = pd.concat([df, df_tmp], ignore_index=True)

    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'component_model.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def make_model_family_weight_dataframe(sns=gl.participants, glm=3, atlas_name='ROI', comp_names=None):
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
        stem = f'{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'
        with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.theta_in.{stem}'), 'rb') as f:
            fit = pickle.load(f)

        target = set(fit['comp_names'] if comp_names is None else comp_names)
        idx    = [i for i, m in enumerate(fit['model_names']) if set(m.split('+')) == target]
        if len(idx) != 1:
            raise ValueError(f'subj{sn} {stem}: no model of the family is built from exactly {sorted(target)}')

        comps = fit['model_names'][idx[0]].split('+')
        df.append(pd.DataFrame({'component': comps,
                                'weight'   : np.exp(fit['theta'][idx[0]][:len(comps), 0]),
                                'sn'       : sn,
                                'session'  : session,
                                'Hem'      : H,
                                'roi'      : roi}))

    df = pd.concat(df, ignore_index=True)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_family.weight.{atlas_name}.glm{glm}.{"+".join(comps)}.tsv'), sep='\t', index=False)


def make_likelihood_dataframe(sns=gl.participants, glm=3, atlas_name='ROI'):
    """Likelihood of every model of the component fit, one row per (sn, session, Hem, roi).

    The tsv is named after the component set, read off the fitted models: every model
    but null, the component model and the ceiling is one of the components.
    """
    rois = gl.rois[atlas_name]

    LL = []
    for sn, session, H, roi in itertools.product(sns, gl.sessions, gl.Hem, rois):
        path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'component_model.T_in.{atlas_name}.glm{glm}.{session}.{H}.{roi}.p')
        with open(path, 'rb') as f:
            T_in = pickle.load(f)

        ll            = T_in.likelihood
        comp          = '_'.join(c for c in ll.columns if c not in ('null', 'component', 'ceil'))
        ll['sn']      = sn
        ll['session'] = session
        ll['Hem']     = H
        ll['roi']     = roi
        LL.append(ll)

    LL = pd.concat(LL, ignore_index=True)
    LL.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'likelihood.{atlas_name}.glm{glm}.{comp}.tsv'), sep='\t', index=False)


def make_model_family_dataframe(sns=gl.participants, glm=3, atlas_name='ROI', crossval=True):
    """Likelihood of every model of the model family, one row per (sn, session, Hem, roi).

    The columns are the family's model names, in the family's order, so the mean over
    participants of a cell can go straight into ``PcmPy.vis.plot_tree``. The tsv is
    named after the component set the family fit stored with its theta.

    ``crossval=True`` collects the leave-one-subject-out likelihoods of step
    ``fit_crossval`` instead, into ``model_family_cv.likelihood.<atlas>.glm<glm>.tsv``;
    take their Bayes factors with ``util.component_bayesfactor(..., penalty=0)``.
    """
    rois = gl.rois[atlas_name]
    cv   = 'cv' if crossval else 'in'

    LL = []
    for sn, session, H, roi in itertools.product(sns, gl.sessions, gl.Hem, rois):

        print(f'model family: subj{sn}, session {session}, {H}, {roi}')

        stem = f'{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'
        with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.T_{cv}.{stem}'), 'rb') as f:
            T = pickle.load(f)
        with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.theta_{cv}.{stem}'), 'rb') as f:
            fit = pickle.load(f)

        ll            = T.likelihood.copy()
        ll['sn']      = sn
        ll['session'] = session
        ll['Hem']     = H
        ll['roi']     = roi
        LL.append(ll)

    LL = pd.concat(LL, ignore_index=True)
    LL.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_family_{cv}.likelihood.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


# Step name -> function: the PCM fits, then the tsvs collected from them.
FUNC = {
    'fit_indiv'      : fit_model_indiv,
    'fit_crossval'   : fit_model_crossval,
    'weight'         : make_component_weight_dataframe,
    'likelihood'     : make_likelihood_dataframe,
    'model_family_in': functools.partial(make_model_family_dataframe, crossval=False),
    'model_family_cv': functools.partial(make_model_family_dataframe, crossval=True),
    'family_weight'  : make_model_family_weight_dataframe,
}


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
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)
