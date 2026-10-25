import argparse
import os
import pickle
import itertools
import numpy as np
import PcmPy as pcm
import pandas as pd
import inspect
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.betas as betas
import EFC_learningfMRI.pcm as pcm_


def fit_component_model_rois(sns            =gl.participants, 
                             glm            = 3,
                             atlas_name     = 'ROI',
                             residual_fname = 'residual.dtseries.nii',
                             comp_names     = ['type', 'trained', 'untrained', 'MD', 'finger', 'pattern'],
                             sessions       = (3, 9, 23)):

    """Fit the PCM component model per (subject, Hem, roi, session).

    Prewhitens the betas once per (subject, roi) with BetasPrewithenedLoader, then
    fits the component model (which includes 'base', the group-mean observed G of
    session 3) and pickles each fit's ``theta_in`` next to the participant's Gs, as
    ``component_model.theta_in.<atlas_name>.glm<glm>.<session>.<Hem>.<roi>.p`` --
    which component_summary reads back.
    """
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    pcm_.fit_component_model(loader, comp_names=comp_names, sessions=sessions)


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

        # log Bayes factor of each component from the model family fit of the same cell
        comp = '_'.join(comp_names)
        stem = f'{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'
        with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.theta.{stem}'), 'rb') as f:
            fam_names = list(pickle.load(f)['comp_names'])
        with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.T.{stem}'), 'rb') as f:
            T_mf = pickle.load(f)

        # only the family's structure is needed, which zero matrices with the same names reproduce
        MF = pcm.model.ModelFamily(np.zeros((len(fam_names), 8, 8)), comp_names=fam_names)
        bf = MF.component_bayesfactor(T_mf.likelihood, method='AIC')[0]
        bf = dict(zip(fam_names, bf))          # the family's base components are in every model: no Bayes factor

        df_tmp = pd.DataFrame({'weight'   : weight.squeeze(), 
                               'component': comp_names,
                               'log_bf'   : [bf.get(c, np.nan) for c in comp_names]})
        df_tmp['sn']      = sn
        df_tmp['Hem']     = H
        df_tmp['roi']     = roi
        df_tmp['session'] = session
        df = pd.concat([df, df_tmp], ignore_index=True)

    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'component_model.{atlas_name}.glm{glm}.{comp}.tsv'), sep='\t', index=False)


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


def make_model_family_dataframe(sns=gl.participants, glm=3, atlas_name='ROI'):
    """Likelihood of every model of the model family, one row per (sn, session, Hem, roi).

    The columns are the family's model names, in the family's order, so the mean over
    participants of a cell can go straight into ``PcmPy.vis.plot_tree``. The tsv is
    named after the component set the family fit stored with its theta.
    """
    rois = gl.rois[atlas_name]

    LL = []
    for sn, session, H, roi in itertools.product(sns, gl.sessions, gl.Hem, rois):
        stem = f'{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'
        with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.T.{stem}'), 'rb') as f:
            T = pickle.load(f)
        with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_family.theta.{stem}'), 'rb') as f:
            fit = pickle.load(f)

        ll            = T.likelihood.copy()
        ll['sn']      = sn
        ll['session'] = session
        ll['Hem']     = H
        ll['roi']     = roi
        LL.append(ll)

    LL = pd.concat(LL, ignore_index=True)
    LL.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_family.likelihood.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


# Step name -> function: the PCM fits, then the tsvs collected from them.
FUNC = {
    'fit'         : fit_component_model_rois,
    'weight'      : make_component_weight_dataframe,
    'likelihood'   : make_likelihood_dataframe,
    'model_family': make_model_family_dataframe,
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
    parser.add_argument('--comp_names', nargs='+', type=str, default=['trained', 'untrained', 'MD', 'finger', 'pattern'], help='components to include')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)
