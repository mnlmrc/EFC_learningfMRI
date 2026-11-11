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


def fit_model_transfer(sns            = gl.participants,
                       glm            = 3,
                       atlas_name     = 'ROI',
                       residual_fname = 'residual.dtseries.nii',
                       comp_names     = None,
                       learn_comps    = ('type', 'trained', 'untrained'),
                       ref_session    = 3,
                       sessions       = (3, 9, 23)):
    """Likelihood of every session's data under the component weights fitted in ``ref_session``,
    with and without the learning components free (:func:`pcm.fit_model_transfer`).

    ``comp_names`` picks the model of the family whose weights carry over (any order);
    the default is the family's full model. ``learn_comps`` are the components that
    capture learning, keys of ``models.make_G_dict``. Needs the family fits of
    ``ref_session`` (step ``fit_indiv``). Collect the pickles with step ``transfer``.
    """
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    pcm_.fit_model_transfer(loader, comp_names=comp_names, learn_comps=learn_comps, ref_session=ref_session, sessions=sessions)


def make_transfer_dataframe(sns=gl.participants, glm=3, atlas_name='ROI', comp_names=None, learn_comps=('type', 'trained', 'untrained'), ref_session=3, sessions=(3, 9, 23)):
    """Likelihoods of step ``fit_transfer`` next to the family fit of the same cell,
    one row per (sn, session, Hem, roi).

    Columns, all from fits to the same session's data, without a scale:
        transfer        the ``ref_session`` G, scaled                         (``model_transfer.T``)
        transfer_learn  transfer + learn_comps                                (``model_transfer.T``)
        free_learn      comp_names + learn_comps, weights fitted here         (``model_transfer.T``)
        null            the family's empty model, ``base``                    (``model_family.T_in``)
        identity        identity alone                                        (``model_family.T_in``)
        free            the comp_names member, weights fitted here            (``model_family.T_in``)
        n_comp          len(comp_names)
        n_learn         len(learn_comps)

    Free G parameters, for an AIC-type penalty: identity 1, transfer 1, transfer_learn
    ``1 + n_learn``, free ``n_comp``, free_learn ``n_comp + n_learn`` (one fewer effective
    with identity, trained and untrained all in, see :func:`pcm.fit_model_transfer`).
    ``free - transfer`` is the change of the geometry; ``free_learn - transfer_learn`` is
    the part of it that the learning components cannot account for.

    The pickles are named after the components in the family's stored order, which is
    read off each subject's reference fit, so ``comp_names`` can be in any order.
    Written to ``model_transfer.<pcm.transfer_stem>.tsv`` in the pcm dir.
    """
    rois = gl.rois[atlas_name]

    df = []
    for sn, H, roi in itertools.product(sns, gl.Hem, rois):
        path     = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}')
        comps, _ = pcm_.family_weights(sn, f'{atlas_name}.glm{glm}.{ref_session}.{H}.{roi}.p', comp_names)
        stem     = pcm_.transfer_stem(atlas_name, glm, ref_session, comps, learn_comps)

        for session in sessions:
            with open(os.path.join(path, f'model_transfer.T.{stem}.{session}.{H}.{roi}.p'), 'rb') as f:
                T = pickle.load(f)
            with open(os.path.join(path, f'model_family.T_in.{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'), 'rb') as f:
                T_mf = pickle.load(f)

            df.append({**{m: T.likelihood[m].iloc[0] for m in ('transfer', 'transfer_learn', 'free_learn')},
                       'null'       : T_mf.likelihood['base'].iloc[0],
                       'identity'   : T_mf.likelihood['identity'].iloc[0],
                       'free'       : T_mf.likelihood['+'.join(comps)].iloc[0],
                       'n_comp'     : len(comps),
                       'n_learn'    : len(learn_comps),
                       'ref_session': ref_session,
                       'sn'         : sn,
                       'session'    : session,
                       'Hem'        : H,
                       'roi'        : roi})

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_transfer.{stem}.tsv'), sep='\t', index=False)


def fit_model_transfer_null(sns            = gl.participants,
                            glm            = 3,
                            atlas_name     = 'ROI',
                            residual_fname = 'residual.dtseries.nii',
                            comp_names     = None,
                            learn_comps    = ('type', 'trained', 'untrained'),
                            ref_session    = 3,
                            sessions       = (3, 9, 23),
                            n_sim          = 100,
                            seed           = 0):
    """Null distribution of the transfer metrics under a geometry that does not change
    (:func:`pcm.fit_model_transfer_null`): ``n_sim`` simulations per cell, from G_ref
    with each session's own scale and noise.

    Same ``comp_names``/``learn_comps``/``ref_session`` as step ``fit_transfer``, whose
    real metrics it is the null for. Needs the family fits of ``ref_session`` (step
    ``fit_indiv``). Collect the pickles with step ``transfer_null``.
    """
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    pcm_.fit_model_transfer_null(loader, comp_names=comp_names, learn_comps=learn_comps, ref_session=ref_session,
                                 sessions=sessions, n_sim=n_sim, seed=seed)


def make_transfer_null_dataframe(sns=gl.participants, glm=3, atlas_name='ROI', comp_names=None, learn_comps=('type', 'trained', 'untrained'), ref_session=3):
    """Likelihoods of step ``fit_transfer_null``, one row per (sn, session, Hem, roi, sim).

    Columns transfer, transfer_learn, free and free_learn are the models of step
    ``transfer`` (free refitted here, it is the family member there), fitted to data
    simulated with the same geometry in every session; compute the metrics from them as
    from the real ones (``model_transfer.<stem>.tsv``) and compare within each sn,
    session, Hem and roi. ``scale`` and ``noise`` are the generating signal and noise of
    that session, ``n_comp`` and ``n_learn`` as in step ``transfer``. Every session but
    ``ref_session`` is there. Written to ``model_transfer_null.<pcm.transfer_stem>.tsv``
    in the pcm dir.
    """
    rois = gl.rois[atlas_name]

    df = []
    for sn, H, roi in itertools.product(sns, gl.Hem, rois):
        comps, _ = pcm_.family_weights(sn, f'{atlas_name}.glm{glm}.{ref_session}.{H}.{roi}.p', comp_names)
        stem     = pcm_.transfer_stem(atlas_name, glm, ref_session, comps, learn_comps)

        with open(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'model_transfer_null.{stem}.{H}.{roi}.p'), 'rb') as f:
            null = pickle.load(f)

        df.append(null['likelihood'].merge(null['generating'], on='session')
                                    .assign(n_comp=len(comps), n_learn=len(learn_comps), ref_session=ref_session,
                                            sn=sn, Hem=H, roi=roi))

    df = pd.concat(df, ignore_index=True)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_transfer_null.{stem}.tsv'), sep='\t', index=False)


def fit_model_transfer_chord(sns            = gl.participants,
                             glm            = 3,
                             atlas_name     = 'ROI',
                             residual_fname = 'residual.dtseries.nii',
                             comp_names     = None,
                             ref_session    = 3,
                             sessions       = (3, 9, 23)):
    """The transfer test separately for the trained and the untrained chords
    (:func:`pcm.fit_model_transfer_chord`). Collect the pickles with step ``transfer_chord``.
    """
    loader = betas.BetasPrewithenedLoader(sns=sns, glm=glm, atlas_name=atlas_name, residual_fname=residual_fname)
    pcm_.fit_model_transfer_chord(loader, comp_names=comp_names, ref_session=ref_session, sessions=sessions)


def make_transfer_chord_dataframe(sns=gl.participants, glm=3, atlas_name='ROI', comp_names=None, ref_session=3, sessions=(3, 9, 23)):
    """Likelihoods of step ``fit_transfer_chord``, one row per (sn, chord, session, Hem, roi).

    Columns identity, transfer and free (see :func:`pcm.fit_model_transfer_chord`), plus
    ``n_comp``. ``free - transfer`` is what does not transfer (0 in ``ref_session`` by
    construction); ``transfer - identity`` goes negative when the reference structure fits
    worse than none. Avoid the ratio of the two to ``free - identity``, which blows up when a
    chord set has no structure left beyond identity. Written to
    ``model_transfer_chord.<atlas_name>.glm<glm>.ref<ref_session>.<comps>.tsv`` in the pcm dir.
    """
    rois = gl.rois[atlas_name]

    df = []
    for sn, H, roi in itertools.product(sns, gl.Hem, rois):
        path     = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}')
        comps, _ = pcm_.family_weights(sn, f'{atlas_name}.glm{glm}.{ref_session}.{H}.{roi}.p', comp_names)
        member   = '+'.join(comps)

        for chord, session in itertools.product(('trained', 'untrained'), sessions):
            with open(os.path.join(path, f'model_transfer_chord.T.{atlas_name}.glm{glm}.ref{ref_session}.{member}.{chord}.{session}.{H}.{roi}.p'), 'rb') as f:
                T = pickle.load(f)

            df.append({**{m: T.likelihood[m].iloc[0] for m in ('identity', 'transfer', 'free')},
                       'n_comp'     : len(comps),
                       'ref_session': ref_session,
                       'sn'         : sn,
                       'chord'      : chord,
                       'session'    : session,
                       'Hem'        : H,
                       'roi'        : roi})

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'model_transfer_chord.{atlas_name}.glm{glm}.ref{ref_session}.{member}.tsv'), sep='\t', index=False)


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
        stem      = f'{atlas_name}.glm{glm}.{session}.{H}.{roi}.p'
        comps, w  = pcm_.family_weights(sn, stem, comp_names)
        df.append(pd.DataFrame({'component': comps,
                                'weight'   : w,
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
    'fit_transfer'   : fit_model_transfer,
    'transfer'       : make_transfer_dataframe,
    'fit_transfer_null': fit_model_transfer_null,
    'transfer_null'  : make_transfer_null_dataframe,
    # 'fit_transfer_chord': fit_model_transfer_chord,
    # 'transfer_chord' : make_transfer_chord_dataframe,
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
    parser.add_argument('--learn_comps', nargs='+', default=None, type=str, help='components that capture learning, added on top of the transfer (steps fit_transfer, transfer; default: type trained untrained)')
    parser.add_argument('--ref_session', type=int, default=None, help='session whose component weights carry over to the others (steps fit_transfer*, transfer*; default 3)')
    parser.add_argument('--n_sim', type=int, default=None, help='simulations per cell (step fit_transfer_null; default 100)')
    parser.add_argument('--seed', type=int, default=None, help='seed of the simulations, combined with subject, Hem and ROI (step fit_transfer_null; default 0)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)
