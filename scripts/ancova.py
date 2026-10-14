import argparse
import functools
import inspect
import itertools
import os
import numpy as np
import pandas as pd
import EFC_learningfMRI.globals as gl


def _ancova_cell(cell, d, group=True, force=False):
    """Fit d ~ intercept[sess] + slope_chord[sess] * chord + slope_group * d_group
    + slope_force[sess] * d_force on all sessions of one cell, then take the covariates out
    of the observed distance.

    chord is coded +1 (trained) / -1 (untrained). Returns one long-format row per pair, with
    the observed distance in ``value`` and the adjusted one in ``adjusted``.
    """
    n = len(gl.sessions)
    y = cell[d].to_numpy()
    S = (cell.session.to_numpy()[:, None] == np.array(gl.sessions)).astype(float)  # session dummies, (n_pairs, n)
    c = cell.chord.map({'trained': 1, 'untrained': -1}).to_numpy()[:, None]
    i = S.argmax(axis=1)                                                            # each pair's session index

    X = [S, c * S]                     # intercept and chord effect, one per session
    if group:
        g = cell[f'{d}_group'].to_numpy()
        X.append(g[:, None])           # group geometry, one slope for all sessions
    if force:
        f = cell[f'{d}_force'].to_numpy()
        X.append(f[:, None] * S)       # force geometry, one slope per session
    B = np.linalg.pinv(np.hstack(X)) @ y

    # covariates are centred on the cell's mean so the adjusted values stay on the scale of y
    adjusted = y.copy()
    if group:
        slope_group = B[2 * n]
        adjusted   -= slope_group * (g - g.mean())
    if force:
        slope_force = B[-n:][i]
        adjusted   -= slope_force * (f - f.mean())

    fit = cell.filter(['sn', 'Hem', 'roi', 'force_metric', 'session', 'chord', 'pair']).copy()
    fit['distance'] = d
    fit['value']    = y
    fit['adjusted'] = adjusted
    if group:
        fit['group']       = g
        fit['slope_group'] = slope_group
    if force:
        fit['force']       = f
        fit['slope_force'] = slope_force
    fit['slope_chord'] = B[n:2 * n][i]
    fit['intercept']   = B[:n][i]

    return fit


def make_rois_ancova_group_dataframe(glm      = 3,
                                     atlas    = 'ROI',
                                     rois     = None,
                                     sns      = gl.participants,
                                     distance = ('crossnobis', 'cosine')):

    """Trained vs untrained, with the ref-session group geometry regressed out.

    Long format: one row per pair and distance, with the distance in the ``distance`` column.
    """
    if rois is None:
        rois = gl.rois[atlas]

    dist_neural = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas}.glm{glm}.tsv'), sep='\t')
    dist_neural = dist_neural[dist_neural.chord.isin(['trained', 'untrained']) & dist_neural.session.isin(gl.sessions)]

    cells = []
    for sn, H, roi, d in itertools.product(sns, gl.Hem, rois, distance):

        print(f'Ancova ({d}), participant {sn}, {H}, {roi}...')

        cell = dist_neural[(dist_neural.sn == sn) & (dist_neural.Hem == H) & (dist_neural.roi == roi)]
        cells.append(_ancova_cell(cell, d))

    df_ancova = pd.concat(cells, ignore_index=True)

    df_ancova.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_ancova_group.within_session.{atlas}.glm{glm}.tsv'), sep='\t', index=False)

    return df_ancova


def make_rois_ancova_force_dataframe(glm      = 3,
                                     atlas    = 'ROI',
                                     rois     = None,
                                     sns      = gl.participants,
                                     distance = ('crossnobis', 'cosine'),
                                     force    = ('abs', 'der',),
                                     group    = True):

    """Trained vs untrained, with the subject's own force geometry in the same session
    regressed out (one slope per session), and with ``group`` the ref-session group
    geometry too (one slope for all sessions).

    Each force metric is fit separately. The force distance of each chord pair (from
    make_force_distance_dataframe) is matched to the neural one on sn/session/pair, and the
    same distance is used for both (crossnobis on crossnobis, cosine on cosine).

    Long format: one row per pair, force metric and distance, with the distance in the
    ``distance`` column.
    """
    if rois is None:
        rois = gl.rois[atlas]

    dist_neural = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas}.glm{glm}.tsv'), sep='\t')
    dist_neural = dist_neural[dist_neural.chord.isin(['trained', 'untrained']) & dist_neural.session.isin(gl.sessions)]
    dist_force = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t')
    dist_force = dist_force[dist_force.force_metric.isin(force)]
    dist_force = dist_force.rename(columns={d: f'{d}_force' for d in distance})

    dm = dist_neural.merge(dist_force[['sn', 'session', 'pair', 'force_metric', *[f'{d}_force' for d in distance]]],
                           on=['sn', 'session', 'pair'], how='left')

    model = 'ancova_group_force' if group else 'ancova_force'

    cells = []
    for sn, H, roi, fm, d in itertools.product(sns, gl.Hem, rois, force, distance):

        print(f'{model.capitalize()} ({fm}, {d}), participant {sn}, {H}, {roi}...')

        cell = dm[(dm.sn == sn) & (dm.Hem == H) & (dm.roi == roi) & (dm.force_metric == fm)]
        cells.append(_ancova_cell(cell, d, force=True))

    df_ancova = pd.concat(cells, ignore_index=True)

    df_ancova.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_{model}.within_session.{atlas}.glm{glm}.tsv'), sep='\t', index=False)

    return df_ancova


def make_force_ancova_group_dataframe(sns      = gl.participants,
                                      distance = ('crossnobis', 'cosine'),
                                      force    = ('raw', 'abs', 'der')):

    """The neural ancova's counterpart over the force distances written by make_force_distance_dataframe.

    Same model, fit once per subject x force metric instead of subject x Hem x roi.
    Long format: one row per pair, force metric and distance.
    """
    dist_force = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t')
    dist_force = dist_force[dist_force.chord.isin(['trained', 'untrained']) & dist_force.session.isin(gl.sessions)]

    cells = []
    for sn, fm, d in itertools.product(sns, force, distance):

        print(f'Ancova ({fm}, {d}), participant {sn}...')

        cell = dist_force[(dist_force.sn == sn) & (dist_force.force_metric == fm)]
        cells.append(_ancova_cell(cell, d, ['sn', 'force_metric']))

    df_ancova = pd.concat(cells, ignore_index=True)

    df_ancova.to_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity_ancova_group.within_session.force.tsv'), sep='\t', index=False)

    return df_ancova


def _force_prediction_cell(cell, d, ref_sess):
    """Fit d ~ intercept + slope * d_force + slope * d_group + slope * chord on the ``ref_sess``
    pairs of one cell, then predict every session of the cell from its own force geometry
    (and the ref-session group geometry, the same in every session).

    chord is coded +1 (trained) / -1 (untrained). Returns one long-format row per pair, with
    the observed distance in ``value``.
    """
    c   = cell.chord.map({'trained': 1, 'untrained': -1}).to_numpy(dtype=float)
    X   = np.c_[np.ones(len(cell)), cell[f'{d}_force'], cell[f'{d}_group'],] # c]
    y   = cell[d].to_numpy()
    ref = (cell.session == ref_sess).to_numpy()
    B   = np.linalg.pinv(X[ref]) @ y[ref]

    fit = cell[['sn', 'Hem', 'roi', 'force_metric', 'session', 'chord', 'pair']].copy()
    fit['distance']    = d
    fit['value']       = y
    fit['force']       = X[:, 1]
    fit['group']       = X[:, 2]
    fit['pred']        = X @ B
    fit['resid']       = y - X @ B
    fit['intercept']   = B[0]
    fit['slope_force'] = B[1]
    fit['slope_group'] = B[2]
    # fit['slope_chord'] = B[3]

    return fit


def make_rois_force_prediction_dataframe(glm      = 3, 
                                         atlas    = 'ROI',
                                         rois     = None,
                                         sns      = gl.participants,
                                         distance = ('crossnobis', 'cosine'),
                                         force    = ('abs', 'der'),
                                         ref_sess = 3,
                                         chords   = ('trained', 'untrained')):

    """Neural geometry predicted from the subject's own force geometry, the ref-session group
    geometry and the chord (trained/untrained), with the mapping learnt in ``ref_sess``.

    Long format: one row per pair and distance, with the distance in the ``distance`` column.
    """
    if rois is None:
        rois = gl.rois[atlas]

    dist_neural = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas}.glm{glm}.tsv'), sep='\t')
    dist_neural = dist_neural[dist_neural.chord.isin(chords) & dist_neural.session.isin(gl.sessions)]
    dist_force = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t')
    dist_force = dist_force[dist_force.force_metric.isin(force)]
    dist_force = dist_force.rename(columns={d: f'{d}_force' for d in distance})

    dm = dist_neural.merge(dist_force[['sn', 'session', 'pair', 'force_metric', *[f'{d}_force' for d in distance]]],
                           on=['sn', 'session', 'pair'], how='left')

    cells = []
    for sn, H, roi, fm, d in itertools.product(sns, gl.Hem, rois, force, distance):

        print(f'Force prediction ({fm}, {d}, fit on session {ref_sess}), participant {sn}, {H}, {roi}...')

        cell = dm[(dm.sn == sn) & (dm.Hem == H) & (dm.roi == roi) & (dm.force_metric == fm)]
        cells.append(_force_prediction_cell(cell, d, ref_sess))

    df_pred = pd.concat(cells, ignore_index=True)

    df_pred.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_force_prediction.ref{ref_sess}.within_session.{atlas}.glm{glm}.tsv'), sep='\t', index=False)

    return df_pred





# Step name -> function.
FUNC = {
    'make_rois_ancova_group_dataframe'      : make_rois_ancova_group_dataframe,
    'make_force_ancova_group_dataframe'     : make_force_ancova_group_dataframe,
    'make_rois_ancova_group_force_dataframe': functools.partial(make_rois_ancova_force_dataframe, group=True),
    'make_rois_ancova_force_dataframe'      : make_rois_ancova_force_dataframe,
    'make_rois_force_prediction_dataframe'  : make_rois_force_prediction_dataframe,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`glm=`, `sns=`, ...), but only the ones it accepts.
    """
    if what is not None:
        func     = FUNC[what]                                       # select function
        accepted = inspect.signature(func).parameters               # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted})  # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trained vs untrained dissimilarities, with the group (and optionally the force) geometry regressed out.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas come from (default: the step default, 3)')
    parser.add_argument('--sns', nargs='+', type=int, default=gl.participants, help='participant ids to include in the analysis')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)
