import argparse
import functools
import inspect
import itertools
import os
import numpy as np
import pandas as pd
import EFC_learningfMRI.globals as gl


def _ancova_group(y, g, chord, session, h=None, sessions=gl.sessions):
    """Trained - untrained difference in ``y`` per session, adjusted for the covariates ``g`` and ``h``.

    ``g`` is (n_pairs, k): covariates with one slope shared by all sessions (the group
    geometry, the same in every session). ``h`` is (n_pairs, m): covariates with one slope
    per session (the subject's own force geometry, whose mapping onto the neural geometry
    may change with learning). Plus a chord effect and an intercept per session:
    X = [g | h x session | chord x session | session].

    Returns (slopes, slopes_session, slope_chord, intercept): slopes as a length-k array,
    slopes_session as {session: length-m array}, the last two as {session: beta}.
    """
    h = np.empty((len(y), 0)) if h is None else h
    c = chord.map({'trained'  : 1, 'untrained': -1}).to_numpy()
    D = np.stack([(session == s).to_numpy(dtype=float) for s in sessions], axis=1)

    k, m, n = g.shape[1], h.shape[1], len(sessions)

    # column j * n + s is covariate j in session s
    Hs = (h[:, :, None] * D[:, None, :]).reshape(len(y), m * n)
    X  = np.c_[g, Hs, c[:, None] * D, D]
    B  = np.linalg.pinv(X) @ y

    slopes_session = {s: B[k + np.arange(m) * n + i] for i, s in enumerate(sessions)}
    o = k + m * n
    return B[:k], slopes_session, dict(zip(sessions, B[o:o + n])), dict(zip(sessions, B[o + n:]))


def _ancova_cell(cell, keys, metrics, scale, force_metric=(), group=True):
    """Adjusted dissimilarities of one subject's cell (all sessions), one column set per metric.

    The group geometry gets one slope for all sessions, each force metric one slope per session.
    """
    adjusted = cell[keys + ['session', 'chord', 'pair']].copy()

    shared      = ['group'] if group else []
    per_session = [f'force_{fm}' for fm in force_metric]

    for metric in metrics:
        y = cell[metric].to_numpy()
        g = cell[[f'{metric}_{name}' for name in shared]].to_numpy()
        h = cell[[f'{metric}_{name}' for name in per_session]].to_numpy()

        if scale:
            y, g, h = y / y.mean(), g / g.mean(axis=0), h / h.mean(axis=0)

        slopes, slopes_session, slope_chord, intercept = _ancova_group(y, g, cell.chord, cell.session, h)
        S = np.stack([slopes_session[s] for s in cell.session])  # each pair's own session slopes, (n_pairs, m)

        # the pair's own dissimilarity with the covariates taken out, centred on the
        # cell's mean covariates so the adjusted values stay on the scale of y
        adjusted[metric]                  = y - (g - g.mean(axis=0)) @ slopes - ((h - h.mean(axis=0)) * S).sum(axis=1)
        adjusted[f'{metric}_slope_chord'] = cell.session.map(slope_chord)
        for name, slope in zip(shared, slopes):
            adjusted[f'{metric}_slope_{name}'] = slope
        for j, name in enumerate(per_session):
            adjusted[f'{metric}_slope_{name}'] = S[:, j]
        adjusted[f'{metric}_intercept']   = cell.session.map(intercept)

    return adjusted


def make_rois_ancova_group_dataframe(glm=3, atlas_name='ROI', rois=None, sns=gl.participants, metrics=('crossnobis', 'cosine'), scale=False):

    """Trained vs untrained, with the reference-session geometry regressed out.

    """
    if rois is None:
        rois = gl.rois[atlas_name]

    df = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t')
    df = df[df.chord.isin(['trained', 'untrained']) & df.session.isin(gl.sessions)]

    cells = []
    for sn, H, roi in itertools.product(sns, gl.Hem, rois):

        print(f'ancova, participant {sn}, {H}, {roi}...')

        cell = df[(df.sn==sn) & (df.Hem == H) & (df.roi == roi)]
        cells.append(_ancova_cell(cell, ['sn', 'Hem', 'roi'], metrics, scale))

    df_ancova = pd.concat(cells, ignore_index=True)

    df_ancova.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_ancova.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def make_rois_ancova_group_force_dataframe(glm=3, atlas_name='ROI', rois=None, sns=gl.participants, metrics=('crossnobis', 'cosine'), force_metric=('der',), scale=False):
    """As make_rois_ancova_group_dataframe, but also regresses out the subject's own force
    geometry in the same session, one covariate per entry of ``force_metric``, with its own
    slope in each session.

    The force dissimilarity of each chord pair (from make_force_distance_dataframe) is
    matched to the neural one on sn/session/pair, and the same metric is used for both
    (crossnobis on crossnobis, cosine on cosine).
    """
    if rois is None:
        rois = gl.rois[atlas_name]

    df = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t')
    df = df[df.chord.isin(['trained', 'untrained']) & df.session.isin(gl.sessions)]

    force = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t')
    for fm in force_metric:
        f  = force[force.metric == fm][['sn', 'session', 'pair', *metrics]]
        f  = f.rename(columns={m: f'{m}_force_{fm}' for m in metrics})
        df = df.merge(f, on=['sn', 'session', 'pair'], how='left', validate='many_to_one')

    cells = []
    for sn, H, roi in itertools.product(sns, gl.Hem, rois):

        print(f'ancova + force {force_metric}, participant {sn}, {H}, {roi}...')

        cell = df[(df.sn==sn) & (df.Hem == H) & (df.roi == roi)]
        cells.append(_ancova_cell(cell, ['sn', 'Hem', 'roi'], metrics, scale, force_metric))

    df_ancova = pd.concat(cells, ignore_index=True)

    df_ancova.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_ancova_force.{"-".join(force_metric)}.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def make_rois_ancova_force_dataframe(glm=3, atlas_name='ROI', rois=None, sns=gl.participants, metrics=('crossnobis', 'cosine'), force_metric=('der',), scale=False):
    """Trained vs untrained, with only the subject's own force geometry in the same session
    regressed out (no group geometry), one covariate per entry of ``force_metric``, with its
    own slope in each session.

    The force dissimilarity of each chord pair (from make_force_distance_dataframe) is
    matched to the neural one on sn/session/pair, and the same metric is used for both
    (crossnobis on crossnobis, cosine on cosine).
    """
    if rois is None:
        rois = gl.rois[atlas_name]

    df = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t')
    df = df[df.chord.isin(['trained', 'untrained']) & df.session.isin(gl.sessions)]

    force = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t')
    for fm in force_metric:
        f  = force[force.metric == fm][['sn', 'session', 'pair', *metrics]]
        f  = f.rename(columns={m: f'{m}_force_{fm}' for m in metrics})
        df = df.merge(f, on=['sn', 'session', 'pair'], how='left', validate='many_to_one')

    cells = []
    for sn, H, roi in itertools.product(sns, gl.Hem, rois):

        print(f'ancova, force {force_metric} only, participant {sn}, {H}, {roi}...')

        cell = df[(df.sn==sn) & (df.Hem == H) & (df.roi == roi)]
        cells.append(_ancova_cell(cell, ['sn', 'Hem', 'roi'], metrics, scale, force_metric, group=False))

    df_ancova = pd.concat(cells, ignore_index=True)

    df_ancova.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_ancova_force_only.{"-".join(force_metric)}.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


def make_force_ancova_group_dataframe(force_metrics=('raw', 'abs', 'der'), sns=gl.participants, metrics=('crossnobis', 'cosine'), scale=False):
    """The neural ancova's counterpart over the force dissimilarities written by make_force_distance_dataframe.

    Same model, fit once per subject x force metric instead of subject x Hem x roi.
    """
    df = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t')
    df = df[df.chord.isin(['trained', 'untrained']) & df.session.isin(gl.sessions)]

    cells = []
    for sn, force_metric in itertools.product(sns, force_metrics):

        print(f'ancova, participant {sn}, force {force_metric}...')

        cell = df[(df.sn==sn) & (df.metric == force_metric)]
        cells.append(_ancova_cell(cell, ['sn', 'metric'], metrics, scale))

    df_ancova = pd.concat(cells, ignore_index=True)

    df_ancova.to_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity_ancova.within_session.force.tsv'), sep='\t', index=False)


def _force_prediction_cell(cell, metrics, ref_session, group=False):
    """Fit neural ~ intercept + slope * force (+ slope * group) on the ``ref_session`` pairs of
    one cell, then predict every session of the cell from its own force geometry (and the
    ref-session group geometry, the same in every session).

    """
    names     = ['force'] + (['group'] if group else [])
    predicted = cell.drop(columns=[f'{m}_{name}' for m in metrics for name in names]).copy()
    ref       = (cell.session == ref_session).to_numpy()

    for metric in metrics:
        y = cell[metric].to_numpy()
        X = np.c_[np.ones(len(cell)), cell[[f'{metric}_{name}' for name in names]].to_numpy()]
        B = np.linalg.pinv(X[ref]) @ y[ref]

        for n, name in enumerate(names):
            predicted[f'{metric}_{name}'] = X[:, n + 1]

        predicted[f'{metric}_pred']      = X @ B
        predicted[f'{metric}_resid']     = y - X @ B
        predicted[f'{metric}_intercept'] = B[0]
        
        for n, name in enumerate(names):
            predicted[f'{metric}_slope_{name}'] = B[n + 1]

    return predicted


def make_rois_force_prediction_dataframe(glm=3, atlas_name='ROI', rois=None, sns=gl.participants, metrics=('crossnobis', 'cosine'),
                                         force_metrics=('abs', 'der'), ref_session=3, chords=('trained', 'untrained'), group=False):

    """Neural geometry predicted from the subject's own force geometry, with the mapping learnt in ``ref_session``.

    With ``group``, the ref-session group geometry of each pair (the *_group columns) enters
    the model as a second covariate.
    """
    if rois is None:
        rois = gl.rois[atlas_name]

    df = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t')
    df = df[df.chord.isin(chords) & df.session.isin(gl.sessions)]
    df = df[['sn', 'Hem', 'roi', 'session', 'chord', 'pair', *metrics, *([f'{m}_group' for m in metrics] if group else [])]]

    force = pd.read_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t')

    model = 'force_group' if group else 'force'

    cells = []
    for force_metric in force_metrics:
        f  = force[force.metric == force_metric][['sn', 'session', 'pair', *metrics]]
        f  = f.rename(columns={m: f'{m}_force' for m in metrics})
        dm = df.merge(f, on=['sn', 'session', 'pair'], how='left', validate='many_to_one')
        dm.insert(3, 'force_metric', force_metric)

        for sn, H, roi in itertools.product(sns, gl.Hem, rois):

            print(f'{model} prediction ({force_metric}, fit on session {ref_session}), participant {sn}, {H}, {roi}...')

            cell = dm[(dm.sn == sn) & (dm.Hem == H) & (dm.roi == roi)]
            cells.append(_force_prediction_cell(cell, metrics, ref_session, group))

    df_pred = pd.concat(cells, ignore_index=True)

    df_pred.to_csv(os.path.join(gl.baseDir, gl.pcmDir, f'dissimilarity_{model}_prediction.ref{ref_session}.within_session.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)

    return df_pred





# Step name -> function.
FUNC = {
    'make_rois_ancova_dataframe'      : make_rois_ancova_group_dataframe,
    'make_force_ancova_dataframe'     : make_force_ancova_group_dataframe,
    'make_rois_ancova_group_force_dataframe': make_rois_ancova_group_force_dataframe,
    'make_rois_ancova_force_dataframe'      : make_rois_ancova_force_dataframe,
    'make_rois_force_prediction_dataframe'  : make_rois_force_prediction_dataframe,
    'make_rois_force_group_prediction_dataframe': functools.partial(make_rois_force_prediction_dataframe, group=True),
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
