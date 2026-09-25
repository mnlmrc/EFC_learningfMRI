import argparse
import inspect
import itertools
import os
import numpy as np
import pandas as pd
import EFC_learningfMRI.globals as gl


def _ancova_group(y, g, chord, session, sessions=gl.sessions):
    """Trained - untrained difference in ``y`` per session, adjusted for the covariates ``g``.

    ``g`` is (n_pairs, k): the group geometry, plus any other covariate. One slope per
    covariate shared by all sessions, a chord effect and an intercept per session:
    X = [g | chord x session | session].

    Returns (slopes, slope_chord, intercept), slopes as a length-k array, the last two
    as {session: beta}.
    """
    c = chord.map({'trained'  : 1, 'untrained': -1}).to_numpy()
    D = np.stack([(session == s).to_numpy(dtype=float) for s in sessions], axis=1)
    X = np.c_[g, c[:, None] * D, D]
    B = np.linalg.pinv(X) @ y

    k, n = g.shape[1], len(sessions)
    return B[:k], dict(zip(sessions, B[k:k + n])), dict(zip(sessions, B[k + n:]))


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
    geometry in the same session, one covariate per entry of ``force_metric``.

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
    regressed out (no group geometry), one covariate per entry of ``force_metric``.

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


def _ancova_cell(cell, keys, metrics, scale, force_metric=(), group=True):
    """Adjusted dissimilarities of one subject's cell (all sessions), one column set per metric.

    ``keys`` are the columns identifying the cell (sn/Hem/roi or sn/metric), copied
    onto the output next to session/chord/pair. Each entry of ``force_metric`` adds the
    ``{metric}_force_{fm}`` column as a covariate; ``group=False`` drops the group geometry.
    """
    adjusted = cell[keys + ['session', 'chord', 'pair']].copy()

    for metric in metrics:
        names = (['group'] if group else []) + [f'force_{fm}' for fm in force_metric]
        y     = cell[metric].to_numpy()
        g     = cell[[f'{metric}_{name}' for name in names]].to_numpy()

        if scale:
            y, g = y / y.mean(), g / g.mean(axis=0)

        slopes, slope_chord, intercept = _ancova_group(y, g, cell.chord, cell.session)

        # the pair's own dissimilarity with the covariates taken out, centred on the
        # cell's mean covariates so the adjusted values stay on the scale of y
        adjusted[metric]                  = y - (g - g.mean(axis=0)) @ slopes
        adjusted[f'{metric}_slope_chord'] = cell.session.map(slope_chord)
        for name, slope in zip(names, slopes):
            adjusted[f'{metric}_slope_{name}'] = slope
        adjusted[f'{metric}_intercept']   = cell.session.map(intercept)

    return adjusted


# Step name -> function.
FUNC = {
    'make_rois_ancova_dataframe'      : make_rois_ancova_group_dataframe,
    'make_force_ancova_dataframe'     : make_force_ancova_group_dataframe,
    'make_rois_ancova_group_force_dataframe': make_rois_ancova_group_force_dataframe,
    'make_rois_ancova_force_dataframe'      : make_rois_ancova_force_dataframe,
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
