import argparse
import inspect

import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.searchlight as sl

# GLM the betas come from.
GLM = 3

# The searchlight itself is run once and saves each centre's G here (one .npy per
# subject/session/hemisphere); every metric below is read off those Gs afterwards.
G_FNAME = 'searchlight_G'

# One output file per metric, per subject/session/hemisphere, plus the pooled group maps.
OUT_FNAME = {metric: f'searchlight_{metric}' for metric in sl.METRICS}


def define(sns=gl.participants):
    """DEFINE: build each participant's surface-based searchlight (one .h5 per hemisphere).

    Run once per participant; every metric below reuses the same searchlight definition.
    """
    for sn in sns:
        print(f'defining searchlight for participant {sn}...')
        sl.make_searchlight(sn=sn)


def searchlight_G(sns=gl.participants, glm=GLM, sessions=gl.sessions):
    """G: run the searchlight and save each centre's crossvalidated G.

    One (n_centers, 8, 8) .npy per subject, session and hemisphere. The betas are whitened
    by the searchlight-local noise covariance (multivariate noise normalization), so this
    step needs the residual timeseries. It is the slow step, and it is metric-free: every
    distance below is computed from these Gs afterwards.
    """
    searchlight = sl.Searchlight(sns       = sns,
                                 glm       = glm,
                                 sessions  = sessions,
                                 out_fname = G_FNAME)
    searchlight.run()


def searchlight_distance(sns=gl.participants, glm=GLM, metric='crossnobis', sessions=gl.sessions):
    """DISTANCE: read the saved Gs and write one distance map per subject/session/hemisphere.

    `metric` is 'crossnobis' (pattern separation, scales with how strongly the region is
    driven), 'cosine', or 'theta' (the angle between chord patterns, invariant to the
    overall activity) -- all read off the same crossvalidated G, see `sl.G_to_metric`.
    Needs `searchlight_G` to have run for these subjects and sessions.
    """
    sl.make_distance_maps(sns       = sns,
                          glm       = glm,
                          metric    = metric,
                          sessions  = sessions,
                          fname   = G_FNAME,
                          out_fname = OUT_FNAME[metric])


def make_group_maps(sns=gl.participants, glm=GLM, metric='crossnobis', sessions=gl.sessions):
    """SUMMARY: pool one metric's subject maps into group maps.

    Writes the group mean of every column, plus the group mean of the within-subject
    trained-minus-untrained difference (`<fname>_diff.*`), per session and hemisphere.
    """
    sl.pool_searchlight(sns=sns, glm=glm, mask_p=None, fname=OUT_FNAME[metric], sessions=sessions)


# Step name -> function, in the order the full run does them. The metric is a `--metric`
# flag rather than one key per metric, so the keys stay the same as metrics are added.
FUNC = {
    'define'        : define,
    'searchlight_G' : searchlight_G,
    'distance'      : searchlight_distance,
    'distance_group': make_group_maps,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`sns=`, `glm=`, `metric=`, ...), but only the ones
    it accepts -- `define` takes only `sns`.
    """
    if what is not None:
        func     = FUNC[what]                                       # select function
        accepted = inspect.signature(func).parameters               # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted})  # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define and run the surface searchlight, and pool it across participants.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all but define)')
    parser.add_argument('--sns', nargs='+', type=int, default=gl.participants, help='participant ids to include in the analysis')
    parser.add_argument('--glm', type=int, default=GLM, help='GLM the betas come from')
    parser.add_argument('--sessions', nargs='+', type=int, default=gl.sessions, help='session numbers to run')
    parser.add_argument('--metric', default='crossnobis', choices=list(sl.METRICS), help='distance metric, the distance* steps only')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k not in ('what', 'pw') and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        for metric in sl.METRICS:
    #         main('distance',       **{**kwargs, 'metric': metric})
             main('distance_group', **{**kwargs, 'metric': metric})
