import argparse
import inspect
import itertools
import os

import numpy as np
import pandas as pd
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.behaviour as behaviour
import EFC_learningfMRI.G_matrix as G_matrix

# Days of the experiment.
N_SESSIONS = 24

# Descriptor columns.
ID_COLS = ['subNum', 'BN', 'TN', 'Repetition', 'chordID', 'chord', 'session', 'session_type', 'week']

# Performance measures kept in the behaviour.* tables.
PERF_COLS = ['trialPoint', 'ET', 'MD']

# Force columns
FORCE_MEASURES = ['raw', 'abs', 'der',]
FORCE_COLS = {m: [f'{f}_{m}' for f in gl.fingers] for m in FORCE_MEASURES}

# Descriptors kept in the force tables.
FORCE_ID_COLS = ['subNum', 'TN', 'BN', 'session', 'chord', 'chordID', 'Repetition', 'session_type', 'week']

# Groupings
SESSION_BY = ['subNum', 'session', 'chord', 'session_type', 'week']
BLOCK_BY = ['subNum', 'BN', 'session', 'chord', 'chordID', 'Repetition', 'session_type', 'week']

TRIAL     = 'behaviour.trial.tsv'
BSESS     = 'behaviour.session.tsv'
BSESS_REP = 'behaviour.session.repetition.tsv'
FWIDE     = 'force.trial.wide.tsv'
FFMRI     = 'force.run.wide.tsv'
FLONG     = 'force.trial.long.tsv'
FSESS     = 'force.session.avg.tsv'
FSESS_REP = 'force.session.repetition.avg.tsv'


def behaviour_single_session(sns=gl.participants, sessions=None):
    """STTSV: parse the raw .dat/.mov files into one table per participant x session.

    `sn` defaults to every participant, `sessions` to all `N_SESSIONS` days.
    """
    # sns      = gl.participants if sn is None else [sn]
    sessions = range(1, N_SESSIONS + 1) if sessions is None else sessions

    for sn in sns:
        for session in sessions:
            behaviour.analyse_session(sn, session)


def behaviour_by_trial(n_sessions=N_SESSIONS):
    """TRIAL: all single trials of all participants."""
    df = behaviour.concatenate_sessions(n_sessions, ID_COLS)
    df.to_csv(os.path.join(gl.baseDir, gl.behavDir, TRIAL), sep='\t', index=False)


def performance_by_session():
    """PERF / PERF_REP: trial-wise -> session-wise performance."""
    trial    = behaviour.load(TRIAL, ID_COLS)[SESSION_BY + ['Repetition'] + PERF_COLS]
    df_norep = behaviour.group_trials_by(trial, SESSION_BY)
    df_rep   = behaviour.group_trials_by(trial, SESSION_BY + ['Repetition'])
    df_norep.to_csv(os.path.join(gl.baseDir, gl.behavDir, BSESS), sep='\t', index=False)
    df_rep.to_csv(os.path.join(gl.baseDir, gl.behavDir, BSESS_REP), sep='\t', index=False)


def force_by_trial_wide():
    """FWIDE: trial-wise force, one column per finger."""
    trial = behaviour.load(TRIAL, ID_COLS)
    force_wide = trial[FORCE_ID_COLS + ['trialPoint'] + [c for m in FORCE_MEASURES for c in FORCE_COLS[m]]]
    force_wide.to_csv(os.path.join(gl.baseDir, gl.behavDir, FWIDE), sep='\t', index=False)


def force_by_run_wide():
    """FFMRI: trial-wise -> run-wise force, scanning sessions only."""
    force_wide = behaviour.load(FWIDE, ID_COLS)
    force_fmri = force_wide[force_wide.session_type == 'scanning']
    force_fmri = force_fmri.groupby(BLOCK_BY, observed=True).mean(numeric_only=True).reset_index()
    force_fmri.to_csv(os.path.join(gl.baseDir, gl.behavDir, FFMRI), sep='\t', index=False)


def force_by_trial_long():
    """FLONG: trial-wise force, one row per trial x finger."""
    force_wide = behaviour.load(FWIDE, ID_COLS)
    force_long = behaviour.force_wide_to_long(force_wide, FORCE_ID_COLS, FORCE_COLS)
    force_long.to_csv(os.path.join(gl.baseDir, gl.behavDir, FLONG), sep='\t', index=False)


def force_by_session_avg():
    """FSESS / FSESS_REP: trial-wise -> session-wise force.

    Averaged over fingers as well as trials, since the fingers are stacked.
    """
    force_long = behaviour.load(FLONG, ID_COLS)

    force_long_succ = force_long[force_long.trialPoint == 1].drop(columns='trialPoint')
    sess_force      = force_long_succ.groupby(SESSION_BY, observed=True).mean(numeric_only=True).reset_index()
    sess_force.to_csv(os.path.join(gl.baseDir, gl.behavDir, FSESS), sep='\t', index=False)

    # Same success filter as FSESS: failed trials have much lower force, and the
    # success rate differs by repetition (day 1: .33 rep1 vs .39 rep2), so mixing
    # them in would make the repetition contrast partly a success-rate contrast.
    sess_rep_force = force_long_succ.groupby(SESSION_BY + ['Repetition'], observed=True).mean(numeric_only=True).reset_index()
    sess_rep_force.to_csv(os.path.join(gl.baseDir, gl.behavDir, FSESS_REP), sep='\t', index=False)


def calc_G_force(sns=gl.participants, metrics=('raw', 'abs', 'der'), sessions=('all', *gl.sessions), repetitions=('all', 1, 2)):
    """The same Gs as calc_G_rois, but over the five fingers' force instead of voxels.

    Fingers take the place of the voxels, so the matrices come out with the same
    layout as the ROI ones — 8x8 within a session, 24x24 across, trained chords
    first — and land next to them in the pcm directory as
    ``G_obs.<epoch>.force.<metric>.npy``.
    """
    force = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, 'force.run.wide.tsv'), sep='\t')

    for metric in metrics:
        for sn in sns:
            for session in sessions:
                for repetition in repetitions:

                    print(f'force {metric}, doing participant {sn}...')

                    data, cond_vec, part_vec = behaviour.force_patterns(force, sn, metric, session=session, repetition=repetition)

                    G        = G_matrix.calc_G(data, cond_vec, part_vec, centred=False)
                    cov      = G_matrix.calc_G(data, cond_vec, part_vec, centred=True)
                    G_raw    = G_matrix.calc_G(data, cond_vec, part_vec, centred=False, fixed_effect=False)
                    G_noxval = G_matrix.calc_G(data, cond_vec, part_vec, centred=False, fixed_effect=False, crossval=False)

                    save_path = os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}')
                    os.makedirs(save_path, exist_ok=True)

                    fname = G_matrix.make_fname(session, repetition)

                    np.save(os.path.join(save_path, f'G_obs.{fname}.force.{metric}'), G)
                    np.save(os.path.join(save_path, f'cov.{fname}.force.{metric}'), cov)
                    np.save(os.path.join(save_path, f'G_obs_raw.{fname}.force.{metric}'), G_raw)
                    np.save(os.path.join(save_path, f'G_obs_noxval.{fname}.force.{metric}'), G_noxval)


def make_force_distance_dataframe(metrics=('raw', 'abs', 'der'), sns=gl.participants, ref_session=3, crossval=False):
    """The neural dataframe's counterpart over the force Gs written by pattern_G_matrix.

    Same rows, same columns, with ``metric`` (the force measure) standing in for
    ``Hem``/``roi``: the force Gs have the identical 8x8 trained-first layout, so
    every chord pair lines up one-to-one with its neural row.
    """
    sns = gl.participants if sns is None else sns

    rows = []
    for metric, sess in itertools.product(metrics, gl.sessions):
        for sn in sns:

            print(f'doing participant {sn}, session {sess}, force {metric}...')

            G = np.load(os.path.join(gl.baseDir, gl.pcmDir, f'subj{sn}', f'G_obs_raw.within_session.{sess}.force.{metric}.npy'))
            rows += G_matrix.G_rows(G, sn, metric=metric, session=sess)

    df = pd.DataFrame(rows)
    df = G_matrix.add_group_reference(df, ['metric', 'pair'], ref_session, crossval)
    df.to_csv(os.path.join(gl.baseDir, gl.pcmDir, 'dissimilarity.within_session.force.tsv'), sep='\t', index=False)


# Step name -> function, in the order the full run does them. The behaviour/force steps
# write a tsv, so their key is the output file's stem: <domain>_<unit>[_<shape>]; the last
# two build the force Gs and their dissimilarity table (pcm dir).
FUNC = {
    'parse_sessions'   : behaviour_single_session,
    'behaviour_trial'  : behaviour_by_trial,
    'behaviour_session': performance_by_session,
    'force_trial_wide' : force_by_trial_wide,
    'force_run_wide'   : force_by_run_wide,
    'force_trial_long' : force_by_trial_long,
    'force_session'    : force_by_session_avg,
    'G_force'          : calc_G_force,
    'make_force_distance_dataframe': make_force_distance_dataframe,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`sns=`, `sessions=`, ...), but only the ones it
    actually takes -- most steps read the tsv the previous one wrote and take nothing.
    """
    if what is not None:
        func     = FUNC[what]                                       # select function
        accepted = inspect.signature(func).parameters               # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted})  # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse the raw behavioural files and build the trial, session and force tables.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--sns', nargs='+', type=int, default=gl.participants, help='participant numbers, parse_sessions only (default: all participants)')
    parser.add_argument('--sessions', nargs='+', type=int, default=None, help='session numbers, parse_sessions only (default: all sessions)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        # main('parse_sessions', **kwargs)
        main('behaviour_trial',   **kwargs)
        main('behaviour_session', **kwargs)
        main('force_trial_wide',  **kwargs)
        main('force_run_wide',    **kwargs)
        main('force_trial_long',  **kwargs)
        main('force_session',     **kwargs)
