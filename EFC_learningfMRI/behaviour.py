import argparse
import EFC_learningfMRI.globals as gl
import os
import pandas as pd
import numpy as np
import warnings

from EFC_learningfMRI.util import lowpass_fir, get_trained_and_untrained, make_cond_vec, session_index

def calc_md(X):
    N, m = X.shape
    F1 = X[0]
    FN = X[-1] - F1  # Shift the end point

    shifted_matrix = X - F1  # Shift all points

    d = list()

    for t in range(1, N - 1):
        Ft = shifted_matrix[t]

        # Project Ft onto the ideal straight line
        proj = np.dot(Ft, FN) / np.dot(FN, FN) * FN

        # Calculate the Euclidean distance
        d.append(np.linalg.norm(Ft - proj))

    d = np.array(d)
    MD = d.mean()

    return MD


def load_mov(filename):
    """
    load .mov file of one block

    :return:
    """

    try:
        with open(filename, 'rt') as fid:
            trial = 0
            A = []
            for line in fid:
                if line.startswith('Trial'):
                    #print(f'load_mov...reading trial: {trial}')
                    trial_number = int(line.split(' ')[1])
                    trial += 1
                    if trial_number != trial:
                        warnings.warn('Trials out of sequence')
                        trial = trial_number
                    A.append([])
                else:
                    # Convert line to a numpy array of floats and append to the last trial's list
                    data = np.fromstring(line, sep=' ')
                    if A:
                        A[-1].append(data)
                    else:
                        # This handles the case where a data line appears before any 'Trial' line
                        warnings.warn('Data without trial heading detected')
                        A.append([data])

            # Convert all sublists to numpy arrays
            mov = [np.array(trial_data) for trial_data in A]
            # # vizForce = [np.array(trial_data)[:, 9:] for trial_data in A]
            # state = [np.array(trial_data) for trial_data in A]

    except IOError as e:
        raise IOError(f"Could not open {filename}") from e

    return np.concatenate(mov, axis=0)


def force_patterns(force, sn, metric, session='all', repetition='all'):
    """Force patterns of one participant, laid out like the ROI betas.

    The five fingers take the place of the voxels, so the returned triple can be
    handed straight to :func:`calc_G`. The layout mirrors the neural pipeline:
    chords are relabelled 0-7 with this participant's *trained* chords first (see
    ``get_trained_and_untrained``), and the conditions are the same
    ``'session,chord'`` strings ``RegInfo.cond_vec`` builds — so the G comes out
    8x8 within a session and 24x24 across, with the same rows in the same order
    as the ROI Gs.

    ``session`` and ``repetition`` are filtered here rather than in ``calc_G``: the
    repetitions have to be averaged after the selection, not before. Pass the result
    to ``calc_G`` with its own filters left at 'all' — filtering twice is harmless
    (``runs_to_keep`` reads the session off ``cond_vec``, so it handles the force
    table's uneven runs), but it is the averaging here that does the work.

    Args:
        force:      the table from :func:`load_force`.
        sn:         participant number.
        metric:     force measure, one of 'abs', 'der', 'der_peak'.
        session:    3, 9, 23, or 'all'.
        repetition: 1, 2, or 'all'. With 'all' the two repetitions of a run are
                    averaged, giving one pattern per run x chord.

    Returns:
        (data, cond_vec, part_vec): ``data`` is (n_obs, 5), ``cond_vec`` the
        condition strings and ``part_vec`` the run numbers, made unique across
        sessions (BN restarts at 1 in every session).
    """
    cols = [f'{finger}_{metric}' for finger in gl.fingers]

    df = force[force.subNum == sn]
    if session != 'all':
        df = df[df.session == session]
    if repetition != 'all':
        df = df[df.Repetition == repetition]

    # one pattern per run x chord: averages the two repetitions when both are kept,
    # and is a no-op once a single repetition has been selected
    df = df.groupby(['session', 'BN', 'chordID'], as_index=False, observed=True)[cols].mean()

    data = df[cols].to_numpy()

    slot_of  = {chord: i for i, chord in enumerate(np.asarray(get_trained_and_untrained(sn), dtype=int))}

    sess_idx = df.session.map(session_index)
    cond_vec = make_cond_vec(sess_idx, df.chordID.astype(int).map(slot_of))
    part_vec = (sess_idx * 100 + df.BN).to_numpy()   # BN restarts at 1 every session

    return data, cond_vec, part_vec


def find_sustained_threshold_crossing(F, chordID, threshold_instr, threshold_uninstr, fsample, duration_ms=600):
    """
    Find the first timepoint at which all instructed fingers exceed threshold_instr while
    all uninstructed fingers stay below threshold_uninstr, and this holds for duration_ms.

    Parameters:
    - F: np.ndarray of shape (timepoints, fingers), absolute force
    - chordID: int, chord code (1 extension, 2 flexion, 9 uninstructed)
    - threshold_instr: float, force the instructed fingers must exceed
    - threshold_uninstr: float, force the uninstructed fingers must stay below
    - fsample: float, the sampling frequency in Hz
    - duration_ms: float, duration in milliseconds the condition must hold

    Returns:
    - int: index of the first timepoint meeting the condition, or -1 if not found
    """
    n_samples = int((duration_ms / 1000) * fsample)
    channels_instr   = [i for i, c in enumerate(str(chordID)) if c in ('1', '2')]
    channels_uninstr = [i for i, c in enumerate(str(chordID)) if c == '9']

    # True only if all instructed fingers are above and all uninstructed fingers below threshold
    in_target = np.all(F[:, channels_instr] > threshold_instr, axis=1) & \
                np.all(F[:, channels_uninstr] < threshold_uninstr, axis=1)

    # Convolve with a flat window of size n_samples
    sustained = np.convolve(in_target.astype(int), np.ones(n_samples, dtype=int), mode='valid')

    # Find first timepoint where all values in the window are True (i.e. sum == n_samples)
    idx = np.where(sustained == n_samples)[0]

    if len(idx) > 0:
        return int(idx[0])
    else:
        return -1


def _find_blocks(path, prefix, ext='.mov'):
    """find blocks available as .mov on a specified path and starting with a certain prefix."""
    files = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith(ext)]
    return sorted(int(f.split('_')[-1].split('.')[0]) for f in files)


def analyse_trial(F, chordID, n_ord=4, cutoff=5, fsample=gl.fsample['force']):

    # init trial as successful
    trialPoint = 1

    # filter raw forces
    F_filt = lowpass_fir(F, n_ord=n_ord, cutoff=cutoff, fsample=fsample, axis=0)
    
    # absolute force
    F_abs = np.abs(F_filt)

    # calculate derivative of filtered forces
    F_der = np.gradient(F_filt, 1 / fsample, axis=0)

    # absolute derivative
    F_der_abs    = np.abs(F_der)

    # # peak of absolute derivative
    # F_der_peak   = F_der_abs.max(axis=0)

    # find sample at which the participant start executing the chord and holds it for at least 600ms
    ET_samples = find_sustained_threshold_crossing(np.abs(F_filt), chordID, gl.ftarget, gl.fthresh, fsample=fsample)
    
    # if participant is unable to hold the chord for at least 600ms trial is unsuccessful
    if ET_samples < 0:
        trialPoint = 0

    # first sample at which any finger leaves the baseline area (±gl.fthresh)
    crossings  = np.flatnonzero(np.any(np.abs(F_filt) > gl.fthresh, axis=1))
    RT_samples = int(crossings[0]) if crossings.size else -1

    # if participant starts during plan time trial is unsuccessful
    if RT_samples < gl.fsample['force'] * gl.plan_time:
        trialPoint = 0

    # execution time should be always bigger than reaction time (in a succesful trial)
    if trialPoint==1:
        assert ET_samples > RT_samples

    # single trial metrics
    MD = calc_md(F_filt[RT_samples:ET_samples]) if trialPoint==1 else np.nan
    ET = (ET_samples - RT_samples) / gl.fsample['force'] if trialPoint==1 else np.nan
    RT = RT_samples / gl.fsample['force'] if trialPoint==1 else np.nan
    F_avg         = F_filt[RT_samples:].mean(axis=0)
    F_abs_avg     = F_abs[RT_samples:].mean(axis=0)
    F_der_abs_avg = F_der_abs[RT_samples:].mean(axis=0)

    trial_dict = {
        'trialPoint'   : trialPoint,
        'MD'           : MD,
        'ET'           : ET,
        'RT'           : RT,
        'F_avg'        : F_avg,
        'F_abs_avg'    : F_abs_avg,
        'F_der_abs_avg': F_der_abs_avg
    }

    return trial_dict


def analyse_session(sn, session):
    """Trial-wise metrics for one participant x session (writes efc4_<sn>_single_trial.tsv)."""

    # path to behavioural data
    path   = os.path.join(gl.baseDir, 'behavioural', f'day{session}')

    # force channels
    ch_idx = np.array(gl.diffCols)

    # load dat file
    dat = pd.read_csv(os.path.join(path, f'efc4_{sn}.dat'), sep='\t')

    # make sure we are looking at the correct .dat file
    assert dat.subNum.unique() == sn

    # retrieve trained and untrained chords for the participant (first 4 are trained)
    trained = get_trained_and_untrained(sn)[:4]

    # blocks available on path
    BN = _find_blocks(path, prefix=f'efc4_{sn}')
    
    rows = []

    # loop through available blocks
    for bn in BN:

        # .dat file rows for block bn
        dat_bn = dat[dat.BN == bn]

        # load .mov file for block bn and select only WAIT_PLAN and WAIT_EXEC
        mov = load_mov(os.path.join(path, f'efc4_{sn}_{int(bn):02d}.mov'))
        mov = mov[(mov[:, 1] == gl.wait_exec) | (mov[:, 1] == gl.wait_exec - 1)]

        # trials available in .mov file
        TN  = np.unique(mov[:, 0])

        # check that .dat and .mov file have the same number of trials
        assert TN.size == len(dat_bn)

        print(f'Processing... subj{sn}, session{session}, block {bn}, {TN.size} trials found...')

        # loop through trials
        prev_chordID = None
        for tn in TN:

            # finger forces for trial tn
            F = mov[mov[:, 0] == tn][:, ch_idx] * gl.fGain

            # .dat file row for trial tn (made a dictionary for convenience)
            dat_row = dat_bn[dat_bn.TN == tn].to_dict('records')[0]

            # calc performance metrics for a single trial
            trial_dict = analyse_trial(F, dat_row['chordID'])

            print(f"subj{sn}, session {session}, block {bn}, ntrial {tn + 1}, ET {trial_dict['ET']:.2f}s, trialPoint {trial_dict['trialPoint']}")

            row = {
                'subNum'      : dat_row['subNum'],
                'BN'          : dat_row['BN'],
                'Repetition'  : 2 if (prev_chordID is not None and prev_chordID == dat_row['chordID']) else 1,
                'TN'          : dat_row['TN'],
                'trialPoint'  : trial_dict['trialPoint'],
                'RT'          : trial_dict['RT'],
                'ET'          : trial_dict['ET'],
                'MD'          : trial_dict['MD'],
                'chordID'     : dat_row['chordID'],
                'chord'       : 'trained' if str(dat_row['chordID']) in trained else 'untrained',
                'session'     : dat_row['day'],
                'session_type': dat_row['session'],
                'week'        : dat_row['week'],
            }
            for i, f in enumerate(gl.fingers):
                row[f'{f}_raw'] = trial_dict['F_avg'][i]
                row[f'{f}_abs'] = trial_dict['F_abs_avg'][i]
                row[f'{f}_der'] = trial_dict['F_der_abs_avg'][i]

            rows.append(row)

            prev_chordID = dat_row['chordID']

    single_trial_metrics = pd.DataFrame(rows)
    single_trial_metrics.to_csv(os.path.join(path, f'efc4_{sn}_single_trial.tsv'), sep='\t', index=False)


def load(fname, id_cols):
    """Read one summary table back, restoring the identifier columns as categories.

    The cast is what `concatenate_sessions` applies, so a step gets the same
    dtypes whether it is run on its own or right after the step that wrote its
    input.
    """
    df = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, fname), sep='\t')
    cat_cols = [c for c in id_cols if c in df.columns]
    return df.astype({c: 'category' for c in cat_cols})


def concatenate_sessions(n_sessions, id_cols):
    """Concatenate single-session dataframes (STTSV -> TRIAL)."""
    path = os.path.join(gl.baseDir, gl.behavDir)

    # concatenate
    trials = []
    for sn in gl.participants:
        for session in range(1, n_sessions + 1):
            print(f'doing participant {sn}, session {session}')
            trials.append(pd.read_csv(os.path.join(path, f'day{session}', f'efc4_{sn}_single_trial.tsv'), sep='\t'))
    trial = pd.concat(trials)

    cat_cols = [c for c in id_cols if c in trial.columns]
    trial = trial.astype({c: 'category' for c in cat_cols})
    return trial[cat_cols + [c for c in trial.columns if c not in cat_cols]]


def group_trials_by(trial, by):
    """Average trials within each cell of `by`.

    Every measure is averaged over successful trials only, except `trialPoint`,
    which is averaged over all trials and is therefore the success rate.
    """
    success = trial[trial.trialPoint == 1].groupby(by, observed=True).mean(numeric_only=True).reset_index()
    rate = trial.groupby(by, observed=True)[['trialPoint']].mean().reset_index()
    return success.drop(columns='trialPoint').merge(rate, on=by)


def force_wide_to_long(force_wide, id_cols, force_cols):
    """Stack the per-finger force columns into one row per trial x finger (FWIDE -> FLONG).

    `force_cols` maps each measure to its per-finger columns. The measures are
    melted separately and pasted back together column-wise: each melt stacks the
    fingers in the same order, so the rows stay aligned. `finger` is named after
    the first measure melted (`thumb_abs`, ...).
    """
    force_long = None
    for measure, cols in force_cols.items():
        value_name = f'force_{measure}'
        melted = force_wide.melt(id_vars=id_cols + ['trialPoint'], value_vars=cols,
                                 var_name='finger', value_name=value_name)
        if force_long is None:
            force_long = melted
        else:
            force_long[value_name] = melted[value_name].to_numpy()
    return force_long
