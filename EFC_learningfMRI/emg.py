import os
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.covariance import ledoit_wolf
import EFC_learningfMRI.globals as gl

participants  = [100, 102, 103, 104, 105]
days          = [1, 5]
channels      = ['flx_D1', 'flx_D2', 'flx_D3', 'flx_D4', 'flx_D5', 'ext_D1', 'ext_D2', 'ext_D3', 'ext_D4', 'ext_D5']
single_finger = [19999, 29999, 91999, 92999, 99199, 99299, 99919, 99929, 99991, 99992]


def get_blocks(sn, day):
    """EMG blocks of participant sn on day, and the .dat block each one was recorded in.

    Read from the `EMGBlocks_day<day>` (Delsys file number) and `BN_day<day>` (.dat BN)
    columns of participants.tsv, paired by position. The pairing is listed rather than
    derived because blocks dropped from the .dat and restarted recordings do not line up
    (subj100 day 5). Both lists are empty for a day without usable EMG ('nd').
    """
    pinfo = pd.read_csv(os.path.join(gl.baseDir, gl.emgDir, 'participants.tsv'), sep='\t')
    prow  = pinfo.loc[pinfo.sn == sn].iloc[0]
    blocks_emg, blocks_dat = str(prow[f'EMGBlocks_day{day}']), str(prow[f'BN_day{day}'])
    if blocks_emg == 'nd':
        return [], []
    blocks_emg, blocks_dat = [int(b) for b in blocks_emg.split('.')], [int(b) for b in blocks_dat.split('.')]
    assert len(blocks_emg) == len(blocks_dat), f'subj{sn} day{day}: EMGBlocks and BN in participants.tsv differ in length'
    return blocks_emg, blocks_dat


def detect_trig(trig_sig, time_trig, amp_threshold=None, edge='rising', min_duration=.05):
    """
    Detect trigger onsets with optional minimum duration filtering.

    :param trig_sig: signal array
    :param time_trig: time vector (same length as trig_sig)
    :param amp_threshold: threshold to binarize signal
    :param edge: 'rising' or 'falling'
    :param min_duration: minimum duration (in seconds) that the signal must remain above/below threshold
    :return: rise_times, rise_idx
    """

    if edge == 'rising':
        trig_bin = (trig_sig > amp_threshold).astype(int)
    elif edge == 'falling':
        trig_bin = (trig_sig < amp_threshold).astype(int)
    else:
        raise ValueError('edge must be either "rising" or "falling"')

    diff_trig = np.diff(trig_bin)
    rise_idx = np.where(diff_trig == 1)[0]
    fall_idx = np.where(diff_trig == -1)[0]

    # Make sure each rising edge has a corresponding falling edge
    if fall_idx.size == 0 or rise_idx.size == 0:
        return np.array([]), np.array([])

    # Ensure proper ordering
    if fall_idx[0] < rise_idx[0]:
        fall_idx = fall_idx[1:]
    if rise_idx[-1] > fall_idx[-1]:
        rise_idx = rise_idx[:-1]

    valid_rise_idx = []
    for r_idx, f_idx in zip(rise_idx, fall_idx):
        duration = time_trig[f_idx] - time_trig[r_idx]
        if duration >= min_duration:
            valid_rise_idx.append(r_idx)

    rise_idx = np.array(valid_rise_idx)
    rise_times = np.array([float(time_trig[idx]) for idx in rise_idx])

    # Remove triggers that are <4s apart
    if len(rise_idx) > 0:
        filtered_idx = [rise_idx[0]]
        last_time = rise_times[0]
        for i in range(1, len(rise_idx)):
            if rise_times[i] - last_time >= 4:
                filtered_idx.append(rise_idx[i])
                last_time = time_trig[rise_idx[i]]
        rise_idx = np.array(filtered_idx)
        rise_times = np.array([float(time_trig[idx]) for idx in rise_idx])

    return rise_times, rise_idx


def segment(data, start_idx, prestim, poststim, fsample):
    """
    Segment a (chan, samples) or (freq, chan, samples) array into fixed-length trials with no padding.

    Parameters
    ----------
    data : np.ndarray, shape (C, T) or (F, C, T)
    start_idx : array-like of int
        Trial start indices in samples (0-based).
    prestim : float
        Time in seconds before (if negative) or after (if positive) start_idx to begin the segment.
    poststim : float
        Time in seconds after start_idx to end the segment.
    fsample : float
        Sampling frequency (Hz).

    Returns
    -------
    segments : np.ndarray
        Shape (N, [F,] C, L), where L = (abs(prestim) + abs(poststim)) * fsample.
    """
    data = np.asarray(data)
    L = int((np.abs(prestim) + np.abs(poststim)) * fsample)

    segs = []
    for idx in start_idx:
        if prestim < 0:
            start = idx + int(prestim * fsample)
            end   = idx + int(poststim * fsample)
        else:  # prestim >= 0
            start = idx + int(prestim * fsample)
            end   = idx + int((prestim + poststim) * fsample)
        segs.append(data[..., start:end])

    segs = np.stack(segs, axis=0) if segs else np.zeros((0, *data.shape[:-1], L), dtype=data.dtype)

    return segs


def prewhiten(df, groupby=['BN', 'chordID'], multivariate=False, eps=1e-12):
    """Noise-normalise the EMG channels of a trial table (one row per trial).

    The noise is each trial's deviation from the mean of its cell (e.g. ['BN', 'chordID'],
    one cell per block x chord), i.e. the spread of the repetitions of the same chord in
    the same block. The chord effects are in the cell means, so they do not enter it.
    Removing only the block mean instead would leave the chord effects in the residuals
    and scale each channel by its signal + noise, not by its noise.

    Univariate: each channel is divided by its noise s.d. (ResMS, with dof = trials - cells).
    Multivariate: after that, the channels are multiplied by Sigma^-1/2 (Ledoit-Wolf shrinkage
    of the standardised noise covariance), which also removes the noise correlations between channels; the
    columns are then mixtures of channels, fine for G but no longer one muscle each.

    Run it on the trials of one participant and one day (the electrodes are re-applied every
    day), before averaging the repetitions.

    Returns a copy of `df` with the whitened values in the channel columns.
    """

    # trial-wise patterns
    X   = df[channels].to_numpy(dtype=float)

    # predicted trial-wise patterns, correspondin to the average pattern for each BN and chordID
    Y_hat = df.groupby(groupby)[channels].transform('mean').to_numpy(dtype=float)

    # residuals
    R   = X - Y_hat
    dof = len(df) - df.groupby(groupby).ngroups

    # univariate: every channel in units of its own noise s.d.
    sd = np.sqrt(np.clip((R ** 2).sum(axis=0) / dof, eps, None))
    X, R = X / sd, R / sd

    # multivariate: then remove the noise correlations. Shrinking the standardised residuals
    # pulls towards the identity, not towards the mean raw variance, which would inflate the
    # quiet channels when the raw variances differ by orders of magnitude.
    if multivariate:
        Sigma = ledoit_wolf(R, assume_centered=True)[0] * len(R) / dof
        w, U  = np.linalg.eigh(Sigma)
        X     = X @ (U / np.sqrt(np.clip(w, eps, None))) @ U.T
    out = df.copy()
    out[channels] = X
    return out


def load_delsys(filepath, trigger_name=None, muscle_names=None):
    """returns a pandas DataFrame with the raw EMG data recorded using the Delsys system

    :param filepath: Delsys .csv export
    :param trigger_name: name of the trigger column
    :param muscle_names: names of the EMG columns
    :return:
    """

    # read data from .csv file (Delsys output)
    with open(filepath, 'rt') as fid:
        A = []
        for line in fid:
            # Strip whitespace and newline characters, then split
            split_line = [elem.strip() for elem in line.strip().split(',')]
            A.append(split_line)

    # identify columns with data from each muscle
    muscle_columns = {}
    for muscle in muscle_names:
        for c, col in enumerate(A[3]):
            if muscle in col:
                muscle_columns[muscle] = c + 1  # EMG is on the right of Timeseries data (that's why + 1)
                break
        for c, col in enumerate(A[5]):
            if muscle in col:
                muscle_columns[muscle] = c + 1
                break

    df_raw = pd.DataFrame(A[8:])  # get rid of header
    df_out = pd.DataFrame()  # init final dataframe

    for muscle in muscle_columns:
        df_out[muscle] = pd.to_numeric(df_raw[muscle_columns[muscle]],
                                       errors='coerce').replace('', np.nan).dropna()  # add EMG to dataframe

    # add trigger column
    trigger_column = None
    for c, col in enumerate(A[3]):
        if trigger_name in col:
            trigger_column = c + 1

    try:
        trigger = df_raw[trigger_column]
        trigger = resample(trigger.values, len(df_out))
    except IOError as e:
        raise IOError("Trigger not found") from e

    df_out[trigger_name] = trigger

    # add time column
    df_out['time'] = df_raw.loc[:, 0]

    return df_out
