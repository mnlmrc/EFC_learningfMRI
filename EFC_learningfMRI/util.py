import itertools
import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt, firwin
from scipy.special import expit
import EFC_learningfMRI.globals as gl
from scipy.stats import linregress, t
#import EFC_learningfMRI.globals as gl


# --- the cond_vec label format -------------------------------------------------
# A cond_vec entry is a ','-joined string of integer fields:
#
#     'session,chord'             e.g. glm3, whose regressors carry no repetition
#     'session,repetition,chord'  e.g. glm2, whose regressors do
#
# `session` is the 1-based index of the session in gl.sessions (3 -> 1, 9 -> 2,
# 23 -> 3), NOT the session number itself; `chord` is the 0-based slot in the
# participant's trained-first chord order (see get_trained_and_untrained), so
# slots 0-3 are trained and 4-7 untrained for every participant.
#
# make_cond_vec and parse_cond_vec below are the only code that knows this — build
# and read labels through them rather than formatting or splitting strings by hand,
# so a two-field label can never be read as if it were a three-field one.
COND_SEP  = ','
N_TRAINED = 4     # slots 0..N_TRAINED-1 are the trained chords


def session_index(session):
    """1-based index a real IMAGING session number (3, 9, 23) has in a cond_vec label."""
    index = {s: i + 1 for i, s in enumerate(gl.sessions)}
    if session not in index:
        raise ValueError(f"session must be one of {sorted(index)}, got {session!r}")
    return index[session]


def make_cond_vec(session, chord, repetition=None):
    """Assemble cond_vec labels from their fields.

    Args:
        session:    1-based session index per row (see :func:`session_index`).
        chord:      0-based trained-first chord slot per row.
        repetition: repetition label per row, or None for a glm that has none.

    Returns:
        Array of ``len(session)`` label strings e.g. session 3, repetition 1, trained chord, 3,1,<int from 0-3, 4-7 for untrianed>.
    """
    fields = [session, chord] if repetition is None else [session, repetition, chord]
    cols   = [pd.Series(np.asarray(f)).reset_index(drop=True).astype(str) for f in fields]
    return cols[0].str.cat(cols[1:], sep=COND_SEP).to_numpy()


def parse_cond_vec(cond_vec):
    """Split cond_vec labels back into their integer fields.

    Returns:
        DataFrame with one row per label and the integer columns ``session`` and
        ``chord``, plus ``repetition`` when (and only when) the labels carry that
        field. Callers that need the repetition must therefore check for the
        column rather than assume a position — which is the point of parsing here
        instead of splitting on ``,`` at the call site.
    """
    if np.ndim(cond_vec) == 0:
        raise TypeError('cond_vec must be the array of condition labels, not a row count: '
                        'the session is read off the labels so participants with unequal '
                        f'runs per session are split correctly (got {cond_vec!r})')

    parts = pd.Series(np.asarray(cond_vec)).str.split(COND_SEP, expand=True)

    if parts.shape[1] == 2:
        parts.columns = ['session', 'chord']
    elif parts.shape[1] == 3:
        parts.columns = ['session', 'repetition', 'chord']
    else:
        raise ValueError(f"a cond_vec label has 2 or 3 {COND_SEP!r}-separated fields "
                         f"('session,chord' or 'session,repetition,chord'), got "
                         f"{np.asarray(cond_vec).ravel()[0]!r}")

    return parts.astype(int)


def _is_all(value):
    """True for the ``'all'`` sentinel, without comparing an array to a string."""
    return isinstance(value, str) and value == 'all'


@dataclass
class RegInfo:
    """The reginfo.tsv of one subject/glm, parsed into partition and condition vectors.

    The file is read and parsed on first access, then cached, so a pipeline that
    needs ``cond_vec`` and ``part_vec`` more than once does not re-read the TSV
    each time (which is what repeated ``get_cond_part`` calls do)::

        reg = RegInfo(sn, glm)
        reg.cond_vec, reg.part_vec        # parsed once here, reused after
        reg.reginfo                       # the raw DataFrame, if you need it
    """

    sn: int
    glm: int

    @cached_property
    def reginfo(self):
        """The raw reginfo.tsv as a DataFrame."""
        path = os.path.join(gl.baseDir, f'glm{self.glm}', f'subj{self.sn}', 'reginfo.tsv')
        return pd.read_csv(path, sep='\t')

    @cached_property
    def part_vec(self):
        """Partition (run) vector, one entry per regressor."""
        return self.reginfo.run.to_numpy()

    @cached_property
    def condition(self):
        return parse_regressor_name(self.reginfo.name)

    @cached_property
    def condition_unique(self):
        return self.condition.drop_duplicates(ignore_index=True)

    @cached_property
    def make_chord_mapping(self):
        trained_untrained = get_trained_and_untrained(self.sn)
        trained_untrained = np.asarray(trained_untrained, dtype=int)
        return dict(zip(trained_untrained, np.arange(8)))

    @cached_property
    def cond_vec(self):
        """
        transform condition labels into number for correct ordering in G matrix

        The labels themselves are assembled by :func:`make_cond_vec`, which is what
        :func:`parse_cond_vec` reads back.
        """
        session    = self.condition[1].map(gl.sess_mapping)
        chordID    = self.condition[0].astype(int).map(self.make_chord_mapping)
        repetition = self.condition[2] if self.condition.shape[1] > 2 else None

        return make_cond_vec(session, chordID, repetition)


def parse_regressor_name(name, sep=','):
    """Split a contrast regressor name into its component condition labels.

    Regressor names are ``sep``-delimited condition labels, e.g.
    ``'chordID,sess'`` or ``'chordID,sess,rep'`` with the default ``sep=','``.

    Parameters
    ----------
    name : pandas.Series of str
        The regressor names to parse.
    sep : str, optional
        Field separator (default ``','``).

    Returns
    -------
    tuple of pandas.Series
        One ``Series`` per delimited component, in order. The tuple has as many
        elements as the number of parts found, so it can be unpacked directly,
        e.g. ``chordID, sess = _parse_regressor_name(regressor)``.
    """
    parts = name.str.split(sep, expand=True)
    return pd.DataFrame(tuple(parts[col] for col in parts.columns)).T


def get_trained_and_untrained(sn):
    """
    Retrieve which chords where trained and untrained in participant sn.
    Args:
        sn: participant number

    Returns:
        list of chordIDs. First four are trained, last four untrained
    """

    pinfo     = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    trained   = pinfo[pinfo.sn == sn].reset_index()['trained'][0].split('.')
    untrained = pinfo[pinfo.sn == sn].reset_index()['untrained'][0].split('.')
    chords    = list()
    chords.extend(trained)
    chords.extend(untrained)

    return chords


def runs_to_keep(cond_vec, session='all', repetition='all', chord='all'):
    """Boolean mask over regressor rows, selecting a session, repetition and/or chord set.

    The only reader of the cond_vec format is :func:`parse_cond_vec`, so a filter
    here cannot disagree with how the labels were built.

    Args:
        cond_vec:   condition label per regressor row, e.g. ``RegInfo(sn, glm).cond_vec``.
        session:    3, 9 or 23 to keep only that session's rows, a sequence of them to
                    keep a group of sessions (a session pair, say), or ``'all'``.
        repetition: repetition label to keep, or ``'all'``. Only works for a glm whose labels carry a repetition field ('session,repetition,chord', e.g. glm2).
        chord:      ``'trained'`` (slots 0-3) or ``'untrained'`` (slots 4-7) to keep one
                    chord set, or ``'all'``.

    Returns:
        Boolean array of length ``len(cond_vec)``, True for rows passing every filter.
    """
    cond = parse_cond_vec(cond_vec)
    keep = np.ones(len(cond), dtype=bool)

    if not _is_all(session):
        wanted = [session_index(s) for s in np.atleast_1d(session).tolist()]
        keep  &= cond.session.isin(wanted).to_numpy()
        if not keep.any():
            raise ValueError(f'no rows match session {session!r}; cond_vec holds sessions '
                             f'{sorted(cond.session.unique().tolist())} (1-based)')

    if not _is_all(repetition):
        if 'repetition' not in cond:
            raise ValueError("cond_vec has no repetition field (expected 'session,repetition,chord')")
        keep &= (cond.repetition == repetition).to_numpy()
        if not keep.any():
            raise ValueError(f'no rows match repetition {repetition!r}; '
                             f'cond_vec holds {sorted(cond.repetition.unique().tolist())}')

    if not _is_all(chord):
        if chord == 'trained':
            keep &= (cond.chord < N_TRAINED).to_numpy()
        elif chord == 'untrained':
            keep &= (cond.chord >= N_TRAINED).to_numpy()
        else:
            raise ValueError(f"chord must be 'all', 'trained' or 'untrained', got {chord!r}")
        if not keep.any():
            raise ValueError(f'no rows match chord {chord!r} in session {session!r}; '
                             f'cond_vec holds slots {sorted(cond.chord.unique().tolist())}')

    return keep


def add_chord_column(df, chordID_col='chordID', sn_col='sn', out_col='chord'):
    """Add a ``chord`` column labelling each row 'trained' or 'untrained'.

    For every participant in ``sn_col`` the trained/untrained chord IDs are
    looked up with :func:`get_trained_and_untrained` (first four trained, last
    four untrained), and each row's ``chordID_col`` value is labelled
    accordingly.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing a chord-ID column and a participant column.
    chordID_col : str, optional
        Name of the chord-ID column (default ``'chordID'``).
    sn_col : str, optional
        Name of the participant column (default ``'sn'``); needed because the
        trained/untrained split differs per participant.
    out_col : str, optional
        Name of the column to add (default ``'chord'``).

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the ``out_col`` column added.
    """
    df = df.copy()
    df[out_col] = 'untrained'
    for sn in df[sn_col].unique():
        trained = np.array(get_trained_and_untrained(sn)[:4]).astype(int)
        mask = (df[sn_col] == sn) & (df[chordID_col].astype(int).isin(trained))
        df.loc[mask, out_col] = 'trained'

    return df


def add_flexion_imbalance(pair, sep="-"):
    """
    Flexion difference for a chord pair, e.g. "91211-22911".
    Each chord is 5 fingers coded 1=extension, 2=flexion, 9=neutral.
    Returns |n_flexed(A) - n_flexed(B)|; NaN if the entry is malformed.

    Accepts a single string or a whole column: pass a pandas Series (e.g.
    ``df['pair']``) and a Series of the same length is returned, so it can be
    assigned straight back with ``df['flexion_imbalance'] = add_flexion_imbalance(df['pair'])``.
    """
    if isinstance(pair, pd.Series):
        return pair.map(lambda p: add_flexion_imbalance(p, sep))
    if not isinstance(pair, str):
        return float("nan")
    parts = pair.split(sep)
    if len(parts) != 2 or any("99999" in p for p in parts):
        return float("nan")
    a, b = parts
    return abs(a.count("2") - b.count("2"))   # 2 == flexion


def add_behav_pair_diff(df, measures=('ET', 'MD', 'force_abs', 'force_der', 'force_der_peak'), behav=None, pair_col='pair',
                        sn_col='sn', session_col='session', behav_sn_col='subNum', chordID_col='chordID'):

    """Behavioural difference between the two chords of each pair, as a column per measure.

    For every row the two chords in ``pair_col`` are looked up in the trial-level
    behavioural table for that participant and session, averaged over trials, and
    ``|measure(A) - measure(B)|`` is written to ``f'{measure}{suffix}'``. Lets a
    pair's crossnobis distance be regressed on how differently its two chords were
    actually executed.

    The difference is absolute because ``pair`` is unordered -- the two chord IDs
    are sorted when the dissimilarity table is built -- so a signed difference
    would flip meaning from row to row. Same convention as
    :func:`add_flexion_imbalance`.

    Parameters
    ----------
    df : pandas.DataFrame
        Dissimilarity table, one row per chord pair, e.g.
        ``dissimilarity.within_session.{atlas}.glm{glm}.tsv``.
    measures : sequence of str, optional
        Behavioural columns to difference.
    behav : pandas.DataFrame, optional
        Trial-level behaviour. Defaults to ``force.success.tsv``, which is the
        table to use here: the ``*.session.success.tsv`` files are already
        collapsed to trained/untrained, and their ``chordID`` is an average of
        chord IDs rather than a chord.
    pair_col, sn_col, session_col : str, optional
        Columns of ``df`` holding the pair, the participant and the session.
    behav_sn_col, chordID_col : str, optional
        Participant and chord-ID columns of ``behav`` (named differently there).
    sep : str, optional
        Separator between the two chord IDs in ``pair_col``.
    suffix : str, optional
        Appended to each measure name to form the new column name.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with one ``f'{measure}{suffix}'`` column per measure.
        Pairs whose chords are missing from ``behav``, or that are malformed
        (e.g. containing '99999'), come out as NaN.
    """

    measures = list(measures)
    absent   = [m for m in measures if m not in behav.columns]
    if absent:
        raise KeyError(f'{absent} not in the behavioural table; it has {sorted(behav.columns)}')

    # one value per (participant, session, chord), averaged over successful trials
    behav = behav.copy()
    behav[chordID_col] = pd.to_numeric(behav[chordID_col], errors='coerce').astype(float)
    lut = behav.groupby([behav_sn_col, session_col, chordID_col])[measures].mean()

    chords = df[pair_col].str.split('-', expand=True)
    sides  = [pd.to_numeric(chords[i], errors='coerce').astype(float) for i in (0, 1)]

    df = df.copy()
    # reindex rather than merge, so row order and duplicate pairs are preserved
    vals = [lut.reindex(pd.MultiIndex.from_arrays([df[sn_col], df[session_col], c])) for c in sides]
    for m in measures:
        df[m] = np.abs(vals[0][m].to_numpy() - vals[1][m].to_numpy())

    return df


def split_trained_untrained(M):
    """Mean of the trained (first 4) and untrained (last 4) condition blocks."""
    mask4 = np.tri(4, k=-1, dtype=bool)
    mask8 = np.tri(8, k=-1, dtype=bool)
    if M.ndim==2:
        trained   = M[:4, :4][mask4]
        untrained = M[4:, 4:][mask4]
        tot       = M[mask8].mean()
    elif M.ndim==3:
        trained   = M[:, :4, :4][:, mask4]
        untrained = M[:, 4:, 4:][:, mask4]
        tot       = M[:, mask8].mean(axis=1)
    return tot, trained, untrained


def diff_trained_untrained(df, stat, on):
    """Trained minus untrained `stat` in column Diff, matching rows on the `on` columns."""
    diff = df[df.chord=='trained'].merge(df[df.chord=='untrained'], on=on)
    diff['Diff'] = diff[f'{stat}_x'] - diff[f'{stat}_y']
    diff.session = diff.session.astype(int)
    return diff



def load_nat_emg(file_path):
    # Load the .mat file
    mat = scipy.io.loadmat(file_path)

    # Extract the 'dist' cell array from 'emg_natural_dist'
    emg_nat = mat['emg_natural_dist']
    emg_nat = emg_nat['dist'][0, 0]

    emg_nat_list = []
    for e in emg_nat:
        emg_nat_list.append(e[0])

    return emg_nat_list


def lowpass_butter(signal=None, cutoff=None, fsample=None, order=5, axis=-1):
    """
    Apply a low-pass filter to a 5-by-t signal array.

    Parameters:
    signal (np.ndarray): 5-by-t array where each row is a signal to be filtered.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling frequency of the signal.
    order (int): The order of the Butterworth filter (default is 5).

    Returns:
    np.ndarray: The filtered 5-by-t signal array.
    """
    # Design a Butterworth low-pass filter
    nyquist = .5 * fsample
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    filtered_signal = filtfilt(b, a, signal, axis=axis)

    return filtered_signal



def lowpass_fir(data, n_ord=None, cutoff=None, fsample=None, padlen=None, axis=-1):
    """
    Low-pass filter to remove high-frequency noise from the EMG signal.

    :param data: Input signal to be filtered.
    :param n_ord: Filter order.
    :param cutoff: Cutoff frequency of the low-pass filter.
    :param fsample: Sampling frequency of the input signal.
    :return: Filtered signal.
    """
    numtaps = int(n_ord * fsample / cutoff)
    b = firwin(numtaps + 1, cutoff, fs=fsample, pass_zero='lowpass')
    filtered_data = filtfilt(b, 1, data, axis=axis, padlen=padlen)

    return filtered_data


def calc_R2(Y, Yhat):
    ss_res = np.nansum((Y - Yhat) ** 2)
    ss_tot = np.nansum((Y - np.nanmean(Y)) ** 2)

    return 1 - ss_res / ss_tot



def ttest_im_sess(df, rois=None, x='session', y=None, hue=None, hue_order=None, roi_col='roi', subject_col='sn', paired=True, alternative='two-sided', popmean=0):
    """Paired t-tests between ``hue`` levels, one per level of ``x`` (and per ROI).

    The statistical counterpart of the ``plot_*_sess`` functions in
    :mod:`EFC_learningfMRI.vis`: it takes the same long-format table and the same
    ``rois``/``x``/``y``/``hue`` arguments, and tests, within each session, the
    ``y`` values of one ``hue`` level against another. With more than two levels
    every pair is tested (columns ``A`` and ``B`` say which), so the
    trained/untrained case is simply the two-level special case::

        ttest = ttest_im_sess(dist_chord_sess, rois, y='crossnobis', hue='chord',
                              hue_order=['trained', 'untrained'])
        print(ttest[['T', 'dof', 'p_val', 'CI95', 'roi', 'session']])

    Leave ``hue`` out to test the ``y`` values themselves against ``popmean``
    (a one-sample t-test per ROI / ``x`` level), which is how a quantity with no
    contrast to take -- a noise ceiling, a correlation -- is tested against zero::

        ttest = ttest_im_sess(nc[nc.Hem == 'L'], rois, y='lower', alternative='greater')
        print(ttest[['roi', 'session', 'T', 'dof', 'p_val', 'CI95']])

    Set ``roi_col=None`` for a table with no ROI column -- the behavioural ones,
    say, where the test is over sessions only::

        ttest = ttest_im_sess(perf[perf.session.isin(gl.sessions)], y='ET', hue='chord',
                              hue_order=['trained', 'untrained'], roi_col=None, subject_col='subNum')
        print(ttest[['session', 'A', 'B', 'T', 'dof', 'p_val', 'CI95', 'cohen_d']])

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format table with one row per subject / ROI / ``x`` / ``hue`` cell.
        Only the levels of ``x`` present in ``df`` are tested, so restrict the
        sessions by filtering ``df``. Duplicate rows within a cell raise, rather
        than being averaged silently.
    rois : list, optional
        ROIs to test, in order (default: all in ``roi_col``).
    x : str, optional
        Column whose levels are tested separately (default ``'session'``).
    y : str
        Column holding the values to test.
    hue : str, optional
        Column whose levels are compared against each other; ``None`` for a
        one-sample test of ``y`` against ``popmean``.
    hue_order : list, optional
        Levels of ``hue`` to compare, in order; the t-test is signed as the
        first level minus the second (default: order of appearance in ``df``).
    roi_col : str, optional
        Column holding the ROI names (default ``'roi'``); ``None`` to test the
        whole table at once, without splitting it by ROI.
    subject_col : str, optional
        Column identifying the subject, used to pair observations across
        ``hue`` levels (default ``'sn'``; the behavioural tables use ``'subNum'``).
    paired : bool, optional
        Whether to pair the two samples by ``subject_col`` (default True);
        ignored by the one-sample test.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Passed to pingouin.
    popmean : float, optional
        Value the one-sample test compares against (default 0); ignored when
        ``hue`` is given.

    Returns
    -------
    pandas.DataFrame
        One row per ROI / ``x`` level / pair of ``hue`` levels, with columns
        ``A``, ``B``, ``T``, ``dof``, ``p_val``, ``CI95``, ``cohen_d``, ``BF10``,
        ``power``, plus the ``x`` column and, unless ``roi_col`` is None, ``roi_col``.
        The one-sample rows carry ``A = y`` and ``B = popmean``.
    """
    import pingouin as pg   # heavy import; only this helper needs it

    if y is None:
        raise ValueError("y (the values to test) is required")

    one_sample = hue is None    # nothing to contrast against, so test y against popmean

    if hue_order is None and not one_sample:
        hue_order = list(df[hue].unique())

    if roi_col is None:
        rois = [None]           # one pass over the whole table
    elif rois is None:
        rois = df[roi_col].unique()

    x_levels = np.sort(df[x].unique())

    rows = []
    for roi in rois:

        df_roi = df if roi_col is None else df[df[roi_col]==roi]

        for level in x_levels:

            df_level = df_roi[df_roi[x]==level]

            # (A, B, sample_A, sample_B) per test; pingouin takes a scalar as the
            # second argument, which is what makes the one-sample case the same call
            if one_sample:
                vals = df_level.set_index(subject_col)[y]
                if vals.index.has_duplicates:
                    raise ValueError(f'more than one row per {subject_col} in {x}={level}'
                                     + ('' if roi_col is None else f', {roi_col}={roi}'))
                tests = [(y, popmean, vals.dropna(), popmean)]
            else:
                piv        = df_level.pivot(index=subject_col, columns=hue, values=y)
                hue_levels = [h for h in hue_order if h in piv.columns]
                tests      = []
                for a, b in itertools.combinations(hue_levels, 2):
                    pair = piv[[a, b]].dropna()
                    tests.append((a, b, pair[a], pair[b]))

            for a, b, sample_a, sample_b in tests:
                res = pg.ttest(sample_a, sample_b, paired=paired and not one_sample, alternative=alternative).iloc[0]
                row = {x        : level,
                       'A'      : a,
                       'B'      : b,
                       'T'      : res['T'],
                       'dof'    : res['dof'],
                       'p_val'  : res['p_val'],
                       'CI95'   : res['CI95'],       # of the A - B difference (of the mean, one-sample); pingouin rounds it to 2 decimals
                       'cohen_d': res['cohen_d'],    # unsigned -- the sign of the effect is in T
                       'BF10'   : res.get('BF10', np.nan),   # pingouin only reports it for the two-sided test
                       'power'  : res.get('power', np.nan)}
                rows.append(row if roi_col is None else {roi_col: roi, **row})

    return pd.DataFrame(rows)


def load_glm_onset(sn, glm, output_events=False):
    """Trial onsets of participant `sn`, in TRs of the glm's concatenated timeseries.

    The events file numbers the blocks with a fixed offset of `gl.nRuns` per session -- run
    r of session i is BN = r + i * gl.nRuns -- while the runs that actually made it into the
    glm are listed session by session in the `FuncRuns_day<day>` columns of participants.tsv,
    and those differ between days (subj116, for one, kept 7 runs on day 9 and 10 on the other
    two). The glm stacks the runs it kept back to back, so a trial's onset is set by the
    position of its run among the kept ones, not by its BN.
    """
    pinfo  = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    prow   = pinfo.loc[pinfo.participant_id == f"subj{sn}"].iloc[0]

    events = pd.read_csv(os.path.join(gl.baseDir, gl.behavDir, 'day3', f'efc4_subj{sn}_glm{glm}_events.tsv'), sep='\t')
    events = events[events.chordID != 99999]

    # (day, BN) -> position of that run in the concatenated timeseries. Only the sessions
    # this glm covers count towards the position (glm 4, for one, is day 3 only).
    runs = []
    for i, day in enumerate(gl.sessions):
        func_runs = str(prow[f"FuncRuns_day{day}"])
        for run in np.sort(np.array(func_runs.split('.'), dtype=int)):
            runs.append({'day': day, 'BN': run + i * gl.nRuns, 'run_idx': len(runs)})

    events = events.merge(pd.DataFrame(runs), on=['day', 'BN'])   # inner join drops the runs left out of the glm
    onset  = (np.round(events.Onset.to_numpy() * gl.TR) + events.run_idx.to_numpy() * gl.nTR).astype(int)

    order  = np.argsort(onset)                                    # keep events row i paired with onset i
    events = events.drop(columns='run_idx').iloc[order].reset_index(drop=True)
    onset  = onset[order]

    if output_events:
        return onset, events
    else:
        return onset

