import argparse
import inspect
import os
import numpy as np
import pandas as pd
#from sigproc.util import hp_filter
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.util as util
import EFC_learningfMRI.emg as emg
import EFC_learningfMRI.G_matrix as G_matrix

# Default layout of the EMG experiment. Templates, filled with {sn}, {day} and {bl} (1-based block).
PATH_EMG = os.path.join(gl.baseDir, gl.emgDir, 'emg', 'day{day}', 'subj{sn}')
PATH_DEL = os.path.join(PATH_EMG, f'{gl.emgDir}_{{sn}}_{{bl}}.csv')
PATH_DAT = os.path.join(gl.baseDir, gl.emgDir, 'behavioural', 'day{day}', f'{gl.emgDir}_{{sn}}.dat')

PATH_TSV = os.path.join(gl.baseDir, gl.emgDir, 'emg', f'{gl.emgDir}_pattern.tsv')  # all participants, one row per trial

PATH_PCM = os.path.join(gl.baseDir, gl.emgDir, 'pcm', 'subj{sn}')

# Trigger is at the start of the planning phase, go cue comes 1 s later, execution lasts max 3.5 s
GO_DELAY, EXEC_MAX = 3, .5


def extract_raw_and_timestamps(sns=emg.participants):
    """Read the Delsys csv of each block and save the raw EMG (chan, samples)."""
    thresh = 2
    edge   = 'rising'
    for sn in sns:
        for day in emg.days:
            savedir = PATH_EMG.format(sn=sn, day=day)
            os.makedirs(savedir, exist_ok=True)
            for bl in emg.get_blocks(sn, day)[0]:
                print(f'raw, subj{sn} - day{day} - block {bl}')
                df_out  = emg.load_delsys(PATH_DEL.format(sn=sn, day=day, bl=bl), trigger_name='Trigger', muscle_names=emg.channels)

                # extract timestamps
                trig_sig  = np.abs(df_out.Trigger.to_numpy())
                trig_time = df_out.time.astype(float).to_numpy()
                _, timestamp = emg.detect_trig(trig_sig, trig_time, amp_threshold=thresh, edge=edge)
                np.save(os.path.join(savedir, f'timestamp_{bl}.npy'), timestamp)

                # extract bipolar emg
                emg_raw = df_out[emg.channels].to_numpy().T
                np.save(os.path.join(savedir, f'emg_raw_{bl}.npy'), emg_raw)


def rectify(sns=emg.participants):
    """Rectify the EMG of each block."""
    for sn in sns:
        for day in emg.days:
            savedir = PATH_EMG.format(sn=sn, day=day)
            os.makedirs(savedir, exist_ok=True)
            for bl in emg.get_blocks(sn, day)[0]:
                print(f'rectify, subj{sn} - day{day} - block {bl}')
                emg_raw  = np.load(os.path.join(savedir, f'emg_raw_{bl}.npy'))
                emg_rect = np.abs(emg_raw - emg_raw.mean(axis=-1, keepdims=True))
                np.save(os.path.join(savedir, f'emg_rect_{bl}.npy'), emg_rect)


def segment_rect(sns=emg.participants):
    """Cut the rectified EMG into trials (1 s to 4.5 s after the trigger), all blocks stacked."""
    for sn in sns:
        for day in emg.days:
            savedir = PATH_EMG.format(sn=sn, day=day)
            blocks_emg, _ = emg.get_blocks(sn, day)
            emg_rect_seg = []
            for bl in blocks_emg:
                print(f'segment_rect, subj{sn} - day{day} - block {bl}')
                timestamp = np.load(os.path.join(savedir, f'timestamp_{bl}.npy'))
                assert len(timestamp) == 40, f'subj{sn} day{day} block {bl}: {len(timestamp)} triggers, expected 40'
                emg_rect  = np.load(os.path.join(savedir, f'emg_rect_{bl}.npy'))
                emg_rect_seg.append(emg.segment(emg_rect, timestamp, GO_DELAY, EXEC_MAX, gl.fsample['emg']))
            np.save(os.path.join(savedir, 'emg_rect_segmented.npy'), np.vstack(emg_rect_seg))


def rect_avg(sns=emg.participants):
    """Average the segmented rectified EMG over time, one pattern per trial."""
    for sn in sns:
        for day in emg.days:
            if not emg.get_blocks(sn, day)[0]:
                continue
            savedir  = PATH_SAVE.format(sn=sn, day=day)
            emg_rect = np.load(os.path.join(savedir, 'emg_rect_segmented.npy'))
            np.save(os.path.join(savedir, 'emg_rect_avg.npy'), emg_rect.mean(axis=-1))


def make_emg_pattern(sns=emg.participants):
    """Write the per-trial EMG patterns of all participants to one tsv, emg/<experiment>_pattern.tsv.

    Each pattern is the rectified EMG averaged from movement onset (RT) to target reached
    (RT + ET, at most EXEC_MAX), read from the segmented trials (go cue to go + EXEC_MAX).
    One row per trial: the descriptive columns followed by one column per channel.
    Running a subset of `sns` replaces only those participants' rows in an existing file.
    """
    descr = ['subNum', 'day', 'week', 'session', 'block_emg', 'BN', 'TN', 'chordID', 'chord', 'trialPoint', 'RT', 'ET']
    fs    = gl.fsample['emg']
    df    = []
    for sn in sns:
        trained = util.get_trained_and_untrained(sn, gl.emgDir)[:4]
        for day in emg.days:
            print(f'emg_pattern, subj{sn} - day{day}')
            savedir    = PATH_EMG.format(sn=sn, day=day)
            blocks_emg, blocks_dat = emg.get_blocks(sn, day)   # emg block -> dat BN, as in participants.tsv

            # trials of the day, in blocks_emg order, matched one-to-one to the segmented EMG
            dat = pd.read_csv(PATH_DAT.format(sn=sn, day=day), sep='\t')
            dat = pd.concat([dat[dat.BN == BN].assign(block_emg=bl) for bl, BN in zip(blocks_emg, blocks_dat)], ignore_index=True)
            seg = np.load(os.path.join(savedir, 'emg_rect_segmented.npy'))   # (trials, chan, samples)
            assert len(seg) == len(dat), f'subj{sn} day{day}: {len(seg)} segmented trials but {len(dat)} trials in dat'

            # window in samples from the go cue
            start = np.minimum((dat.RT / 1000 * fs).astype(int), seg.shape[-1] - 1)
            end   = np.maximum((np.minimum(dat.RT + dat.ET, EXEC_MAX * 1000) / 1000 * fs).astype(int), start + 1)
            dat[emg.channels] = np.array([seg[tr, :, s:e].mean(axis=-1) for tr, (s, e) in enumerate(zip(start, end))])

            dat['subNum'] = sn
            dat['day']    = day
            dat['chord']  = np.where(dat.chordID.astype(str).isin(trained), 'trained', 'untrained')
            df.append(dat[descr + emg.channels])

    df = pd.concat(df, ignore_index=True)
    df = df.sort_values(['subNum', 'day', 'BN', 'TN'], ignore_index=True)
    df.to_csv(PATH_TSV, sep='\t', index=False)


def calc_G_emg(sns=emg.participants, days=emg.days, prewhiten='univariate'):
    """The same Gs as calc_G_rois, but over the ten EMG channels instead of voxels.

    Channels take the place of the voxels and days the place of sessions: one pattern
    per block x chord (the 5 repetitions averaged), chords relabelled 0-7 trained first,
    so the Gs come out 8x8 within a day and 16x16 across (`days` entry 'all'). They are
    saved in EFC_learningEMG/pcm/subj<sn> as ``G_obs.within_session.<day>.emg.npy``, etc.

    `prewhiten` ('univariate', 'multivariate' or None) noise-normalises the successful
    trials of each participant x day before the repetitions are averaged, with the noise
    taken around each block x chord mean (see emg.prewhiten).
    """
    day_index = {day: d + 1 for d, day in enumerate(emg.days)}
    pattern_all = pd.read_csv(PATH_TSV, sep='\t')
    pattern_all = pattern_all[pattern_all.trialPoint == 1]
    for sn in sns:
        pattern = pattern_all[pattern_all.subNum == sn]

        if prewhiten is not None:
            pattern = pd.concat([emg.prewhiten(pattern[pattern.day == d], ['BN', 'chordID']) for d in pattern.day.unique()])
            
        slot_of = {chord: i for i, chord in enumerate(np.asarray(util.get_trained_and_untrained(sn, gl.emgDir), dtype=int))}
        for day in days:

            print(f'emg, doing participant {sn}, day {day}...')

            df = pattern if day == 'all' else pattern[pattern.day == day]
            df = df.groupby(['day', 'BN', 'chordID'], as_index=False)[emg.channels].mean()

            day_idx  = df.day.map(day_index)
            data     = df[emg.channels].to_numpy()
            cond_vec = util.make_cond_vec(day_idx, df.chordID.map(slot_of))
            part_vec = (day_idx * 100 + df.BN).to_numpy()   # BN restarts every day

            G        = G_matrix.calc_G(data, cond_vec, part_vec, centred=False)
            cov      = G_matrix.calc_G(data, cond_vec, part_vec, centred=True)
            G_raw    = G_matrix.calc_G(data, cond_vec, part_vec, centred=False, fixed_effect=False)
            G_noxval = G_matrix.calc_G(data, cond_vec, part_vec, centred=False, fixed_effect=False, crossval=False)

            savedir = PATH_PCM.format(sn=sn)
            os.makedirs(savedir, exist_ok=True)

            fname = G_matrix.make_fname(day, 'all')

            np.save(os.path.join(savedir, f'G_obs.{fname}.emg'), G)
            np.save(os.path.join(savedir, f'cov.{fname}.emg'), cov)
            np.save(os.path.join(savedir, f'G_obs_raw.{fname}.emg'), G_raw)
            np.save(os.path.join(savedir, f'G_obs_noxval.{fname}.emg'), G_noxval)


# Step name -> function, in pipeline order: timestamps and rectified EMG per block, then
# the per-trial patterns and their Gs. The raw EMG feeds the coherence analysis (coherence.py).
FUNC = {
    'raw'         : extract_raw_and_timestamps,
    'rectify'     : rectify,
    'segment_rect': segment_rect,
    'emg_pattern' : make_emg_pattern,
    'G_emg'       : calc_G_emg,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`sns=`, `thresh=`, `edge=`), but only the ones it accepts.
    """
    if what is not None:
        func = FUNC[what] # select function
        accepted = inspect.signature(func).parameters # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted}) # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the EMG of EFC_learningEMG and extract the per-trial muscle patterns.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: timestamps, rectify, emg_pattern)')
    parser.add_argument('--sns', nargs='+', type=int, default=None, help=f'participant ids (default: {emg.participants})')
    parser.add_argument('--days', nargs='+', type=lambda d: d if d == 'all' else int(d), default=None, help=f'days, G_emg only; "all" for the across-day G (default: {emg.days})')
    parser.add_argument('--thresh', type=float, default=None, help='trigger threshold, timestamps only (default: 2)')
    parser.add_argument('--prewhiten', default=None, choices=['univariate', 'multivariate', 'none'], help='noise normalisation, G_emg only (default: univariate)')
    parser.add_argument('--edge', default=None, choices=['rising', 'falling'], help='trigger edge, timestamps only (default: rising)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    if kwargs.get('prewhiten') == 'none':
        kwargs['prewhiten'] = None
    main(args.what, **kwargs)

    # if args.what is None:
    #     main('timestamps',  **kwargs)
    #     main('rectify',     **kwargs)
    #     main('emg_pattern', **kwargs)
