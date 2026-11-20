import argparse
import inspect
import os
import numpy as np
import pandas as pd
import xarray as xr
import PcmPy as pcm
from sigproc.coherence import calc_mscohere
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.util as util
import EFC_learningfMRI.emg as emg

# Default layout of the EMG experiment. Templates, filled with {sn}, {day} and {bl} (1-based block).
PATH_SAVE   = os.path.join(gl.baseDir, gl.emgDir, 'coherence', 'day{day}', 'subj{sn}')
PATH_DELSYS = os.path.join(gl.baseDir, gl.emgDir, 'emg', 'day{day}', 'subj{sn}', f'{gl.emgDir}_{{sn}}_{{bl}}.csv')
PATH_DAT    = os.path.join(gl.baseDir, gl.emgDir, 'behavioural', 'day{day}', f'{gl.emgDir}_{{sn}}.dat')

# Trials are cut from the go cue (1 s after the trigger) to the end of execution (3.5 s)
GO_DELAY, EXEC_MAX = 1, 3.5
NFREQ = 513 # frequency bins returned by calc_mscohere


def _load_z_coh(sn, day):
    """Fisher-z coherence of each chord trial, (trials, chan, chan, freq), and the matching dat rows.

    The trials are stacked in the order of the EMG blocks (segment_raw), so the dat rows
    are taken block by block from the BNs those blocks were paired with in participants.tsv.
    """
    data = np.load(os.path.join(PATH_SAVE.format(sn=sn, day=day), 'mscohere.npz'))
    dat  = pd.read_csv(PATH_DAT.format(sn=sn, day=day), sep='\t')
    dat  = pd.concat([dat[dat.BN == BN] for BN in emg.get_blocks(sn, day)[1]], ignore_index=True)
    z_coh = np.arctanh(np.sqrt(np.clip(data['coh'], 1e-12, 1 - 1e-12)))
    assert len(z_coh) == len(dat), f'subj{sn} day{day}: {len(z_coh)} coherence trials but {len(dat)} dat rows'
    return z_coh, dat, data['f']


def segment_raw(sns=emg.participants):
    """Cut the raw EMG into trials, all blocks stacked. Needs `raw` and `timestamps` from emg.py."""
    for sn in sns:
        for day in emg.days:
            savedir = PATH_SAVE.format(sn=sn, day=day)
            blocks_emg, _ = emg.get_blocks(sn, day)
            if not blocks_emg:
                continue
            emg_raw_seg = []
            for bl in blocks_emg:
                print(f'segment_raw, subj{sn} - day{day} - block {bl}')
                timestamp = np.load(os.path.join(savedir, f'timestamp_{bl}.npy'))
                emg_raw   = np.load(os.path.join(savedir, f'emg_raw_{bl}.npy'))
                emg_raw_seg.append(emg.segment(emg_raw, timestamp, GO_DELAY, EXEC_MAX, gl.fsample['emg']))
            np.save(os.path.join(savedir, 'emg_raw_segmented.npy'), np.vstack(emg_raw_seg))


def mscohere(sns=emg.participants):
    """Magnitude-squared coherence between all channel pairs in each trial."""
    for sn in sns:
        for day in emg.days:
            if not emg.get_blocks(sn, day)[0]:
                continue
            print(f'mscohere, subj{sn} - day{day}')
            savedir = PATH_SAVE.format(sn=sn, day=day)
            emg_raw = np.load(os.path.join(savedir, 'emg_raw_segmented.npy'))
            f, coh  = calc_mscohere(emg_raw, fsample=gl.fsample['emg'])
            np.savez(os.path.join(savedir, 'mscohere.npz'), coh=coh, f=f)


def mscohere_pattern(sns=emg.participants):
    """Crossvalidated distances and cosines between chords of the channel-pair coherence pattern, per frequency."""
    mask   = np.tri(len(emg.channels), k=-1, dtype=bool)
    Dist   = np.full((len(emg.days), len(sns), NFREQ, 8, 8), np.nan)   # nan where a day has no usable EMG
    Cosine = np.full_like(Dist, np.nan)
    for d, day in enumerate(emg.days):
        for s, sn in enumerate(sns):
            if not emg.get_blocks(sn, day)[0]:
                continue
            print(f'doing participant {sn}, day {day}...')
            chordID = np.array(util.get_trained_and_untrained(sn, gl.emgDir)).astype(int)
            z_coh, dat, _ = _load_z_coh(sn, day)
            dat.chordID = dat.chordID.map({ch: i for i, ch in enumerate(chordID)})
            z_coh_grouped, cond_vec, part_vec = pcm.group_by_condition(z_coh, dat.chordID, dat.BN, axis=0)
            obs_des = {'cond_vec': cond_vec, 'part_vec': part_vec}
            z_coh_grouped = z_coh_grouped[:, mask, :]
            for f in range(NFREQ):
                Y = pcm.Dataset(z_coh_grouped[:, :, f], obs_des)
                G, _ = pcm.est_G_crossval(Y.measurements, Y.descriptors['cond_vec'], Y.descriptors['part_vec'])
                Dist[d, s, f]   = pcm.G_to_dist(G)
                Cosine[d, s, f] = pcm.G_to_cosine(G)
    np.save(os.path.join(gl.baseDir, gl.emgDir, 'coherence', 'Dissimilarity.z_coh.npy'), Dist)
    np.save(os.path.join(gl.baseDir, gl.emgDir, 'coherence', 'Cosine.z_coh.npy'), Cosine)


def mscohere_avg(sns=emg.participants):
    """Fisher-z coherence averaged over channel pairs and trials, trained vs untrained chords."""
    mask      = np.tri(len(emg.channels), k=-1, dtype=bool)
    z_coh_avg = np.full((len(emg.days), NFREQ, len(sns), 2), np.nan)   # nan where a day has no usable EMG
    freq      = None
    for d, day in enumerate(emg.days):
        for s, sn in enumerate(sns):
            if not emg.get_blocks(sn, day)[0]:
                continue
            print(f'doing participant {sn}, day {day}...')
            chordID = np.array(util.get_trained_and_untrained(sn, gl.emgDir)).astype(int)
            z_coh, dat, freq = _load_z_coh(sn, day)
            trained = dat.chordID.isin(chordID[:4])
            z_coh = z_coh[:, mask, :].mean(axis=1)
            z_coh_avg[d, :, s, 0] = z_coh[trained].mean(axis=0)
            z_coh_avg[d, :, s, 1] = z_coh[~trained].mean(axis=0)
    da = xr.DataArray(
        z_coh_avg,
        dims=('day', 'freq', 'sn', 'chord'),
        coords={'day': emg.days, 'freq': freq, 'sn': sns, 'chord': ['trained', 'untrained']},
    )
    da.to_netcdf(os.path.join(gl.baseDir, gl.emgDir, 'coherence', 'mscohere.h5'), engine="h5netcdf")


# Step name -> function, in pipeline order. The raw EMG and the trial timestamps
# come from emg.py (`--what raw`, `--what timestamps`).
FUNC = {
    'segment_raw'     : segment_raw,
    'mscohere'        : mscohere,
    'mscohere_pattern': mscohere_pattern,
    'mscohere_avg'    : mscohere_avg,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`sns=`), but only the ones it accepts.
    """
    if what is not None:
        func = FUNC[what] # select function
        accepted = inspect.signature(func).parameters # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted}) # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intermuscular coherence of the EFC_learningEMG chords.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: segment_raw, mscohere)')
    parser.add_argument('--sns', nargs='+', type=int, default=None, help=f'participant ids (default: {emg.participants})')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        main('segment_raw', **kwargs)
        main('mscohere',    **kwargs)
