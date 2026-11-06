import argparse
import os

import numpy as np
import pandas as pd
import nibabel as nb
import nitools as nt
from nitools import spm
import inspect
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.util as util

# time axis of a segmented trial: 3 samples before onset to 16 after (20 samples).
_tAx = np.arange(-3, 17)


def bold_in_roi(SPM, path_glm, path_rois, atlas_name, H, roi):
    """Save raw, predicted and adjusted BOLD timeseries for a single ROI/hemisphere."""
    roi_img = nb.load(os.path.join(path_rois, f'{atlas_name}.{H}.{roi}.nii'))
    coords  = nt.get_mask_coords(roi_img)
    y_raw   = nt.sample_images(SPM.rawdata_files, coords)
    y_scl   = y_raw * SPM.gSF[:, None] # rescale y_raw

    _, info, _, y_hat, y_adj, _ = SPM.rerun_glm(y_scl)

    return y_scl, y_hat, y_adj

    

def save_bold_rois(sns=gl.participants, glm=3, atlas_name='ROI', rois=None):
    """Save the per-ROI BOLD timeseries (hat / raw / adj) for each participant."""

    if rois is None:
        rois = gl.rois[atlas_name]

    for sn in sns:
        path_glm  = os.path.join(gl.baseDir, f'{gl.glmDir}{glm}', f'subj{sn}')
        path_rois = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
        SPM       = spm.SpmGlm(path_glm)

        SPM.get_info_from_spm_mat()

        for H in gl.Hem:
            for roi in rois:
                print(f'doing participant {sn}, {H}, {roi}')
                y_scl, y_hat, y_adj = bold_in_roi(SPM, path_glm, path_rois, atlas_name, H, roi)

                np.save(os.path.join(path_glm, f'BOLD.hat.{H}.{roi}.npy'), y_hat)
                np.save(os.path.join(path_glm, f'BOLD.raw.{H}.{roi}.npy'), y_scl)
                np.save(os.path.join(path_glm, f'BOLD.adj.{H}.{roi}.npy'), y_adj)


def segment_bold(sns=gl.participants, glm=3, atlas_name='ROI'):
    """Cut the adjusted BOLD around each trial onset into one tidy timecourse table.

    Reloads the `BOLD.adj.<H>.<roi>.npy` timeseries written by `save_bold`, cuts them
    around the glm onsets, averages over voxels and stacks every participant, hemisphere
    and ROI into `bold/bold_segmented.tsv`.
    """
    df = pd.DataFrame()
    for H in gl.Hem:
        for roi in gl.rois[atlas_name]:
            for sn in sns:
                print(f'doing participant {sn}, {H}, {roi}, glm {glm}')

                bold = np.load(os.path.join(gl.baseDir, f'{gl.glmDir}{glm}', f'subj{sn}', f'BOLD.raw.{H}.{roi}.npy'))

                # retrieve onsets
                onset, events = util.load_glm_onset(sn, glm, output_events=True)

                bold_cut = spm.cut(bold, pre=3, at=onset, post=16)  # (n_trials, 20, n_voxels)

                # average over voxels -> one timecourse (20 samples) per trial
                tc = bold_cut.mean(axis=2)

                trained_untrained = util.get_trained_and_untrained(sn)

                # tag each trial's timecourse with its chord and session
                trials = pd.DataFrame(tc, columns=_tAx)
                trials['chordID'] = np.asarray(events.chordID)
                trials['session'] = np.asarray(events.day)
                trials['chord'] = 'untrained'
                trained_ids = [int(c) for c in trained_untrained[:4]]
                trials.loc[trials.chordID.isin(trained_ids), 'chord'] = 'trained'

                # reshape to long/tidy form and add the descriptor columns
                sub = trials.melt(id_vars=['chordID', 'session', 'chord'], var_name='time', value_name='signal')
                sub['sn'] = sn
                sub['roi'] = roi
                sub['Hem'] = H

                df = pd.concat([df, sub], ignore_index=True)

    df.to_csv(os.path.join(gl.baseDir, f'{gl.glmDir}{glm}', 'bold_segmented.tsv'), sep='\t', index=False)
    return df


# Step name -> function. `dataframe_bold` reloads the BOLD.adj files `bold_rois` writes.
FUNC = {
    'bold_rois'     : save_bold_rois,
    'dataframe_bold': segment_bold,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`sns=`, `glm=`, `atlas_name=`, `rois=`), but only
    the ones it accepts -- `dataframe_bold` takes no `rois`.
    """
    if what is not None:
        func     = FUNC[what]                                       # select function
        accepted = inspect.signature(func).parameters               # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted})  # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save per-ROI BOLD timeseries and segment them into a tidy table.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--glm', type=int, default=None, help='GLM the BOLD/onsets come from (default: the step default, 3)')
    parser.add_argument('--sns', nargs='+', type=int, default=None, help='participant numbers (default: all participants)')
    parser.add_argument('--rois', nargs='+', type=str, default=None, help='which rois of the atlas to use, bold_rois only (default: all)')
    parser.add_argument('--atlas_name', default=None, help='atlas whose ROIs to use (default: the step default, ROI)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        pass
        # main('bold_rois',      **kwargs)
        # main('dataframe_bold', **kwargs)
