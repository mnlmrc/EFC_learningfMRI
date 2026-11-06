"""Hypothetical chord patterns built from the single-finger patterns of Arbuckle et al. (2020).

Arbuckle, Pruszynski & Diedrichsen (2020) J Neurosci 40(48):9210, doi:10.1523/JNEUROSCI.0999-20.2020.
Participants flexed or extended single fingers of the right hand at three force levels
(flexion 1.5, 2, 2.5 N; extension 1, 1.5, 2 N) in 8 runs of 30 conditions with 2 repeats.
glm3 has one regressor per condition and run (240 task regressors + 8 run intercepts).

The model pattern of a chord is the sum of the patterns of the fingers it presses, each in
the direction it presses them, e.g. 21911 = thumb flexion + index extension + ring extension
+ pinkie extension. It is built run by run, so the 8 runs can be cross-validated. The patterns
are saved unwhitened, like the real betas; univariate prewhitening (dividing by sqrt(ResMS))
commutes with summing betas, so it can be applied downstream.

Direction codes differ between the two datasets:
    Arbuckle `dirPressed`: 1 flexion, 2 extension
    chordID digits:        1 extension, 2 flexion, 9 no press
"""
import os
import numpy as np
import pandas as pd
import nibabel as nb
import nitools as nt
import scipy.io as sio
import imaging_pipelines.betas as bt

import EFC_learningfMRI.globals as gl

arbuckleDir = os.path.join(gl.baseDir, 'EF_Arbuckle_JNS2020')
participants = [f's{i:02d}' for i in range(1, 10)]

# chordID digit -> Arbuckle dirPressed
chord_to_arbuckle_dir = {1: 2, 2: 1}


def load_reginfo(sn, glm=3):
    """Regressor info of the task betas of `sn`, one row per beta_XXXX.nii (in that order)."""
    info = sio.loadmat(os.path.join(arbuckleDir, f'glm{glm}', sn, 'SPM_info.mat'), squeeze_me=True)
    cols = ['run', 'digitPressed', 'dirPressed', 'forcelvl', 'targetforce', 'regname']
    return pd.DataFrame({c: info[c] for c in cols})


def chord_weights(reginfo, chords=gl.chordID, targetforce=None):
    """Weights mapping the betas onto chord patterns, as (n_runs * n_chords, n_regressors), plus row names '<chordID>.<run>'.

    Within each run, a chord's row averages the regressors of each finger it presses, in the
    direction it presses it, and sums over fingers. With `targetforce=None` the force levels are
    averaged, otherwise only the regressors with that target force (in N) are used. Flexion and
    extension share only 1.5 and 2 N, so those are the two values that keep the directions matched.
    """
    sel  = np.ones(len(reginfo), dtype=bool) if targetforce is None else np.isclose(reginfo.targetforce, targetforce)
    rows = [(chord, run) for run in np.sort(reginfo.run.unique()) for chord in chords]
    W    = np.zeros((len(rows), len(reginfo)))
    for i, (chord, run) in enumerate(rows):
        for d, code in enumerate(str(chord)):
            if code == '9':
                continue
            idx = sel & (reginfo.run == run) & (reginfo.digitPressed == d + 1) & (reginfo.dirPressed == chord_to_arbuckle_dir[int(code)])
            if not idx.any():
                raise ValueError(f'no regressor for run {run}, digit {d + 1}, chord code {code}, targetforce {targetforce}')
            W[i, idx.to_numpy()] = 1 / idx.sum()
    return W, [f'{chord}.{run}' for chord, run in rows]


def make_chord_patterns(sn, chords=gl.chordID, targetforce=None, glm=3):
    """Cortical cifti of the hypothetical chord patterns of Arbuckle participant `sn`, one row per chord and run.

    The betas are sampled at the hemisphere masks arbuckleDir/RegionOfInterest/<sn>/Hem.{L,R}.nii, which must
    already exist. Cortical voxels outside the glm mask are NaN, as in the original betas.
    """
    path    = os.path.join(arbuckleDir, f'glm{glm}', sn)
    masks   = [os.path.join(arbuckleDir, 'RegionOfInterest', sn, f'Hem.{H}.nii') for H in gl.Hem]
    reginfo = load_reginfo(sn, glm)
    betas   = bt.make_cifti_betas(masks, gl.struct_cortex,
                                  betas=[os.path.join(path, f'beta_{i + 1:04d}.nii') for i in range(len(reginfo))],
                                  row_axis=nb.cifti2.ScalarAxis(reginfo.regname.astype(str)))
    W, names = chord_weights(reginfo, chords, targetforce)
    header   = nb.Cifti2Header.from_axes((nb.cifti2.ScalarAxis(names), betas.header.get_axis(1)))
    return nb.Cifti2Image(W @ betas.get_fdata(), header)


def load_chord_patterns(sn, glm=3, fname='chord_patterns.dscalar.nii'):
    """Saved chord patterns of `sn` as a volume, plus cond_vec (chordID) and part_vec (run) read off the row names."""
    cifti    = nb.load(os.path.join(arbuckleDir, f'glm{glm}', sn, fname))
    names    = pd.Series(cifti.header.get_axis(0).name).str.split('.', expand=True).astype(int)
    vol      = nt.volume_from_cifti(cifti, struct_names=gl.struct_cortex)
    return vol, names[0].to_numpy(), names[1].to_numpy()
