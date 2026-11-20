import pandas as pd
import numpy as np
import os
import itertools
from dataclasses import dataclass, field
from functools import cached_property
from typing import Sequence, Iterator

from SUITPy.atlas import summarize_data

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import RegInfo, add_chord_column

import nibabel as nb
import nitools as nt
import imaging_pipelines.betas as bt
from imaging_pipelines.model import calc_prewhitened_betas
import nitools.spm as spm

@dataclass
class BetasRoi:
    """Prewhitened betas of one ROI, with the labels needed to model them."""
    sn      : int
    Hem     : str
    roi     : str
    betas   : np.ndarray
    cond_vec: np.ndarray
    part_vec: np.ndarray


@dataclass
class BetasPrewithenedLoader:
    """Prewhitened betas of every subject and ROI, loading each subject once.

    Iterating is the whole API. The betas and residuals of a subject are read
    once and reused for every ROI, and each ROI is prewhitened once::

        for data in RoiLoader(glm=3):
            G = calc_G(data.betas, data.cond_vec, data.part_vec, session=3)
    """

    glm: int
    sns: Sequence[int]  = field(default_factory=lambda: gl.participants)
    atlas_name: str     = 'ROI'
    residual_fname: str = 'residual.dtseries.nii'
    Hem: Sequence[str]  = ('L', 'R')

    @property
    def rois(self) -> list:
        return gl.rois[self.atlas_name]

    def __iter__(self) -> Iterator[BetasRoi]:
        for sn in self.sns:
            reginfo   = RegInfo(sn, self.glm)
            betas     = load_betas(sn, self.glm)
            residuals = load_residuals(sn, self.glm, self.residual_fname)
            for H, roi in itertools.product(self.Hem, self.rois):
                print(f'doing participant {sn}, {H}, {roi}...')
                mask = nb.load(os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'{self.atlas_name}.{H}.{roi}.nii'))
                betas_prewhitened = calc_prewhitened_betas(betas, residuals, mask)
                yield BetasRoi(sn, H, roi, betas_prewhitened, reginfo.cond_vec, reginfo.part_vec)



def load_residuals(sn, glm, residual_fname='residual.dtseries.nii'):
    """Residuals of one subject.

    Loading does not depend on the ROI, so load once per subject and reuse for
    every mask. A cifti (``residual.dtseries.nii``) is left as-is: converting it
    to a volume densifies it to the full grid, which for a run of this length is
    tens of GB. ``ResMS.nii`` is a plain volume and selects the much cheaper
    univariate prewhitening downstream.
    """
    path_glm = os.path.join(gl.baseDir, f'{gl.glmDir}{glm}', f'subj{sn}')
    return nb.load(os.path.join(path_glm, residual_fname))
    


def load_betas(sn, glm, fname='beta.dscalar.nii'):
    """Betas of one subject, as a volume."""
    path_glm   = os.path.join(gl.baseDir, f'{gl.glmDir}{glm}', f'subj{sn}')
    beta_cifti = nb.load(os.path.join(path_glm, fname))
    return nt.volume_from_cifti(beta_cifti, struct_names=['CortexLeft', 'CortexRight'])


def roi_avg(sns=None, glm=None, atlas_name='ROI', cond_names=['chordID', 'session'], fname='contrast.dscalar.nii'):

    frames = []

    for sn in sns:
        # `desc` must have one row per FRAME of `fname` -- summarize_data's `frame`
        # column (0..n_frames-1) is what we merge on. Read the labels off the file's
        # own scalar axis. (reginfo has one row per run-wise regressor, e.g. 240, not
        # per contrast frame, e.g. 24, so building desc from it misaligns the merge.)
        reginfo   = RegInfo(sn, glm)
        condition = reginfo.condition_unique
        cond_name = pd.DataFrame({name: condition[c].to_numpy() for c, name in enumerate(cond_names)})
        vol       = load_betas(sn, glm, fname)

        for H in gl.Hem:
            print(f'doing participant {sn}, {H}...')

            
            masks = nb.load(os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'{atlas_name}.{H}.nii'))
            tmp   = summarize_data(vol, label_image=masks, region_names=['S1', 'M1', 'PMd', 'PMv', 'SMA', 'V1', 'SPLa', 'SPLp'])

            # attach condition labels by frame, then subject / hemisphere / chord
            tmp            = tmp.merge(cond_name, left_on='frame', right_index=True)
            tmp['sn']      = sn
            tmp['Hem']     = H
            tmp            = add_chord_column(tmp)
            tmp['session'] = tmp.session.map({'sess03': 3, 'sess09': 9, 'sess23': 23})

            frames.append(tmp)

    df = pd.concat(frames, ignore_index=True)
    
    df.to_csv(os.path.join(gl.baseDir, f'{gl.glmDir}{glm}', f'{atlas_name}.activation.tsv'), sep='\t', index=False)



@dataclass
class CiftiCortex:
    """Build and save the cortical CIFTI of one subject, one output ``type`` at a time.

    The shared setup (paths, masks, reginfo, row axis) lives in properties and each
    output type is its own method, so ``make(type)`` just dispatches to the method
    of that name. ``make_cifti_cortex`` remains as a thin wrapper for existing callers.
    """

    sn: int
    glm: int = None

    # --- shared setup --------------------------------------------------------
    @property
    def path_glm(self):
        return os.path.join(gl.baseDir, f'glm{self.glm}', f'subj{self.sn}')

    @property
    def path_rois(self):
        return os.path.join(gl.baseDir, gl.roiDir, f'subj{self.sn}')

    @property
    def masks(self):
        return [os.path.join(self.path_rois, f'Hem.{H}.nii') for H in gl.Hem]

    @cached_property
    def reginfo(self):
        return pd.read_csv(os.path.join(self.path_glm, 'reginfo.tsv'), sep='\t')

    @property
    def row_axis(self):
        return nb.cifti2.ScalarAxis(self.reginfo['name'] + '.' + self.reginfo['run'].astype(str))

    def _save(self, cifti, fname):
        nb.save(cifti, os.path.join(self.path_glm, fname))
        return cifti

    # --- one method per output type ------------------------------------------
    def beta(self):
        print(f'doing betas, participant {self.sn}')
        cifti = bt.make_cifti_betas(self.masks, gl.struct_cortex, path_glm=self.path_glm, row_axis=self.row_axis, )
        return self._save(cifti, 'beta.dscalar.nii')

    def repetition_suppression(self):
        cifti          = bt.make_cifti_contrasts(self.path_glm, self.masks, gl.struct_cortex, self.reginfo.name)
        brain_axis     = cifti.header.get_axis(1)
        regr           = pd.Series(cifti.header.get_axis(0).name)[::2]
        chord_sess_rep = regr.str.split(',', expand=True)
        row_axis       = chord_sess_rep.astype(str)[0] + ',' + chord_sess_rep[1]
        row_axis       = nb.cifti2.ScalarAxis(row_axis)
        data           = cifti.get_fdata()
        rep1           = data[::2]
        rep2           = data[1::2]
        suppr          = rep2 - rep1
        header         = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti_suppr    = nb.Cifti2Image(dataobj=suppr,  header=header)
        return self._save(cifti_suppr, 'rep_suppr.dscalar.nii')

    def residual(self):
        print(f'doing residuals, participant {self.sn}')
        residuals = bt.make_cifti_residuals(path_glm=self.path_glm, masks=self.masks, struct=gl.struct_cortex)
        return self._save(residuals, 'residual.dtseries.nii')

    def contrast(self):
        print(f'doing contrasts, participant {self.sn}')
        cifti = bt.make_cifti_contrasts(self.path_glm, self.masks, gl.struct_cortex, self.reginfo.name)
        return self._save(cifti, 'contrast.dscalar.nii')

    def psc(self):
        contrast  = nb.load(os.path.join(self.path_glm, 'contrast.dscalar.nii'))
        intercept = nb.load(os.path.join(self.path_glm, 'intercept.dscalar.nii'))
        SPM       = spm.SpmGlm(self.path_glm)
        SPM.get_info_from_spm_mat()
        cifti = bt.make_cifti_psc(contrast=contrast, intercept=intercept, SPM=SPM, masks=self.masks, struct=gl.struct_cortex)
        return self._save(cifti, 'psc.dscalar.nii')

    def intercept(self):
        session     = self.reginfo.name.str.split(',', n=1, expand=True)[1]
        nRuns       = [self.reginfo[session == sess].run.nunique() for sess in session.unique()]
        nRegressors = self.reginfo.shape[0]
        intercept   = []
        for sess in range(gl.nSess):
            for run in range(nRuns[sess]):
                intercept.append(os.path.join(self.path_glm, f'beta_0{nRegressors + run + 1 + sess * nRuns[0]}.nii'))
        cond_vec = np.sort(np.array([f'{sess},{run}' for run in range(nRuns[sess]) for sess in range(gl.nSess)]))
        row_axis = nb.cifti2.ScalarAxis(cond_vec)
        cifti    = bt.make_cifti_betas(self.masks, gl.struct_cortex, intercept, row_axis=row_axis, )
        return self._save(cifti, 'intercept.dscalar.nii')

