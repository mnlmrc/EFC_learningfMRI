import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.hrf import Optimise_HRF
from EFC_learningfMRI.util import load_glm_onset
import nitools.spm as spm
import numpy as np
import pandas as pd
import os

if __name__=='__main__':

    tAx = np.arange(-3, 17)

    glm = 3

    ### save segmented BOLD 
    df = pd.DataFrame()
    for H in gl.Hem:
        for r, roi in enumerate(gl.rois['ROI']):
            adj, hat, raw = [], [], []
            for sn in gl.participants:
                print(f'doing participant {sn}, {H}, {roi}, glm {glm}, fitted')

                # retrieve onsets
                onset = load_glm_onset(sn, glm)

                HRF = Optimise_HRF(sn=sn, glm=glm, H=H)

                y_cut_hat, y_cut_adj, y_cut_raw = HRF.cut(pre=3, roi=roi, post=16)

                raw.append(y_cut_raw.mean(axis=(0, 2)))
                hat.append(y_cut_hat.mean(axis=(0, 2)))
                adj.append(y_cut_adj.mean(axis=(0, 2)))
            
            raw = np.array(raw)
            adj = np.array(adj)
            hat = np.array(hat)

            df = pd.concat([df, pd.DataFrame(np.c_[raw.T, tAx], columns=sns + ['time']).assign(kind='raw', Hem=H, roi=roi)])
            df = pd.concat([df, pd.DataFrame(np.c_[adj.T, tAx], columns=sns + ['time']).assign(kind='adj', Hem=H, roi=roi)])
            df = pd.concat([df, pd.DataFrame(np.c_[hat.T, tAx], columns=sns + ['time']).assign(kind='hat', Hem=H, roi=roi)])
            
    df.to_csv(os.path.join(gl.baseDir, f'glm{glm}', 'hrf_fitted.tsv'), sep='\t', index=False)
