import argparse
import inspect
import itertools
import os
import pandas as pd
import numpy as np
import PcmPy as pcm
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.pcm as pcm_


def correlation_between_sessions(sns=gl.participants, glm=3, Hem=('L', 'R'), atlas_name='ROI', residual_fname='residual.dtseries.nii'):
    """Between-session correlation per (Hem, ROI, session-pair, chord set).

    Writes two tsvs to ``path_pcm``:
      - ``MLE_correlation`` : the PCM correlation-model fit (r_group/r_indiv + the
        model's own SNR).
      - ``xval_correlation``: the cross-validated cosine on voxel-centred betas
        (matched-chord diagonal of the across-session block -> Pearson-style r),
        plus the diag(G)/diag(Sig) SNR. The (n_subj, 8, 8) cov of each group is
        also saved as ``cov.corr_across_sess.*.npy``.
    """
    path_pcm = os.path.join(gl.baseDir, gl.pcmDir)
    Mflex    = pcm.CorrelationModel("flex", num_items=4, corr=None, cond_effect=True)
    corrs    = list(itertools.combinations(gl.sessions, 2))   # real session numbers, e.g. (3, 9)
    chords   = ['trained', 'untrained']

    correlation  = pcm_.CorrelationBetweenSessions(glm, sns=sns, atlas_name=atlas_name, Hem=Hem, residual_fname=residual_fname)
    Y, cov, snr  = correlation.group_datasets(corrs, chords)

    mle, xval = [], []
    for (H, roi, sessions, chord), Y_ in Y.items():

        spair = f'{sessions[0]}-{sessions[1]}'

        # PCM correlation model (MLE) -- keeps the model's own SNR
        r_indiv, r_group, SNR = pcm_.fit_correlation(Y_, Mflex)
        mle.append(pd.DataFrame({'sn'     : sns,
                                 'r_group': r_group,
                                 'r_indiv': r_indiv,
                                 'SNR'    : SNR,
                                 'chord'  : chord,
                                 'corr'   : spair,
                                 'roi'    : roi,
                                 'Hem'    : H,}))

        # cross-validated cosine -- matched-chord diagonal + diag(G)/diag(Sig) SNR
        G_group = cov[H, roi, sessions, chord]
        np.save(os.path.join(path_pcm, f'cov.corr_across_sess.glm{glm}.{spair}.{chord}.{H}.{roi}.npy'), G_group)

        r_xval = pcm.G_to_cosine(G_group)
        r_avg  = np.diagonal(r_xval[:, :4, 4:], axis1=1, axis2=2).mean(axis=1)
        xval.append(pd.DataFrame({'sn'     : sns,
                                  'r_indiv': r_avg,
                                  'r_group': r_avg.mean(),
                                  'SNR'    : snr[H, roi, sessions, chord],
                                  'corr'   : spair,
                                  'chord'  : chord,
                                  'roi'    : roi,
                                  'Hem'    : H}))

    pd.concat(mle, ignore_index=True).to_csv(os.path.join(path_pcm, f'MLE_correlation.{atlas_name}.glm{glm}.tsv'),  sep='\t', index=False)
    pd.concat(xval, ignore_index=True).to_csv(os.path.join(path_pcm, f'xval_correlation.{atlas_name}.glm{glm}.tsv'), sep='\t', index=False)


# Step name -> function.
FUNC = {
    'fit_correlation': correlation_between_sessions,
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`glm=`, `sns=`, ...), but only the ones it accepts.
    """
    if what is not None:
        func     = FUNC[what]                                       # select function
        accepted = inspect.signature(func).parameters               # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted})  # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Between-session correlation of the neural activity patterns.')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas come from (default: the step default, 3)')
    parser.add_argument('--sns', nargs='+', type=int, default=gl.participants, help='participant ids to include in the analysis')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)
