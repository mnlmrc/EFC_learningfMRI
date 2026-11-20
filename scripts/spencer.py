import argparse
import inspect
import itertools
import os

import numpy as np
import scipy.io as sio
import nibabel as nb
import imaging_pipelines.model as model

import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.G_matrix as G_matrix
import EFC_learningfMRI.model_from_spencer as spencer


def single_finger_chord_patterns(sns=spencer.participants, glm=3, targetforce=None):
    """Write each Arbuckle participant's hypothetical chord patterns as cifti.

    Saved to arbuckleDir/glm<glm>/<sn>/chord_patterns.dscalar.nii, or chord_patterns.force<targetforce>.dscalar.nii
    when only one target force is used.
    """
    fname = 'chord_patterns.dscalar.nii' if targetforce is None else f'chord_patterns.force{targetforce}.dscalar.nii'
    for sn in sns:
        print(f'doing participant {sn}...')
        cifti = spencer.make_chord_patterns(sn, targetforce=targetforce, glm=glm)
        nb.save(cifti, os.path.join(spencer.arbuckleDir, f'glm{glm}', sn, fname))


def calc_G_rois(sns=spencer.participants, glm=3, atlas_name='ROI', fname='chord_patterns.dscalar.nii'):
    """G_rois: the ROI Gs of the chord patterns, as in pattern.calc_G_rois.

    The saved patterns are unwhitened, so each ROI is first prewhitened with the glm's ResMS.nii
    (what BetasPrewithenedLoader does for the real betas). Arbuckle has one session, so the
    filenames carry no session part: arbuckleDir/pcm/<sn>/G_obs.glm<glm>.<H>.<roi>.npy, etc.
    """
    for sn in sns:
        vol, cond_vec, part_vec = spencer.load_chord_patterns(sn, glm, fname)
        resms = nb.load(os.path.join(spencer.arbuckleDir, f'glm{glm}', sn, 'ResMS.nii'))
        for H, roi in itertools.product(gl.Hem, gl.rois[atlas_name]):

            print(f'rois, doing participant {sn}, {H}, {roi}...')

            mask = nb.load(os.path.join(spencer.arbuckleDir, 'RegionOfInterest', sn, f'{atlas_name}.{H}.{roi}.nii'))
            Y    = model.calc_prewhitened_betas(vol, resms, mask)

            G        = G_matrix.calc_G(Y, cond_vec, part_vec, centred=False)
            cov      = G_matrix.calc_G(Y, cond_vec, part_vec, centred=True)
            G_raw    = G_matrix.calc_G(Y, cond_vec, part_vec, centred=False, fixed_effect=False)
            G_noxval = G_matrix.calc_G(Y, cond_vec, part_vec, centred=False, fixed_effect=False, crossval=False)

            save_path = os.path.join(spencer.arbuckleDir, gl.pcmDir, sn)
            os.makedirs(save_path, exist_ok=True)

            np.save(os.path.join(save_path, f'G_obs.glm{glm}.{H}.{roi}'), G)
            np.save(os.path.join(save_path, f'cov.glm{glm}.{H}.{roi}'), cov)
            np.save(os.path.join(save_path, f'G_obs_raw.glm{glm}.{H}.{roi}'), G_raw)
            np.save(os.path.join(save_path, f'G_obs_noxval.glm{glm}.{H}.{roi}'), G_noxval)


def calc_G_natural(fname='gloveData_MCP_11.mat'):
    """G_natural: the chord G predicted from the natural statistics of hand use (glove data).

    Glove data: CyberGlove recordings of everyday hand use (Ingram et al., 2008; 6 subjects,
    11 sessions), as used in Arbuckle et al. (2020, J Neurosci). Signals are MCP joint angular
    velocities of the five digits (CyberGlove sensors 2, 5, 8, 12, 16; thumb to little finger).

    Preprocessing in gloveData_MCP_11.mat:
        1. Each digit's velocity was split into a flexion and an extension signal: flexion keeps
           the positive (flexing) samples, extension keeps the negative (extending) samples, and
           the opposite direction is set to 0. This gives 10 columns: flexion digits 1-5, then
           extension digits 1-5.
        2. Absolute values were taken, so extension columns are positive.
        3. Each column was normalized to unit length (L2 norm = 1) within each session.
    Because of step 3, X.T @ X for a session has ones on the diagonal and holds cosine
    similarities between the 10 finger actions (co-occurrence in natural use).
    Other versions: _00 = steps 1 only; _10 = steps 1 and 3 (extension columns stay negative).

    G_nat (10x10) is computed per session, averaged within subject, then across subjects.
    Each chord is coded as a 10-element indicator vector (chord digit 2 = flexion, 1 = extension,
    9 = not pressed), and the predicted chord G is C @ G_nat @ C.T (n_chords x n_chords).
    """

    nat_path = os.path.join(gl.baseDir, gl.arbuckleDir, gl.naturalDir)

    # ------------------------------------------------------------------------------------------
    # 1. Load. The .mat file holds one entry per recording SESSION (11 in total), not per subject.
    #    Subjects and sessions:  sn   = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    #                            sess = [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    #    i.e. subject 1 has a single session, subjects 2-6 have two sessions each.
    # ------------------------------------------------------------------------------------------
    d    = sio.loadmat(os.path.join(nat_path, fname))
    data = d['data'][:, 0]                                          # (n_sess,) object array; each = (n_samples x 10)
    sn   = d['sn'][:, 0]                                            # (n_sess,) subject id of each session

    # ------------------------------------------------------------------------------------------
    # 2. SESSION level: one 10x10 matrix per session.
    #    X has one column per finger action, each scaled to unit length within the session, so
    #        G_sess[k, l] = sum_t X[t, k] * X[t, l] = cos(angle between action k and action l)
    #    -> 1 on the diagonal, and off-diagonals = how much two actions co-occur over time.
    #    Notes:
    #      - Because of the unit-length normalization, the number of samples in a session
    #        (~430k-540k) does not matter: every session contributes a matrix on the same scale.
    #      - The columns are NOT mean-centred, so this is a cosine similarity (uncentred second
    #        moment), not a correlation. All signals are >= 0 (rectified |velocity|), so all
    #        entries are >= 0: two actions that are both often active get a high value.
    # ------------------------------------------------------------------------------------------
    G_sess = np.array([X.T @ X for X in data])                      # (n_sess x 10 x 10)

    # ------------------------------------------------------------------------------------------
    # 3. SUBJECT level: average the sessions belonging to the same subject.
    #    Subject 1 -> its single session; subjects 2-6 -> mean of their two sessions.
    # ------------------------------------------------------------------------------------------
    subjects = np.unique(sn)
    G_subj   = np.array([G_sess[sn == s].mean(axis=0) for s in subjects])  # (n_subj x 10 x 10)

    # ------------------------------------------------------------------------------------------
    # 4. GROUP level: average across subjects.
    #    Averaging in two steps (sessions -> subject -> group) gives every SUBJECT equal weight
    #    (1/6). Averaging the 11 sessions directly would instead give subject 1 weight 1/11 and
    #    each other subject 2/11.
    # ------------------------------------------------------------------------------------------
    G_nat = G_subj.mean(axis=0)                                     # (10 x 10)

    # ------------------------------------------------------------------------------------------
    # 5. From finger actions to chords.
    #    Each chord ID has one digit per finger (thumb first): 2 = flex, 1 = extend, 9 = not
    #    pressed. Code each chord as a 0/1 row over the 10 finger actions (same column order
    #    as above): e.g. 21911 -> thumb flex, index ext, ring ext, little ext
    #                          -> [1 0 0 0 0 | 0 1 0 1 1]
    # ------------------------------------------------------------------------------------------
    C = np.zeros((len(gl.chordID), 10))                             # (n_chords x 10)
    for i, ch in enumerate(gl.chordID):
        for f, code in enumerate(str(int(ch))):                     # f = finger index (0 = thumb, 4 = little)
            if code == '2':
                C[i, f]     = 1                                     # flexion of finger f   -> column f
            elif code == '1':
                C[i, 5 + f] = 1                                     # extension of finger f -> column 5 + f
            # code == '9': finger not pressed, row stays 0

    # ------------------------------------------------------------------------------------------
    # 6. Predicted chord G.
    #    Assumption: a chord's pattern is the SUM of the patterns of its finger actions, and the
    #    finger-action patterns have inner products G_nat. Then
    #        G[i, j] = sum_{k in chord i} sum_{l in chord j} G_nat[k, l] = (C @ G_nat @ C.T)[i, j]
    #    The diagonal (chord "size") is 4 + 2 * (sum of co-occurrences among the chord's 4
    #    actions), so chords whose actions co-occur more in natural use get larger, and chords
    #    sharing actions that co-occur get more similar.
    # ------------------------------------------------------------------------------------------
    G = C @ G_nat @ C.T                                             # (n_chords x n_chords), rows/cols in gl.chordID order

    np.save(os.path.join(nat_path, 'G_natural'), G)


# Step name -> function.
FUNC = {
    'single_finger_chord_patterns': single_finger_chord_patterns,
    'G_rois'                      : calc_G_rois,
    'G_natural'                   : calc_G_natural
}


def main(what, **kwargs):
    """Run one step.

    `kwargs` are forwarded to the step (`sns=`, `glm=`, `targetforce=`), but only the ones it accepts.
    """
    if what is not None:
        func     = FUNC[what]                                       # select function
        accepted = inspect.signature(func).parameters               # find what parameters are acceptable
        func(**{k: v for k, v in kwargs.items() if k in accepted})  # run the function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hypothetical chord patterns from the single-finger patterns of Arbuckle et al. (2020).')
    parser.add_argument('--what', default=None, choices=list(FUNC), help='which step to run (default: all)')
    parser.add_argument('--glm', type=int, default=None, help='GLM the betas come from (default: the step default, 3)')
    parser.add_argument('--sns', nargs='+', type=str, default=None, help='Arbuckle participant ids, e.g. s01 (default: all participants)')
    parser.add_argument('--targetforce', type=float, default=None, help='use only this target force in N, 1.5 or 2 keep the directions matched, single_finger_chord_patterns only (default: average the force levels)')
    parser.add_argument('--atlas_name', default=None, help='atlas whose ROIs to use, G_rois only (default: the step default, ROI)')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'what' and v is not None}
    main(args.what, **kwargs)

    if args.what is None:
        main('single_finger_chord_patterns', **kwargs)
        main('G_rois',                       **kwargs)
        main('G_natural',                    **kwargs)
