import argparse
from imaging_pipelines.searchlight import searchlight_surf
from sklearn.covariance import ledoit_wolf
import imaging_pipelines.model as model
import EFC_learningfMRI.globals as gl
import EFC_learningfMRI.betas as betas
import EFC_learningfMRI.G_matrix as G_matrix
import EFC_learningfMRI.util as util
import AnatSearchlight.searchlight as sl
from joblib import Parallel, delayed
import nibabel as nb
import nitools as nt
import PcmPy as pcm
import time
import os
import numpy as np


def make_searchlight(sn):
    path_surf = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}')
    white = [os.path.join(path_surf, f'subj{sn}.{H}.white.32k.surf.gii') for H in gl.Hem]
    pial = [os.path.join(path_surf, f'subj{sn}.{H}.pial.32k.surf.gii') for H in gl.Hem]
    mask = [os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'Hem.{H}.nii') for H in gl.Hem]
    savedir = os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}')
    searchlight_surf(white, pial, mask, savedir, maxradius=10, maxvoxels=100)


def calc_G_searchlight(data, cond_vec, part_vec, session):
    """Crossvalidated second moment matrix of one searchlight, with searchlight-local
    multivariate noise normalization (MNN).

    ``data`` stacks this session's raw betas (first ``part_vec.size`` rows) on top of the
    residual timeseries (remaining rows), both sampled for the same searchlight. The
    betas are whitened by the searchlight-local noise covariance (Sigma^-1/2, Ledoit-Wolf)
    before G is estimated, following Walther et al. (2016). Estimating the covariance
    within each ~100-voxel sphere keeps it tiny, so no global voxel x voxel matrix is ever
    formed.

    Returns the (K, K) G itself rather than a summary of it, so every distance
    (see :func:`G_to_metric`) can be read off the saved searchlight afterwards without
    re-running it. All-NaN where the sphere has too few usable voxels to whiten.
    """
    n_reg = part_vec.size
    K     = np.unique(cond_vec).size

    beta  = data[:n_reg]
    resid = data[n_reg:]

    # keep only voxels usable for whitening: present in the cifti (finite residuals),
    # with non-degenerate noise, and non-empty betas. Edge searchlights can lose them
    # all, which would make the covariance undefined -- return NaN there.
    ok = np.isfinite(resid).all(axis=0) & (resid.var(axis=0) > 1e-10) & ~np.isnan(beta).all(axis=0)
    if ok.sum() < 2:
        return np.full((K, K), np.nan)

    beta_pw = model._multivariate_prewhitening(beta[:, ok], resid[:, ok])
    beta_pw = beta_pw[:, ~np.isnan(beta_pw).any(axis=0)]
    if beta_pw.shape[1] == 0:
        return np.full((K, K), np.nan)

    return G_matrix.calc_G(beta_pw, cond_vec, part_vec, session, fixed_effect=False)


def G_to_metric(G, metric='crossnobis'):
    """Summarise saved searchlight Gs as overall / trained / untrained dissimilarity.

    Parameters
    ----------
    G : ndarray (n_centers, K, K) or (K, K)
        The searchlight Gs written by :class:`Searchlight`. Centres whose G is not finite
        (edge searchlights) come back as NaN rather than blanking the whole map.
    metric : {'crossnobis', 'cosine', 'theta'}
        Dissimilarity read off G. ``theta`` is the arccos of the *mean* cosine, so it is
        the angle of the average pair, not the average of the pairwise angles.

    Returns
    -------
    ndarray (n_centers, 3)
        Columns in the order of ``METRIC_LABELS[metric]``.
    """

    out = np.full((G.shape[0], 3), np.nan)
    ok  = np.isfinite(G).all(axis=(1, 2))      # G_to_cosine raises on non-finite input
    if not ok.any():
        return out

    if metric == 'crossnobis':
        D = pcm.G_to_dist(G[ok])
    elif metric in ('cosine', 'theta'):
        with np.errstate(divide='ignore', invalid='ignore'):
            C = pcm.G_to_cosine(G[ok])         # zero-variance chords -> NaN for that centre
        D = 1 - C if metric == 'cosine' else C
    else:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")

    tot, trained, untrained = util.split_trained_untrained(D)
    summary = np.column_stack([tot, trained.mean(axis=1), untrained.mean(axis=1)])

    out[ok] = np.arccos(summary) if metric == 'theta' else summary
    return out



def _residuals_at_candidate_voxels(R_all, brain_axis, voxel_indx, struct):
    """Sample the residual timeseries at a searchlight's candidate voxels, without densifying.

    The function takes the timeseries matrix from each voxel that is stored in the residual cifti and finds to which voxel it corresponds among the candidate voxels in the matrix. It returns
    a (n_timepoints, n_candidate_voxels) matrix, where candidate voxels are ordered in the same way as the get_fdata for the betas 

    Parameters
    ----------
    R_all : ndarray (n_timepoints, n_grayordinates)
        The raw cifti residual matrix, exactly as stored -- one column per grayordinate.
    brain_axis : nb.cifti2.BrainModelAxis
        The cifti's second axis. Each column is tagged with the structure it belongs to and, for a
        volume-based model like this GLM's residuals, the ``(i, j, k)`` voxel it came from.
    voxel_indx : ndarray (3, n_candidate_voxels), int
        ``SL.voxel_indx``: the i/j/k indices (in the functional volume grid) of every candidate
        voxel of this hemisphere's searchlight, i.e. the union of all spheres. ``SL.voxlist`` then
        indexes into these columns to pick out one sphere.
    struct : str
        ``'CortexLeft'`` or ``'CortexRight'`` -- which cifti structure to look the voxels up in.

    Returns
    -------
    ndarray (n_timepoints, n_candidate_voxels), float32
        Column ``p`` is the residual timeseries of candidate voxel ``voxel_indx[:, p]``, or all-NaN
        if that voxel is not in the cifti.

    """

    # target structure in cifti (e.g., CIFTI_STRUCTURE_CORTEX_LEFT for CortexLeft)
    target = nb.cifti2.BrainModelAxis.to_cifti_brain_structure_name(struct)
    
    shape  = brain_axis.volume_shape

    # init a vector with as many element as voxels in the original FOV, -1 where the cifti has no such voxel
    col_of = np.full(np.prod(shape), -1, dtype=np.int32)
    for nam, slc, bm in brain_axis.iter_structures():
        if nam == target:
            col_of[np.ravel_multi_index(bm.voxel.T, shape)] = np.arange(R_all.shape[1])[slc]
            break

    idx  = col_of[np.ravel_multi_index(voxel_indx, shape)]
    good = idx >= 0
    out  = np.full((R_all.shape[0], idx.size), np.nan, dtype=np.float32)
    out[:, good] = R_all[:, idx[good]]
    return out



class Searchlight():

    def __init__(self,
                 sns           = None,
                 glm           = None,
                 metric_fn     = None,
                 out_shape     = None,
                 sessions      = (3, 9, 23),
                 out_fname     = 'searchlight_G'):
        """Run one searchlight per subject and save its raw per-centre output.

        ``metric_fn`` defaults to :func:`calc_G_searchlight`: each centre's crossvalidated,
        multivariate-noise-normalized G, saved as an (n_centers, K, K) ``.npy``. Distances
        are then read off those Gs (:func:`G_to_metric`, :func:`make_distance_maps`), so a
        new metric costs a pass over the ``.npy`` instead of another searchlight run.

        ``out_shape`` is what one centre returns, and is only needed for the NaN filler of
        empty spheres; ``None`` infers ``(K, K)`` from the session's ``cond_vec``, which is
        right for any metric_fn returning a G.
        """
        self.sns             = gl.participants if sns is None else sns
        self.glm             = glm
        self.metric_fn       = calc_G_searchlight if metric_fn is None else metric_fn
        self.out_shape       = out_shape
        self.sessions        = sessions
        self.out_fname       = out_fname
        self.residual_fname = 'residual.dtseries.nii'


    def _searchlight_subject(self, sn):
        """

        """

        print('loading betas and residual timeseries...')

        # load reginfo
        reginfo = betas.RegInfo(sn, self.glm)

        # load betas
        beta_cifti = betas.load_betas(sn, self.glm)                          
        beta_array = beta_cifti.get_fdata(dtype=np.float32)

        # load residuals                
        residual_cifti = betas.load_residuals(sn, self.glm, self.residual_fname)  # residual.dtseries.nii (cifti)
        residual_array = np.asarray(residual_cifti.dataobj, dtype=np.float32)     # (n_timepoints, n_grayordinates)

        # retrieve brain axis
        brain_axis = residual_cifti.header.get_axis(1)

        for H in gl.Hem:
            print(f'loading searchlight {H}...')
            SL     = sl.load(os.path.join(gl.baseDir, gl.roiDir, f'subj{sn}', f'searchlight.{H}.h5'))
            struct = 'CortexLeft' if H == 'L' else 'CortexRight'

            # keep only voxels used in searchlight
            beta_cand  = beta_array[SL.voxel_indx[0], SL.voxel_indx[1], SL.voxel_indx[2]].T  # (n_cond, n_cand)
            resid_cand = _residuals_at_candidate_voxels(residual_array, brain_axis, SL.voxel_indx, struct) # (n_timepoints, n_cand)

            for session in self.sessions:
                print(f'running searchlight {H}, session {session}...')

                keep          = util.runs_to_keep(reginfo.cond_vec, session=session)
                function_args = {'cond_vec': reginfo.cond_vec[keep],
                                 'part_vec': reginfo.part_vec[keep],
                                 'session' : session,}
                data          = np.vstack([beta_cand[keep], resid_cand])   # betas on top of residuals

                K         = np.unique(function_args['cond_vec']).size
                out_shape = (K, K) if self.out_shape is None else self.out_shape

                result = self._run_parallel(data, SL.voxlist, self.metric_fn, function_args, out_shape)

                # (n_centers, K, K): one G per searchlight centre, so distance/cosine/theta
                # can be computed from it later (see make_distance_maps)
                fname = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'{self.out_fname}.{session}.glm{self.glm}.{H}.npy')
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                np.save(fname, result.astype(np.float32))


    @staticmethod
    def _run_parallel(data, voxlists, metric_fn, function_args, out_shape, n_jobs=8):
        """Run ``metric_fn`` over pre-sampled searchlight data (mirrors ``sl.run_parallel``).

        ``data`` is (n_measurements, n_candidate_voxels) already sampled at the searchlight's
        candidate voxels, so the residual timeseries never has to be densified to a volume for
        ``run_parallel`` to sample it. Each center slices its own voxels out of ``data``.

        ``out_shape`` is what one centre returns -- ``(K, K)`` for a G -- and is used to fill
        empty spheres with NaN so every centre stacks into the same array.
        """
        def _process_one(vl):
            if len(vl) == 0:
                return np.full(out_shape, np.nan, dtype=float)
            return metric_fn(data[:, vl], **function_args)

        with Parallel(n_jobs=n_jobs, batch_size=100, verbose=10) as parallel:
            results = parallel(delayed(_process_one)(vl) for vl in voxlists)
        return np.array(results)


    def run(self):
        """Run the searchlight for every subject; each writes one (n_centers, K, K) ``.npy``
        per hemisphere and session. Turn those into surface maps with
        :func:`make_distance_maps`, then pool across subjects with :func:`pool_searchlight`."""

        for sn in self.sns:
            print(f'doing participant {sn}...')
            self._searchlight_subject(sn)


def make_distance_maps(sns=None, glm=None, metric='crossnobis', sessions=gl.sessions,
                       fname='searchlight_G', out_fname=None):
    """Turn each subject's saved searchlight Gs into one metric map per hemisphere and session.

    Reads the (n_centers, K, K) ``.npy`` written by :class:`Searchlight` and writes
    ``<out_fname>.<session>.glm<glm>.<H>.func.gii`` with the three columns of
    ``METRIC_LABELS[metric]`` -- which is what :func:`pool_searchlight` then pools. Every
    metric reads the same Gs, so adding one is a pass over the ``.npy``, not a new searchlight.
    """
    sns         = gl.participants if sns is None else sns
    out_fname   = f'searchlight_{metric}' if out_fname is None else out_fname
    struct_dict = dict(zip(gl.Hem, gl.struct_cortex))

    for sn in sns:
        for H in gl.Hem:
            for session in sessions:
                G      = np.load(os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'{fname}.{session}.glm{glm}.{H}.npy'))
                result = G_to_metric(G, metric=metric)

                gifti = nt.make_func_gifti(result, anatomical_struct=struct_dict[H],
                                           column_names=list(METRIC_LABELS[metric]))
                nb.save(gifti, os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}',
                                            f'{out_fname}.{session}.glm{glm}.{H}.func.gii'))


def _trained_untrained_columns(cols):
    """Indices of the trained and untrained columns of a searchlight gifti.

    Found by suffix rather than by a hard-coded name, so the same pooling serves
    every metric in ``METRIC_LABELS`` ('encoding-trained', 'theta-trained', ...).
    """
    trained   = [i for i, c in enumerate(cols) if c.endswith('-trained')]
    untrained = [i for i, c in enumerate(cols) if c.endswith('-untrained')]
    if len(trained) != 1 or len(untrained) != 1:
        raise ValueError(f'expected exactly one -trained and one -untrained column, got {cols}')
    return trained[0], untrained[0]


def pool_searchlight(sns=None, glm=None, fname='searchlight_crossnobis', sessions=gl.sessions):
    """Average the per-subject searchlight maps into group maps, per hemisphere and session.

    Writes two files per hemisphere and session, in ``surfDir`` next to the subject
    folders they are pooled from:

    ``<fname>.<session>.glm<glm>.<H>.func.gii``
        group mean of every metric column (nan-safe, so a subject missing a searchlight
        centre does not blank it for everyone).
    ``<fname>_diff.<session>.glm<glm>.<H>.func.gii``
        group mean of each subject's trained-minus-untrained difference, i.e. the
        learning contrast, computed within subject before averaging.
    """
    sns         = gl.participants if sns is None else sns
    struct_dict = dict(zip(gl.Hem, gl.struct_cortex))

    for H in gl.Hem:
        struct = struct_dict[H]
        for session in sessions:
            data, data_diff = [], []
            for sn in sns:
                fpath = os.path.join(gl.baseDir, gl.surfDir, f'subj{sn}', f'{fname}.{session}.glm{glm}.{H}.func.gii')
                gifti = nb.load(fpath)
                cols  = nt.get_gifti_column_names(gifti)
                tr_col, untr_col = _trained_untrained_columns(cols)
                data_ = nt.get_gifti_data_matrix(gifti)
                data.append(data_)
                data_diff.append(data_[:, tr_col] - data_[:, untr_col])

            group = np.nanmean(np.stack(data, axis=0), axis=0)
            gifti = nt.make_func_gifti(group, anatomical_struct=struct, column_names=cols)
            nb.save(gifti, os.path.join(gl.baseDir, gl.surfDir, f'{fname}.{session}.glm{glm}.{H}.func.gii'))

            group_diff = np.nanmean(np.stack(data_diff, axis=0), axis=0)
            gifti      = nt.make_func_gifti(group_diff, anatomical_struct=struct, column_names=['trained-untrained'])
            nb.save(gifti, os.path.join(gl.baseDir, gl.surfDir, f'{fname}_diff.{session}.glm{glm}.{H}.func.gii'))


