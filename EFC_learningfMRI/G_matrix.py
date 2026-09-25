import os
import itertools
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Sequence
import statsmodels.formula.api as smf   # heavy + optional; import only when used
import numpy as np
import pandas as pd
import PcmPy as pcm
from scipy.stats import linregress
from scipy.linalg import orthogonal_procrustes

import EFC_learningfMRI.globals as gl
from EFC_learningfMRI.util import get_trained_and_untrained, runs_to_keep, split_trained_untrained


def G_sorted(G, sns, order=None):
    """Reorder the 8 chords (rows AND columns) of one or more subjects' matrices.

    In G, subject s's rows/cols follow get_trained_and_untrained(sns[s]) — that
    subject's trained chord IDs first, then untrained. This permutes every subject
    onto a common `order` of chord IDs, so slot k holds the same chord for all.

    Args:
        G:     (8, 8) for a single subject, or (n_subj, 8, 8) with the first axis
               aligned with `sns`.
        sns:   participant number(s): a scalar for a 2D `G`, one per row for a 3D `G`.
        order: desired sequence of the 8 chord IDs. Any permutation is fine.
               Defaults to `gl.chordID` (the IDs sorted numerically), which is the
               same for every subject — so the default framing never depends on `sns`.

    Returns:
        Array shaped like `G` — (8, 8) in, (8, 8) out — reordered to `order`.
    """
    G      = np.asarray(G)
    sns    = np.atleast_1d(sns)
    single = G.ndim == 2                      # remember, so we hand back a 2D result

    if G.ndim not in (2, 3) or G.shape[-1] != G.shape[-2]:
        raise ValueError(f"G must be (8, 8) or (n_subj, 8, 8), got shape {G.shape}")
    if single:
        G = G[None]

    if len(sns) != len(G):
        raise ValueError(f"got {len(sns)} participant(s) for {len(G)} matrix/matrices")

    order = np.asarray(gl.chordID if order is None else order, dtype=int)

    chords_per_subj = [np.asarray(get_trained_and_untrained(sn), dtype=int) for sn in sns]

    if order.size != G.shape[1]:
        raise ValueError(f"order lists {order.size} chords but G is {G.shape[1]}x{G.shape[1]}")

    out = np.empty_like(G)
    for s, (sn, chords) in enumerate(zip(sns, chords_per_subj)):
        if set(chords.tolist()) != set(order.tolist()):
            raise ValueError(f"sn {sn}: chord set {sorted(chords.tolist())} "
                             f"!= order {sorted(order.tolist())}")
        slot_of = {chord: i for i, chord in enumerate(chords.tolist())}  # chordID -> its index in G[s]
        perm    = [slot_of[chord] for chord in order.tolist()]          # source index for each target slot
        out[s]  = G[s][np.ix_(perm, perm)]

    return out[0] if single else out


def calc_G(data, cond_vec, part_vec, session='all', repetition='all', centred=False, fixed_effect=True, crossval=True):
    """
    calc G matrix for runs in session

    ``centred`` removes each regressor's mean across voxels. It returns a new
    array rather than centring in place, so the same betas can be reused for
    another session.

    ``session`` and ``repetition`` are both read out of ``cond_vec`` (see
    ``runs_to_keep``); ``repetition`` therefore only works for a glm whose regressors
    carry a repetition field (``'session,repetition,chordID'``, e.g. glm2).
    """
    if centred:
        data = data - data.mean(axis=1, keepdims=True)

    keep = runs_to_keep(cond_vec, session=session, repetition=repetition)

    if crossval:
        G_obs, _ = pcm.est_G_crossval(data[keep],
                                      cond_vec[keep],
                                      part_vec[keep],
                                      X=pcm.indicator(part_vec[keep]) if fixed_effect else None)
    else:
        G_obs, _ = pcm.est_G(data[keep],
                            cond_vec[keep],
                            part_vec[keep],
                            X=pcm.indicator(part_vec[keep]) if fixed_effect else None)

    return G_obs


def G_scaling(G_ref, G_tar):
    """Full comparison of observed vs. scaling-predicted target dissimilarity.

    Returns a one-row DataFrame with all intermediate quantities and the column
    `residual` (observed - predicted), which is the statistic tested in the paper.

    Parameters
    ----------
    G_ref, G_tar : (K, K) array_like

    Returns
    -------
    pandas.DataFrame, one row, with columns:
        act_ref, act_target, scale,
        diss_ref, diss_target_observed, diss_pred,
        residual  (= diss_target_observed - diss_pred)
    """
    K = G_ref.shape[0]
    
    act_ref = np.nanmean(np.sqrt(np.diag(G_ref)))
    act_tar = np.nanmean(np.sqrt(np.diag(G_tar)))

    scale = act_tar / act_ref

    D_ref = np.sqrt(pcm.G_to_dist(G_ref))
    D_tar = np.sqrt(pcm.G_to_dist(G_tar))

    mask = np.tri(K, k=-1, dtype=bool)
 
    diss_ref = np.nanmean(D_ref[mask])
    diss_obs = np.nanmean(D_tar[mask])
    diss_pred = scale * diss_ref
 
    return pd.DataFrame({
        "act_ref"             : act_ref,
        "act_target"          : act_tar,
        "scale"               : scale,
        "diss_ref"            : diss_ref,
        "diss_target_observed": diss_obs,
        "diss_pred"           : diss_pred,
        "residual"            : diss_obs - diss_pred,
    }, index=[0])


def pair_index():
    """(row, col) indices of the chord pairs, split into the three pair groups.

    """
    mask                           = np.tri(8, k=-1, dtype=bool)
    mask_trained                   = mask.copy()
    mask_untrained                 = mask.copy()
    mask_trained[4:]               = False
    mask_untrained[:, :4]          = False
    mask_trained_untrained         = np.zeros((8, 8), dtype=bool)
    mask_trained_untrained[:4, 4:] = True

    # (classification, row indices, col indices) for the three chord-pair groups
    masks = {'trained'          : mask_trained,
             'untrained'        : mask_untrained,
             'trained_untrained': mask_trained_untrained}
    return {chord: np.where(m) for chord, m in masks.items()}


def G_rows(G, sn, **labels):
    """One row per chord pair of a single 8x8 G: its crossnobis distance, cosine and angle.

    ``labels`` are copied onto every row, and are what identifies the G the pair
    came from — Hem/roi/session for a neural G, metric/session for a force one.
    The ``pair`` label is order-normalised so both Gs key on the same pair id.
    """
    chords = get_trained_and_untrained(sn)
    D      = pcm.G_to_dist(G)
    cos    = pcm.G_to_cosine(G)

    rows = []
    for chord, (r, c) in pair_index().items():
        for ri, ci in zip(r, c):
            rows.append({'sn'        : sn,
                         **labels,
                         'chord'     : chord,
                         'pair'      : '-'.join(sorted([str(chords[ri]), str(chords[ci])])),
                         'crossnobis': D[ri, ci],
                         'cosine'    : 1 - cos[ri, ci],
                         #'theta'     : np.arccos(cos[ri, ci])
                         })
    return rows


def add_group_reference(df, keys, ref_session=3, crossval=False):
    """Attach the reference-session group geometry.

    """

    # group-mean geometry: across-subject mean of each chord pair in the reference
    # session, pooled over trained/untrained/mixed (no 'chord'/'session' in the key,
    # so all subjects contribute). Merged onto every row, so the *_group columns hold
    # the ref-session reference for every session.
    s_ref = df[df.session == ref_session]
    if crossval:
        # leave-one-subject-out: subtract each subject's own value from the pair sum,
        # so their reference is the mean over the other subjects. Keyed by subject too,
        # then merged so it broadcasts across that subject's sessions.
        ref = s_ref[keys + ['sn']].copy()
        for metric in ('crossnobis', 'cosine'):
            g = s_ref.groupby(keys)[metric]
            ref[f'{metric}_group'] = (g.transform('sum') - s_ref[metric]) / (g.transform('size') - 1)
        df = df.merge(ref, on=keys + ['sn'], how='left')
    else:
        ref = (s_ref.groupby(keys, as_index=False)
                 .agg(crossnobis_group=('crossnobis', 'mean'),
                      cosine_group    =('cosine',     'mean')))
        df  = df.merge(ref, on=keys, how='left')

    df['theta_group'] = np.arccos(df.cosine_group)

    return df


def make_fname(session, repetition):
    """The 'within_session.3.1' / 'across_session' part of a G filename."""
    fname = 'across_session' if session == 'all' else 'within_session'
    fname = fname + '.' + str(session) if session != 'all' else fname
    fname = fname + '.' + str(repetition) if repetition != 'all' else fname
    return fname
