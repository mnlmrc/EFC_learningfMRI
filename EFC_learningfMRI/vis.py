import numpy as np
import PcmPy as pcm
import seaborn as sb
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from scipy.stats import ttest_rel
import EFC_learningfMRI.globals as gl
import os
import EFC_learningfMRI.util as util
import pandas as pd


def custom_legend(fig, labels, colors, loc='outside right center', kind='line'):

    if kind == 'box':
        handles = [
            Patch(facecolor=color, label=label) for color, label in zip(colors, labels)
        ]
    elif kind == 'point':
        handles = [
            Line2D([0], [0], color=color, ls='', marker='o', label=label) for color, label in zip(colors, labels)
        ]
    else:
        handles = [
            Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(colors, labels)
        ]
    fig.legend(handles=handles, frameon=False, loc=loc)


def plot_example_trial(ax, tAx, F, n_ord=4, cutoff=5, plot_legend=False, plot_derivative=False):
    force_lp = util.lowpass_fir(F, n_ord=n_ord, cutoff=cutoff, fsample=gl.fsample['force'], axis=0)
    force_der = np.gradient(force_lp, 1 / gl.fsample['force'], axis=0)
    if plot_derivative==False:
        ax.plot(tAx, force_lp, lw=2, label=['thumb', 'index', 'middle', 'ring', 'pinkie'])
    else:
        ax.plot(tAx, force_der, lw=2, label=['thumb', 'index', 'middle', 'ring', 'pinkie'])
    ax.axhspan(-gl.fthresh, gl.fthresh, color='grey', alpha=0.2, lw=0)
    ax.axvline(0, color='k', lw=.8)
    ax.axhline(2, color='k', lw=.8, ls=':')
    ax.axhline(-2, color='k', lw=.8, ls=':')
    ax.axhline(5, color='k', lw=.8, ls=':')
    ax.axhline(-5, color='k', lw=.8, ls=':')
    ax.set_xlabel('time (s)')
    ax.set_yticks([-5, -2.5, 0, 2.5, 5])
    ax.legend(bbox_to_anchor=(1, .5), loc='center left', frameon=False) if plot_legend else None



def plot_example_trials(fig, axs, sessions, sn=101, n_block: list=None, n_trial: list=None, plot_derivative=False):

    ch_idx = np.array(gl.diffCols)

    for s, sess in enumerate(sessions):
        ax = axs[s]
        filename = os.path.join(gl.baseDir, 'behavioural', f'day{sess}', f'efc4_{sn}_{n_block[s]:02d}.mov')
        mov = load_mov(filename)[n_trial[s]]
        mov = mov[(mov[:, 1] == gl.wait_exec) | (mov[:, 1] == gl.wait_exec - 1) | (mov[:, 1] == gl.wait_exec + 1)]
        F = mov[:, ch_idx] * gl.fGain

        plot_example_trial(ax, F, f'Session {sess}', plot_legend=s==1, plot_derivative=plot_derivative)

        dat = pd.read_csv(os.path.join(gl.baseDir, 'behavioural', f'day{sess}', f'efc4_{sn}.dat'), sep='\t')
        dat_row = dat[(dat.TN == n_trial[s]) & (dat.BN == n_block[s])].reset_index()

        trained = util.get_trained_and_untrained(sn)
        
        trial_row = pd.Series(_trial_row(F, dat_row, trained=trained))
        print(f'Session {sess}, chordID: {trial_row["chordID"]}')
        print('Absolute force:')
        print(trial_row[['thumb_abs', 'index_abs', 'middle_abs', 'ring_abs', 'pinkie_abs']])
        print('Absolute force derivative:')
        print(trial_row[['thumb_der', 'index_der', 'middle_der', 'ring_der', 'pinkie_der']])
        print('\n')
    
        sb.despine(ax=ax, trim=True, left=(s>0))

    fig.suptitle('Example trials')

def plot_hrf_cut(fig, axs, df, rois):
    for r, roi in enumerate(rois):
        ax = axs[r]
        sb.lineplot(data=df[df.roi==roi], ax=ax, x='time', y='hrf', hue='type', errorbar='se', 
        err_kws={'linewidth': 0}, color='darkorange', palette=['darkorange', 'purple'], legend=False if r<len(rois)-1 else True)
        ax.set_title(roi)
        ax.set_ylabel('activation (a.u.)')
        ax.set_xlabel('time (s)')
        ax.axvline(0, color='k', lw=.8, ls=':')
        ax.axvline(6, color='k', lw=.8, ls='--')
        ax.axvline(12, color='k', lw=.8, ls='-.')
        ax.axhline(0, color='k', lw=.8, ls=':')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend('', frameon=False)
        if r>0:
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelbottom=False)
    fig.legend(loc='lower center', frameon=False, ncol=2)
    

def _hue_colors(df, hue, hue_order, palette, color):
    """Map each hue level to the colour seaborn will give it.

    Lets the group-level lines be drawn in the same colours as the scatter points
    without hard-coding a palette.
    """
    if hue is None:
        return {None: color}
    if isinstance(palette, dict):
        return palette
    if palette is None:
        colors = [color] * len(hue_order) if color is not None else sb.color_palette(n_colors=len(hue_order))
    else:
        colors = sb.color_palette(palette, n_colors=len(hue_order)) if isinstance(palette, str) else list(palette)
    return dict(zip(hue_order, colors))


def plot_pcm_corr(fig, axs, df, corr, rois=None, x='SNR', y='r_indiv', hue='chord', hue_order=None, palette=None, color=None, add_zero=True, group_col='r_group', roi_col=None, legend=True, **kwargs):

    kws = dict(s=25) | kwargs

    if roi_col is None:
        candidates = ['roi', 'regionname', 'region']
        roi_col = next((c for c in candidates if c in df.columns), None)
        if roi_col is None:
            raise KeyError(f"None of {candidates} in df contain the requested rois; pass roi_col explicitly.")

    df = df[df['corr'] == corr]

    if rois is None:
        rois = df[roi_col].unique()
    if hue is not None and hue_order is None:
        hue_order = list(df[hue].unique())

    colors = _hue_colors(df, hue, hue_order, palette, color)

    for r, roi in enumerate(rois):
        ax     = axs[r]
        df_roi = df[df[roi_col] == roi]
        sb.scatterplot(data      = df_roi ,
                       ax        = ax ,
                       x         = x ,
                       y         = y ,
                       hue       = hue ,
                       hue_order = hue_order ,
                       palette   = palette if hue is not None else None ,
                       color     = color ,
                       legend    = legend and r == len(rois)-1 ,
                       **kws)

        # group_col=None skips the dashed lines. Needed for permuted data, where
        # each relabeling has its own group r and picking one would be arbitrary.
        for level in (hue_order if hue is not None else [None]) if group_col else []:
            df_lvl = df_roi if level is None else df_roi[df_roi[hue] == level]
            if df_lvl.empty:
                continue
            ax.hlines(df_lvl[group_col].to_numpy()[0], df_lvl[x].min(), df_lvl[x].max(),
                      color=colors[level], ls='--')

        ax.set_title(roi)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        if r>0:
            sb.despine(ax=ax, left=True, bottom=True)
            ax.tick_params(left=False, bottom=False, labelbottom=False)
        else:
            sb.despine(ax=ax)
        if add_zero is not False:
            ax.axhline(0 if add_zero is True else add_zero, lw=.8, color='k', ls=':')


def plot_mat_sess(fig, axs, mats, sessions=gl.sessions, labels=None, n_trained=4, cmap='viridis',
                  vmin=None, vmax=None, cbar=True, cbar_label=None, **kwargs):
    """One square matrix per session, every panel on the same colour scale.

    The shared scale is the point: a change in overall magnitude across sessions
    shows as a change in colour instead of being normalised away.

    Args:
        mats:       (n_sess, K, K) array.
        sessions:   session label per panel; defaults to gl.sessions.
        labels:     tick labels for the K rows/columns, e.g. the chord IDs in
                    ``get_trained_and_untrained(sn)`` order.
        n_trained:  where to draw the trained/untrained divider; None for no divider.
        cbar_label: label for the single shared colourbar.
    """
    M      = np.asarray(mats)
    K      = M.shape[-1]
    labels = range(K) if labels is None else labels
    vmin   = M.min() if vmin is None else vmin
    vmax   = M.max() if vmax is None else vmax

    for a, sess in enumerate(sessions):
        ax = axs[a]
        im = ax.imshow(M[a], cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

        ax.set_xticks(range(K), labels, rotation=90)
        ax.set_yticks(range(K), labels if a == 0 else [''] * K)
        ax.set_title(f'session {sess}')
        ax.tick_params(length=0)

        # imshow centres the cells on the integers, so the block edge is at n-.5
        if n_trained:
            ax.axhline(n_trained - .5, color='k', lw=1)
            ax.axvline(n_trained - .5, color='k', lw=1)

    if cbar:
        fig.colorbar(im, ax=list(axs[:len(sessions)]), label=cbar_label, shrink=.6)


def plot_im_sess(fig, axs, df, rois=None, x='session', y=None, hue=None, hue_order=None, palette=None, color=None, add_zero=False, kind='point', roi_col=None, native_scale=True, legend=True, alpha=1, **kwargs):
    kws = {
        'point' : dict(dodge=.2, lw=2, ls='-', errorbar='se', estimator='mean'),
        'box'   : dict(showfliers=False, boxprops=dict(alpha=alpha)),
        'bar'   : dict(errorbar='se'),
        'strip' : dict(jitter=True, dodge=True, alpha=.2),
        'violin': dict(inner='point', split=False),
    }[kind] | kwargs

    # dodge only makes sense across hue levels; seaborn crashes on a numeric dodge without hue
    if hue is None:
        kws.pop('dodge', None)

    if roi_col is None:
        candidates = ['roi', 'regionname', 'region']
        roi_col = next((c for c in candidates if c in df.columns), None)
        if roi_col is None:
            raise KeyError(f"None of {candidates} in df contain the requested rois; pass roi_col explicitly.")
        
    if rois is None:
        rois = df[roi_col].unique()
        
    for r, roi in enumerate(rois):
        ax = axs[r] if len(rois) > 1 else axs
        common = dict(data         = df[df[roi_col]==roi], 
                      ax           = ax,
                      x            = x,
                      y            = y,
                      hue          = hue,
                      hue_order    = hue_order,
                      palette      = palette,
                      color        = color,
                      native_scale = native_scale,
                      legend       = legend and r == len(rois)-1)
        if kind == 'point':
            sb.pointplot(**common, **kws)
        elif kind == 'box':
            sb.boxplot(**common, **kws)
        elif kind == 'bar':
            sb.barplot(**common, **kws)
        elif kind == 'violin':
            sb.violinplot(**common, **kws)
        elif kind == 'strip':
            sb.stripplot(**common, **kws)
        ax.set_title(roi)
        ax.set_xlabel(None)
        if x is not None:
            ax.set_xticks(df[x].unique()) if df[x].dtype == int else None
        if r>0:
            sb.despine(ax=ax, left=True, bottom=True)
            ax.tick_params(left=False, bottom=False, labelbottom=False) if r>0 else None
        else:
            sb.despine(ax=ax)
        if add_zero is not False:
            ax.axhline(0 if add_zero is True else add_zero, lw=.8, color='k', ls=':')


def plot_ci_im_sess(fig, axs, df, rois=None, x='session', y=None, ci=('ci_low', 'ci_high'), hue=None, hue_order=None, palette=None, color=None, add_zero=False, roi_col=None, dodge=0, legend=True, **kwargs):
    """Point estimate plus a *precomputed* confidence interval, one panel per roi.

    Mirrors :func:`plot_im_sess`, but the interval is read from the two ``ci``
    columns instead of being estimated from repeated observations. Use it when the
    frame already holds one fitted value per (hue, roi, x) -- a regression intercept
    and its ``conf_int()``, say -- where seaborn has nothing left to aggregate and
    would draw a zero-width bar.

    ``dodge`` is in data units (not seaborn's categorical offset), since the session
    axis is numeric: it is the total width the hue levels are spread over at each x.
    """
    kws = dict(marker='o', ms=6, lw=2, capsize=0, elinewidth=2) | kwargs

    if roi_col is None:
        candidates = ['roi', 'regionname', 'region']
        roi_col = next((c for c in candidates if c in df.columns), None)
        if roi_col is None:
            raise KeyError(f"None of {candidates} in df contain the requested rois; pass roi_col explicitly.")

    if rois is None:
        rois = df[roi_col].unique()

    levels  = [None] if hue is None else list(hue_order if hue_order is not None else df[hue].unique())
    offsets = np.zeros(len(levels)) if len(levels) == 1 else np.linspace(-dodge / 2, dodge / 2, len(levels))

    lo, hi = ci
    for r, roi in enumerate(rois):
        ax    = axs[r]
        d_roi = df[df[roi_col] == roi]
        for l, level in enumerate(levels):
            d = (d_roi if level is None else d_roi[d_roi[hue] == level]).sort_values(by=x)
            # errorbar wants distances from the estimate, not the bounds themselves
            yerr = np.abs(np.c_[d[y] - d[lo], d[hi] - d[y]].T)
            ax.errorbar(d[x].to_numpy() + offsets[l], d[y].to_numpy(), yerr=yerr,
                        color=color if palette is None else palette[l],
                        label=level if (legend and r == len(rois) - 1) else '_nolegend_',
                        **kws)
        ax.set_title(roi)
        ax.set_xlabel(None)
        ax.set_xticks(df[x].unique()) if df[x].dtype == int else None
        if r > 0:
            sb.despine(ax=ax, left=True, bottom=True)
            ax.tick_params(left=False, bottom=False, labelbottom=False)
        else:
            sb.despine(ax=ax)
        if add_zero is not False:
            ax.axhline(0 if add_zero is True else add_zero, lw=.8, color='k', ls=':')


# sessions belonging to each training week, used to lay out the behavioural x axis
WEEKS = {1: (1, 5), 2: (6, 10), 3: (11, 15), 4: (16, 20), 5: (21, 24)}

def plot_behav_sess(fig, ax, df, x='session', y=None, hue=None, hue_order=None, palette=None, color=None, add_zero=False, kind='point', native_scale=False, legend=True, alpha=1, decor=True, **kwargs):
    kws = {
        'point' : dict(dodge=.2, lw=2, ls='-', errorbar='se', estimator='mean'),
        'box'   : dict(showfliers=False, boxprops=dict(alpha=alpha)),
        'bar'   : dict(errorbar='se'),
        'strip' : dict(jitter=True, dodge=True, alpha=.2),
        'violin': dict(inner='point', split=False),
    }[kind] | kwargs

    # dodge only makes sense across hue levels; seaborn crashes on a numeric dodge without hue
    if hue is None:
        kws.pop('dodge', None)

    common = dict(data         = df,
                  ax           = ax,
                  x            = x,
                  y            = y,
                  hue          = hue,
                  hue_order    = hue_order,
                  palette      = palette,
                  color        = color,
                  native_scale = native_scale,
                  legend       = legend)
    if kind == 'point':
        sb.pointplot(**common, **kws)
    elif kind == 'box':
        sb.boxplot(**common, **kws)
    elif kind == 'bar':
        sb.barplot(**common, **kws)
    elif kind == 'violin':
        sb.violinplot(**common, **kws)
    elif kind == 'strip':
        sb.stripplot(**common, **kws)
    ax.set_xlabel('')

    if add_zero is not False:
        ax.axhline(0 if add_zero is True else add_zero, lw=.8, color='k', ls=':')

    if not decor:
        sb.despine(ax=ax)
        return

    # decor is specified in session units; with native_scale=False seaborn puts the
    # sessions on categorical positions instead, so map session -> axis coordinate.
    # Read the categories off the axis rather than off df: df may cover only a subset
    # of the sessions already drawn (untrained chords are tested in 8 of 24 sessions),
    # and seaborn keeps the categories set by the first plot on these axes.
    ticks  = np.asarray(ax.get_xticks(), dtype=float)
    try:
        levels = np.asarray([float(t.get_text()) for t in ax.get_xticklabels()])
    except ValueError:  # non-numeric tick labels, fall back to the sessions in df
        levels = np.array([])
    if levels.size != ticks.size or levels.size == 0:
        levels, ticks = np.sort(df[x].unique()), np.arange(df[x].nunique())
    order  = np.argsort(levels)
    to_pos = (lambda v: np.asarray(v, dtype=float)) if native_scale else \
             (lambda v: np.interp(v, levels[order], ticks[order]))

    starts = [w[0] for w in WEEKS.values()]
    for start in starts[1:]:
        ax.axvline(to_pos(start - .5), color='k', lw=.8, ls=':')

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    for sess in gl.sessions:
        ax.axvspan(to_pos(sess - .5), to_pos(sess + .5), facecolor='lightgrey')

    ylim0, ylim1 = ax.get_ylim()[0], ax.get_ylim()[1]
    ydiff  = (ylim1 - ylim0)
    xticks = ylim0 - (.025 * ydiff)

    for sess in gl.sessions:
        ax.text(to_pos(sess), ylim0 + .05 * ydiff, 'fMRI', rotation=90, ha='center', va='bottom')
    for week, (first, last) in WEEKS.items():
        ax.text(to_pos((first + last) / 2), xticks, f'{week}', ha='center', va='top', transform=ax.transData)
    # 'week' goes under the week numbers, centred on the axes: in figure coordinates it
    # drifts with the layout (the ylabel pushes the axes centre off the figure centre)
    ax.annotate('week', xy=(.5, xticks), xycoords=blended_transform_factory(ax.transAxes, ax.transData),
                xytext=(0, -1.4 * rcParams['font.size']), textcoords='offset points',
                ha='center', va='top', annotation_clip=False)

    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False,)


CHORD_DIRECTION = {'1': 1, '2': -1, '9': 0}  # chordID digits: extension, flexion, finger not instructed


def plot_force_pattern(fig, axs, df, chords=None, x='finger', y='force_raw', order=None, hue=None, hue_order=None,
                       palette=None, color=None, add_zero=True, kind='point', chord_col='chordID',
                       instructed=True, legend=True, alpha=1, **kwargs):
    """One panel per chord: the force of each finger, i.e. the chord's force pattern.

    Mirrors :func:`plot_im_sess` with the chords in the place of the rois, so the
    panels share a y axis and the eight patterns can be compared by eye.

    `df` is expected to be the *long* force table (one row per trial x finger),
    already averaged to one value per subject x chord x finger -- the error bars
    are then the s.e.m. across subjects.

    Args:
        df:         long force table, pre-averaged within subject.
        chords:     chords to draw, one panel each; defaults to every chord in df.
        order:      finger order along x; defaults to gl.fingers.
        chord_col:  column holding the chord identity (default 'chordID').
        instructed: mark the instructed direction of each finger (up = extension,
                    down = flexion, nothing = finger not instructed), read off the
                    chordID digits. Only meaningful for a signed measure.
    """
    kws = {
        'point' : dict(dodge=.2, lw=2, ls='-', errorbar='se', estimator='mean'),
        'box'   : dict(showfliers=False, boxprops=dict(alpha=alpha)),
        'bar'   : dict(errorbar='se'),
        'strip' : dict(jitter=True, dodge=True, alpha=.2),
        'violin': dict(inner='point', split=False),
    }[kind] | kwargs

    # dodge only makes sense across hue levels; seaborn crashes on a numeric dodge without hue
    if hue is None:
        kws.pop('dodge', None)

    axs    = np.atleast_1d(np.asarray(axs, dtype=object)).ravel()
    order  = gl.fingers if order is None else order
    chords = sorted(df[chord_col].unique()) if chords is None else chords

    for c, chord in enumerate(chords):
        ax     = axs[c]
        common = dict(data      = df[df[chord_col] == chord],
                      ax        = ax,
                      x         = x,
                      y         = y,
                      order     = order,
                      hue       = hue,
                      hue_order = hue_order,
                      palette   = palette,
                      color     = color,
                      legend    = legend and c == len(chords) - 1)
        if kind == 'point':
            sb.pointplot(**common, **kws)
        elif kind == 'box':
            sb.boxplot(**common, **kws)
        elif kind == 'bar':
            sb.barplot(**common, **kws)
        elif kind == 'violin':
            sb.violinplot(**common, **kws)
        elif kind == 'strip':
            sb.stripplot(**common, **kws)

        ax.set_title(chord)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks(range(len(order)), order, rotation=90)

        # the digits sit in gl.fingers order, so digit i belongs to tick i
        if instructed:
            for i, digit in enumerate(str(chord)):
                d = CHORD_DIRECTION[digit]
                if d:
                    ax.plot(i, .98 if d > 0 else .02, marker='^' if d > 0 else 'v', ms=4, color='grey',
                            transform=blended_transform_factory(ax.transData, ax.transAxes), clip_on=False)

        # the y axis is shared, so only the leftmost column of the grid keeps it
        first_col = ax.get_subplotspec().is_first_col() if ax.get_subplotspec() is not None else c == 0
        if not first_col:
            sb.despine(ax=ax, left=True)
            ax.tick_params(left=False, labelleft=False)
        else:
            sb.despine(ax=ax)
        if add_zero is not False:
            ax.axhline(0 if add_zero is True else add_zero, lw=.8, color='k', ls=':')


def plot_behav(fig, ax, df, metric='ET', ylim=[0, 2.5], melt=False, id_vars=None, value_vars=None, var_name=None,
               ylabel=None, title=None):
    """
    Plot behavioural metrics assessed trial by trial
    Args:
        metric: 

    Returns: fig, ax

    """
    if melt and (id_vars is None or value_vars is None):
        pass # implement error

    max_bn = 0
    inset_list = []
    lines = []

    for day in df.day.unique():
        dat_tmp = df[df['day'] == day]
        if melt:
            dat_tmp = dat_tmp.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=metric)

        dat_tmp['BN'] = dat_tmp['BN'] + max_bn
        max_bn = dat_tmp.BN.max()
        min_bn = dat_tmp.BN.min()
        ax.text((max_bn + min_bn) / 2, ylim[0], f'{day}', ha='center', va='center', fontsize=8)
        ax.axvline(max_bn + .5, color='k', linestyle='-', lw=.8)
        ax.tick_params('x', bottom=False, labelbottom=False)
        ax.spines[['bottom', 'top', 'right']].set_visible(False)
        dat_bn = dat_tmp.groupby(['subNum', 'day', 'chord', 'BN']).mean(numeric_only=True).reset_index()

        dat_d = dat_tmp.groupby(['subNum', 'day', 'chord']).mean(numeric_only=True).reset_index()

        if len(dat_tmp.chord.unique()) == 1:
            fixed_width = 2.5
        else:
            fixed_width = 5

        center = (min_bn + max_bn) / 2
        x0 = center - fixed_width / 2
        y0 = ylim[1]
        height = (ylim[1] - ylim[0]) * .2
        inset = ax.inset_axes([x0, y0, fixed_width, height], transform=ax.transData)
        sb.barplot(data=dat_d, ax=inset, hue='chord', y=metric, legend=False, palette=['red', 'blue'],
                    hue_order=['trained', 'untrained'], errorbar='se')
        inset.spines[['top', 'right', 'bottom']].set_visible(False)
        inset.set_xticks([])

        # add sig bars
        if len(dat_tmp.chord.unique()) > 1:
            # do t-test
            a, b = dat_d[dat_d['chord'] == 'trained'][metric], dat_d[dat_d['chord'] == 'untrained'][metric]
            tval, pval = ttest_rel(a, b)
            lines.append(f'trained vs. untrained, day{day}: tval={tval:.3f}, pval={pval:.3f}')
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            elif pval < 0.05:
                stars = '*'
            else:
                stars = None
            ab = np.c_[a, b]
            bars = inset.patches
            # x1 = bars[0].get_x() + bars[0].get_width() / 2
            # x2 = bars[1].get_x() + bars[1].get_width() / 2
            if stars:
                # offset = .05 * inset.get_ylim()[1]
                y_max = ab.mean(axis=1).max()
                y_argmax = ab.mean(axis=1).argmax()
                se = ab[y_argmax].std() / np.sqrt(ab.shape[1])
                y_max += se

        if day == 1:
            sb.lineplot(data=dat_bn, ax=ax, x='BN', y=metric, hue='chord', errorbar='se', lw=1,
                         palette=['red', 'blue'], err_kws={'linewidth': 0}, legend=True)
            inset.set_ylabel(ylabel, fontsize=8)
        else:
            sb.lineplot(data=dat_bn, ax=ax, x='BN', y=metric, hue='chord', errorbar='se', lw=1,
                         palette=['red', 'blue'], err_kws={'linewidth': 0}, legend=False)
            inset.spines[['left']].set_visible(False)
            inset.set_yticks([])
            inset.set_ylabel('', fontsize=8)
        inset.set_ylim(ylim)
        inset_list.append(inset)

    ax.set_ylim(ylim)
    ax.set_xlim([-10, max_bn])
    ax.spines['left'].set_bounds(ylim)
    ax.text(max_bn / 2, ylim[0] - .05 * (ylim[1] - ylim[0]), '# session', ha='center', va='top', fontsize=10)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right', bbox_to_anchor=(1, -.01), ncol=2, frameon=False)
    ax.set_title(title)

    print("\n".join(lines))

    return fig, ax, inset_list

def plot_rep(fig, ax, df, metric='ET', ylim=[0, 2.5], ylabel=None, title=None):
    offset_rep = 0
    for day in df.day.unique():
        dat_tmp = df[df['day'] == day]
        dat_tmp.Repetition = dat_tmp.Repetition + offset_rep
        max_rep = dat_tmp.Repetition.max()
        min_rep = dat_tmp.Repetition.min()
        offset_rep += 2
        ax.tick_params('x', bottom=False, labelbottom=False)
        ax.spines[['bottom', 'top', 'right']].set_visible(False)

        dat_d = dat_tmp.groupby(['subNum', 'day', 'chord', 'Repetition']).mean(numeric_only=True).reset_index()

        sb.lineplot(data=dat_d, ax=ax, x='Repetition', y=metric, hue='chord', errorbar='se', lw=1, marker='s',
                    markeredgecolor=None, ms=3, palette=['red', 'blue'], err_kws={'linewidth': 0},
                    legend=True if day == 1 else False)
        ax.text((max_rep + min_rep) / 2, ylim[0], f'{day}', ha='center', va='center', fontsize=8)
        ax.axvline(max_rep + .5, color='k', linestyle='-', lw=.8)

    ax.set_ylim(ylim)
    ax.spines['left'].set_bounds(ylim)
    ax.set_ylabel(ylabel)
    ax.text(max_rep / 2, ylim[0] - .05 * (ylim[1] - ylim[0]), '# day', ha='center', va='top', fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1, -.01), ncol=2, frameon=False)
    ax.set_title(title)

    return fig, ax

def lineplot_roi_avg(fig, axs, df, metric, hue=None, hue_order=None, color=None, label=None,
                     H='L', rois=['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1'], ls='-',
                     bbox_to_anchor=(1, .5)):
    if isinstance(color, list):
        palette=color
    else:
        palette = None

    sess_map = {
        3: 0,
        9: 1,
        23: 2
    }
    df.loc[:, 'session'] = df.loc[:, 'session'].map(sess_map)
    for r, roi in enumerate(rois):
        ax = axs[r]
        sb.lineplot(df[(df['roi'] == roi) & (df['Hem'] == H)],
                        ax=ax,
                        y=metric,
                        x='session',
                        hue=hue,
                        palette=None if hue is None else palette,
                        color=None if isinstance(color, list) else color,
                        hue_order=hue_order,
                        errorbar='se',
                        legend=False,
                        err_kws={'linewidth': 0},
                        ls=ls
                        )
        ax.axhline(0, ls='-', color='k', lw=.8)
        ax.set_title(roi)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_facecolor('lightgrey')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines[['bottom']].set_bounds(0, 2)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['3', '9', '23'])
        if r == 0:
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_position(('data', -1))
        else:
            ax.tick_params(axis=('y'), labelleft=False, length=0)
    # fig.supylabel('activation (a.u.)')
    # fig.suptitle(f'Average activity in ROIs, hemisphere:{H}, N={N}')
    if label is not None:
        legend_handles = [Line2D([0], [0], color=col, label=lab, ls=ls) for col, lab in zip(color, label)]
        fig.legend(handles=legend_handles,
                   loc='center left',
                   bbox_to_anchor=bbox_to_anchor,
                   frameon=False,
                   ncol=1,
                   fontsize=10)

    return fig, axs


def add_significance_bars(ax, tAx, sig, color='black', position='bottom', height=0.02, alpha=.5, spacing=0.005,
                          linestyle='-', linewidth=4):
    """
    Adds a horizontal significance line above or below the signal.
    Automatically stacks multiple bars to avoid overlap.

    Parameters:
    - ax: matplotlib axis
    - tAx: time axis (1D)
    - sig: boolean array (same shape as tAx) indicating significance
    - color: line color
    - position: 'top' or 'bottom'
    - height: line height as fraction of axis height (0.02 = 2%)
    - spacing: vertical spacing between stacked bars (axes coords)
    - linestyle: style of the line ('-', '--', ':', '-.')
    - linewidth: thickness of the line
    """
    from itertools import groupby
    from operator import itemgetter
    import matplotlib.lines as mlines

    # Initialize storage for stacking info
    if not hasattr(ax, "_sig_bar_counts"):
        ax._sig_bar_counts = {"top": 0, "bottom": 0}

    offset_idx = ax._sig_bar_counts[position]

    transform = ax.get_xaxis_transform()  # x in data, y in axes coords

    if position == 'top':
        y = 1 - height - offset_idx * (height + spacing)
    else:
        y = offset_idx * (height + spacing)

    # Identify contiguous significant regions
    sig_regions = [(tAx[g[0][0]], tAx[g[-1][0]])
                   for k, g in groupby(enumerate(sig), key=itemgetter(1))
                   if k for g in [list(g)]]

    # Add lines for each significant region
    for start, end in sig_regions:
        line = mlines.Line2D([start, end], [y, y],
                             transform=transform,
                             color=color, alpha=alpha,
                             linestyle=linestyle,
                             linewidth=linewidth,
                             solid_capstyle='butt',
                             zorder=1e6)
        ax.add_line(line)

    # Increment stacking counter
    ax._sig_bar_counts[position] += 1


def add_grid_legend(fig, anchor=(0.05, 0.1, 0.1, 0.18),
                    col_labels=('day 1','day 2','day 3'),
                    row_labels=('trained','untrained'),
                    markers=('o','+','s'),
                    row_colors=('red','blue'),
                    markersize=30,
                    facecolor='lightgrey'):
    # anchor = (left, bottom, width, height) in figure fraction
    legax = fig.add_axes(anchor)
    legax.set_axis_off()

    nrows, ncols = len(row_labels), len(col_labels)
    legax.set_xlim(0, ncols + 1.2)   # +space for row labels
    legax.set_ylim(0, nrows + 1.2)
    legax.set_facecolor(facecolor)

    # column headers
    for j, lbl in enumerate(col_labels, start=1):
        legax.text(j, nrows + 0.9, lbl, ha='center', va='center', fontsize=10)

    # row labels + symbol cells
    for i, (r_lbl, c) in enumerate(zip(row_labels, row_colors), start=1):
        y = nrows + 1 - i
        legax.text(.075, y, r_lbl, ha='right', va='center', fontsize=10)
        for j, m in enumerate(markers, start=1):
            legax.scatter(j, y, s=markersize, marker=m, c=c)


def plot_mds(fig, axs, W, trained, rois, sessions=None, chords=None, hue_order=None, palette=None,
             origin=False, label_chords=False, legend=True):
    """Grid of MDS scatters: mean position of every chord when trained vs untrained.

    Each chord is trained in about half the participants, so it gets two points —
    the mean over those who trained it and the mean over those who did not — joined
    by an arrow pointing untrained -> trained. Error bars are the SEM over
    participants in each group.

    Args:
        fig, axs:     figure and a (n_sessions, n_rois) array of axes.
        W:            dict roi -> (n_subj, n_sess, n_chords, 2) coordinates, already
                      in a common frame (see `G_matrix.procrustes_mean`).
        trained:      (n_subj, n_chords) boolean, True where that participant
                      trained that chord. Columns must follow `chords`.
        rois:         ROI names, one per column of `axs`.
        sessions:     session labels, one per row of `axs`. Defaults to `gl.sessions`.
        chords:       chord IDs, used only for labelling. Defaults to `gl.chordID`.
        origin:       mark the origin and start the x axis there. Use with
                      coordinates from an uncentred MDS, where distance from the
                      origin is activity and dim 0 is the activity axis. With
                      centred coordinates the origin is just the mean pattern and
                      the axes are equal-aspect instead.
        label_chords: annotate each chord with its ID.
    """
    sessions  = gl.sessions if sessions is None else sessions
    chords    = gl.chordID  if chords   is None else chords
    hue_order = ['trained', 'untrained'] if hue_order is None else hue_order
    palette   = ['red', 'blue'] if palette is None else palette

    for r, roi in enumerate(rois):
        for e, session in enumerate(sessions):
            ax = axs[e, r]
            ax.axhline(0, lw=.4, color='lightgrey', zorder=-1)
            if origin:
                ax.plot(0, 0, '+', color='k', ms=7, mew=1)     # zero activity
            else:
                ax.axvline(0, lw=.4, color='lightgrey', zorder=-1)

            for c, chord in enumerate(chords):
                # mean over the participants who trained this chord, and over those who did not
                grp = [trained[:, c], ~trained[:, c]]
                m   = np.array([W[roi][g, e, c].mean(axis=0) for g in grp])
                se  = np.array([W[roi][g, e, c].std(axis=0, ddof=1) / np.sqrt(g.sum()) for g in grp])
                ax.annotate('', xy=m[0], xytext=m[1], zorder=0,          # untrained -> trained
                            arrowprops=dict(arrowstyle='->', color='grey', lw=.6, shrinkA=3, shrinkB=3))
                for i in range(2):
                    ax.errorbar(*m[i], xerr=se[i, 0], yerr=se[i, 1], fmt='o', ms=4,
                                color=palette[i], ecolor=palette[i], elinewidth=.6, alpha=.6)
                if label_chords:
                    ax.annotate(chord, m[0], textcoords='offset points', xytext=(4, 3),
                                fontsize=4.5, color='k', alpha=.7)

            ax.tick_params(labelsize=7)
            if origin:
                ax.set_xlim(left=0)
            else:
                ax.set_aspect('equal')
            if e == 0:
                ax.set_title(roi)
            if r == 0:
                ax.set_ylabel(f'session {session}\ndim 2', fontsize=8)
            if e == len(sessions) - 1:
                ax.set_xlabel('activity (dim 1)' if origin else 'dim 1', fontsize=8)

    if legend:
        custom_legend(fig, hue_order, palette, loc='outside lower right', kind='point')
