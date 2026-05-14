"""
Multi-observation FTS bandpass analysis.

Loads bandstats.csv files produced by passband_compute() across multiple
FTS observations and generates summary plots:

  plot_ivw_bandpasses   — IVW-averaged bandpass per feedhorn (all channels
                          overlaid) and a grand all-feeds average with optional
                          smoothing.
  plot_center_offsets   — Scatter of (measured − expected) band center vs
                          channel index, per feedhorn and averaged.
  plot_fwhm_comparison  — Scatter of measured FWHM vs channel index compared
                          to expected, per feedhorn and averaged.

Usage
-----
Edit the configuration block at the bottom, then run:

    python fts_multi_obs_analysis.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from scipy.ndimage import uniform_filter1d

from fts import expected_spec_f0, FMIN, FMAX

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_PATTERN   = './*mc_*/bandstats.csv'   # glob pattern for bandstats CSVs
SAVE_DIR      = './multi_obs_analysis/'
SMOOTH_KERNEL = 5    # uniform-filter width in samples; set to None to disable
# ─────────────────────────────────────────────────────────────────────────────

FREQ_GRID = np.linspace(FMIN, FMAX, 1000)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _reconstruct_bandpass(center, fwhm):
    """Normalised Gaussian bandpass on FREQ_GRID from fitted parameters."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    bp = np.exp(-(FREQ_GRID - center) ** 2 / (2.0 * sigma ** 2))
    return bp / bp.max()


def _ivw_avg(centers, fwhms, smooth_kernel=None):
    """
    IVW-average of reconstructed bandpasses weighted by 1/fwhm^2.

    Returns a normalised array on FREQ_GRID.
    """
    w = 1.0 / np.asarray(fwhms) ** 2
    w /= w.sum()
    avg = sum(wi * _reconstruct_bandpass(c, fw)
              for c, fw, wi in zip(centers, fwhms, w))
    peak = avg.max()
    if peak > 0:
        avg /= peak
    if smooth_kernel is not None and int(smooth_kernel) > 1:
        avg = uniform_filter1d(avg, size=int(smooth_kernel))
        peak2 = avg.max()
        if peak2 > 0:
            avg /= peak2
    return avg


def load_bandstats(csv_paths):
    """Load and concatenate bandstats.csv files from multiple observations."""
    dfs = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            df['source_csv'] = path
            dfs.append(df)
        except Exception as exc:
            print('[load_bandstats] Cannot read %s: %s' % (path, exc))
    if not dfs:
        raise RuntimeError('[load_bandstats] No CSV files loaded.')
    return pd.concat(dfs, ignore_index=True)


# ── Public analysis functions ─────────────────────────────────────────────────

def plot_ivw_bandpasses(csv_paths, save_dir, smooth_kernel=None):
    """
    IVW-average bandpasses across observations and save plots.

    Two output figures:
      ivw_bandpasses_xf0_<N>.png — per-feedhorn: all channels overlaid,
                                   normalised to peak = 1.
      ivw_bandpass_all_feeds.png — grand average over all detectors (faint
                                   per-detector lines + bold mean ±
                                   optional smoothed overlay).

    Parameters
    ----------
    csv_paths : list of str
        Paths to bandstats.csv files from individual observations.
    save_dir : str
        Output directory.
    smooth_kernel : int or None
        Uniform filter width (samples) applied to the all-feeds grand average.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = load_bandstats(csv_paths)

    feedhorns = sorted(df['xf0'].unique())
    channels  = sorted(df['xf1'].unique())
    n_ch      = max(channels) + 1 if channels else 60
    ch_cmap   = plt.get_cmap('viridis', n_ch)

    all_det_avgs = []

    for xf0 in feedhorns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Normalised power')
        ax.set_title('IVW-averaged bandpasses — feedhorn %s' % xf0)
        ax.set_xlim(FMIN, FMAX)

        sub = df[df['xf0'] == xf0]
        for xf1 in sorted(sub['xf1'].unique()):
            rows   = sub[sub['xf1'] == xf1]
            c_vals = rows['center'].values
            fw_vals = rows['fwhm'].values
            valid  = np.isfinite(c_vals) & np.isfinite(fw_vals) & (fw_vals > 0)
            if valid.sum() == 0:
                continue
            avg_bp = _ivw_avg(c_vals[valid], fw_vals[valid])
            ax.plot(FREQ_GRID, avg_bp,
                    color=ch_cmap(xf1 / (n_ch - 1)), linewidth=1, alpha=0.85)
            all_det_avgs.append(avg_bp)

        sm = plt.cm.ScalarMappable(cmap=ch_cmap,
                                   norm=plt.Normalize(0, n_ch - 1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Channel index')
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'ivw_bandpasses_xf0_%s.png' % xf0),
                    dpi=150)
        plt.close(fig)

    if all_det_avgs:
        grand_avg = np.nanmean(np.array(all_det_avgs), axis=0)
        grand_avg /= grand_avg.max()

        fig, ax = plt.subplots(figsize=(10, 5))
        for bp in all_det_avgs:
            ax.plot(FREQ_GRID, bp, color='steelblue', alpha=0.08, linewidth=0.6)
        ax.plot(FREQ_GRID, grand_avg, color='black', linewidth=2,
                label='Grand average')
        if smooth_kernel is not None and int(smooth_kernel) > 1:
            smoothed = uniform_filter1d(grand_avg, size=int(smooth_kernel))
            peak_s = smoothed.max()
            if peak_s > 0:
                smoothed /= peak_s
            ax.plot(FREQ_GRID, smoothed, color='red', linewidth=2,
                    linestyle='--',
                    label='Smoothed (kernel=%d)' % int(smooth_kernel))
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Normalised power')
        ax.set_title('All-feeds IVW-averaged bandpass  (%d detectors)' % len(all_det_avgs))
        ax.set_xlim(FMIN, FMAX)
        ax.legend(fontsize=9)
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'ivw_bandpass_all_feeds.png'), dpi=150)
        plt.close(fig)

    print('[plot_ivw_bandpasses] Saved to %s' % save_dir)


def plot_center_offsets(csv_paths, save_dir):
    """
    Scatter of (measured − expected) band center vs channel index.

    Two output figures:
      center_offset_per_feedhorn.png — grid of subplots, one per feedhorn.
      center_offset_all_feeds.png    — mean ± std/√N averaged over feedhorns.

    Parameters
    ----------
    csv_paths : list of str
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)
    df = load_bandstats(csv_paths)
    df['offset'] = df['center'] - df['expected_center']

    feedhorns = sorted(df['xf0'].unique())
    n_fh  = len(feedhorns)
    ncols = min(4, n_fh)
    nrows = int(np.ceil(n_fh / ncols))

    fig_per, axs_per = plt.subplots(nrows, ncols,
                                     figsize=(4.5 * ncols, 3.5 * nrows),
                                     sharex=True, sharey=True,
                                     squeeze=False)
    axs_flat = axs_per.ravel()

    for ai, xf0 in enumerate(feedhorns):
        ax  = axs_flat[ai]
        sub = df[df['xf0'] == xf0]
        grp = sub.groupby('xf1')['offset']
        means  = grp.mean()
        counts = grp.count()
        errs   = grp.std().fillna(0) / np.sqrt(np.maximum(counts, 1))

        ax.errorbar(means.index, means.values, yerr=errs.values,
                    fmt='o', ms=3, capsize=2, elinewidth=0.8, color='steelblue')
        ax.axhline(0, color='red', linestyle='dashed', linewidth=0.8)
        ax.set_title('Feed %s' % xf0, fontsize=9)
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True,
                       labelsize=7)

    for ai in range(n_fh, len(axs_flat)):
        axs_flat[ai].set_visible(False)

    fig_per.supxlabel('Channel index', fontsize=10)
    fig_per.supylabel(r'$\Delta f_0$ [GHz]  (measured $-$ expected)', fontsize=10)
    fig_per.suptitle('Band centre offset per feedhorn', fontsize=11)
    fig_per.tight_layout()
    fig_per.savefig(os.path.join(save_dir, 'center_offset_per_feedhorn.png'), dpi=150)
    plt.close(fig_per)

    grp_all    = df.groupby('xf1')['offset']
    means_all  = grp_all.mean()
    counts_all = grp_all.count()
    errs_all   = grp_all.std().fillna(0) / np.sqrt(np.maximum(counts_all, 1))

    fig_avg, ax_avg = plt.subplots(figsize=(10, 4))
    ax_avg.errorbar(means_all.index, means_all.values, yerr=errs_all.values,
                    fmt='o', ms=4, capsize=3, elinewidth=1, color='steelblue',
                    label='Mean ± std/√N')
    ax_avg.axhline(0, color='red', linestyle='dashed', linewidth=1,
                   label='Expected')
    ax_avg.set_xlabel('Channel index')
    ax_avg.set_ylabel(r'$\Delta f_0$ [GHz]  (measured $-$ expected)')
    ax_avg.set_title('Band centre offset — all feedhorns averaged')
    ax_avg.legend(fontsize=9)
    ax_avg.minorticks_on()
    ax_avg.tick_params(which='both', direction='in', top=True, right=True)
    fig_avg.tight_layout()
    fig_avg.savefig(os.path.join(save_dir, 'center_offset_all_feeds.png'), dpi=150)
    plt.close(fig_avg)

    print('[plot_center_offsets] Saved to %s' % save_dir)


def plot_fwhm_comparison(csv_paths, save_dir, expected_fwhm=None):
    """
    Scatter of measured FWHM vs channel index, compared to expected.

    Two output figures:
      fwhm_per_feedhorn.png — grid of subplots, one per feedhorn.
      fwhm_all_feeds.png    — mean ± std/√N averaged over feedhorns.

    Parameters
    ----------
    csv_paths : list of str
    save_dir : str
    expected_fwhm : float, array-like, or None
        Expected FWHM [GHz] per channel.  Scalar = same for all channels.
        None = use channel spacing from expected_spec_f0 as a reference.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = load_bandstats(csv_paths)

    if expected_fwhm is None:
        ch_spacing = np.diff(expected_spec_f0)
        exp_fw = np.append(ch_spacing, ch_spacing[-1])
    elif np.isscalar(expected_fwhm):
        exp_fw = np.full(len(expected_spec_f0), float(expected_fwhm))
    else:
        exp_fw = np.asarray(expected_fwhm, dtype=float)

    exp_ch_idx = np.arange(len(exp_fw))

    feedhorns = sorted(df['xf0'].unique())
    n_fh  = len(feedhorns)
    ncols = min(4, n_fh)
    nrows = int(np.ceil(n_fh / ncols))

    fig_per, axs_per = plt.subplots(nrows, ncols,
                                     figsize=(4.5 * ncols, 3.5 * nrows),
                                     sharex=True, sharey=True,
                                     squeeze=False)
    axs_flat = axs_per.ravel()

    for ai, xf0 in enumerate(feedhorns):
        ax  = axs_flat[ai]
        sub = df[df['xf0'] == xf0]
        grp = sub.groupby('xf1')['fwhm']
        means  = grp.mean()
        counts = grp.count()
        errs   = grp.std().fillna(0) / np.sqrt(np.maximum(counts, 1))

        ax.errorbar(means.index, means.values, yerr=errs.values,
                    fmt='o', ms=3, capsize=2, elinewidth=0.8,
                    color='steelblue', label='Measured')
        ax.plot(exp_ch_idx, exp_fw, 'r--', linewidth=0.8, label='Expected')
        ax.set_title('Feed %s' % xf0, fontsize=9)
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True,
                       labelsize=7)

    if len(axs_flat) > 0:
        axs_flat[0].legend(fontsize=7)
    for ai in range(n_fh, len(axs_flat)):
        axs_flat[ai].set_visible(False)

    fig_per.supxlabel('Channel index', fontsize=10)
    fig_per.supylabel('FWHM [GHz]', fontsize=10)
    fig_per.suptitle('Bandpass FWHM per feedhorn', fontsize=11)
    fig_per.tight_layout()
    fig_per.savefig(os.path.join(save_dir, 'fwhm_per_feedhorn.png'), dpi=150)
    plt.close(fig_per)

    grp_all    = df.groupby('xf1')['fwhm']
    means_all  = grp_all.mean()
    counts_all = grp_all.count()
    errs_all   = grp_all.std().fillna(0) / np.sqrt(np.maximum(counts_all, 1))

    fig_avg, ax_avg = plt.subplots(figsize=(10, 4))
    ax_avg.errorbar(means_all.index, means_all.values, yerr=errs_all.values,
                    fmt='o', ms=4, capsize=3, elinewidth=1, color='steelblue',
                    label='Measured mean ± std/√N')
    ax_avg.plot(exp_ch_idx, exp_fw, 'r--', linewidth=1.5, label='Expected')
    ax_avg.set_xlabel('Channel index')
    ax_avg.set_ylabel('FWHM [GHz]')
    ax_avg.set_title('Bandpass FWHM — all feedhorns averaged')
    ax_avg.legend(fontsize=9)
    ax_avg.minorticks_on()
    ax_avg.tick_params(which='both', direction='in', top=True, right=True)
    fig_avg.tight_layout()
    fig_avg.savefig(os.path.join(save_dir, 'fwhm_all_feeds.png'), dpi=150)
    plt.close(fig_avg)

    print('[plot_fwhm_comparison] Saved to %s' % save_dir)


if __name__ == '__main__':
    csv_paths = sorted(glob(CSV_PATTERN))
    print('Found %d bandstats files.' % len(csv_paths))
    if not csv_paths:
        raise SystemExit('No CSV files found — check CSV_PATTERN.')

    plot_ivw_bandpasses(csv_paths,
                        os.path.join(SAVE_DIR, 'bandpasses'),
                        smooth_kernel=SMOOTH_KERNEL)
    plot_center_offsets(csv_paths,
                        os.path.join(SAVE_DIR, 'center_offsets'))
    plot_fwhm_comparison(csv_paths,
                         os.path.join(SAVE_DIR, 'fwhm'))
