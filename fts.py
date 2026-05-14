from timesoft.timestream.timestream_tools import Timestream 
import numpy as np
import matplotlib.pyplot as plt 
from glob import glob
import os 
import pandas as pd
import traceback

from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt
from collections import defaultdict

# Design bands
expected_spec_f0 = [183.71801232, 185.16136904, 186.64257634, 188.16279125,
       189.72321432, 191.32509162, 192.96971692, 194.65843399,
       196.25228765, 198.0296243 , 199.85528592, 201.73083689,
       203.6579049 , 205.63818422, 207.67343911, 209.76550744,
       211.73616035, 213.94258352, 216.21163982, 218.54549518,
       220.94641077, 223.41674823, 225.95897538, 228.57567218,
       230.67665297, 232.49762763, 234.35463586, 236.24861235,
       238.18052371, 240.15136978, 242.16218512, 244.21404055,
       246.30804472, 248.44534582, 250.62713335, 252.85464004,
       254.95881035, 257.27800588, 259.64679464, 262.06659921,
       264.53889634, 267.06521948, 269.6471616 , 272.28637807,
       274.9845898 , 277.74358649, 280.56523012, 283.45145867,
       286.17572325, 289.19192083, 292.27884272, 295.43876537,
       298.67406315, 301.98721366, 305.38080342, 308.85753398,
       312.42022833, 316.07183793, 319.81545004, 323.65429567]

# Global variables
FMIN, FMAX, FRANGE = 180, 330, 12
PRUNE_CUTOFF = 300
SNR = 2
LW = 1.5  # global linewidth for all debug plots

# Helper methods
bin_edges = np.arange(-850, 150, 0.05)

def gaussian(f, sigma, f0, A):
    return A * np.exp(-(f - f0) ** 2 / (2 * sigma ** 2))

def gaussian_fwhm(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * np.abs(sigma)

def log2ceil(x):
    return int(2 ** np.ceil(np.log2(x)))

def prune(arr, cutoff=PRUNE_CUTOFF):
    if len(arr) == 0:
        return np.array([])
    keep = [arr[0]]
    for v in arr[1:]:
        if v - keep[-1] >= cutoff:
            keep.append(v)
    return np.array(keep)

def bandpass_filter(data, low_ghz, high_ghz, step_m):
    """Apply a bandpass filter to interferogram data.

    Parameters
    ----------
    data : array
        The interferogram data to filter
    low_ghz : float
        Lower frequency cutoff in GHz
    high_ghz : float
        Upper frequency cutoff in GHz
    step_m : float
        The position step size in meters (used to compute Nyquist frequency)
    """
    nyquist = 1 / (2 * step_m)
    low = (low_ghz * 1e9 / 3e8) / nyquist
    high = (high_ghz * 1e9 / 3e8) / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)


def passband_compute(MCE, obs_ids, file_paths, encoder_paths, OPD_FACTOR=2, debug=True,
                     run_log_path='run_failures.log'):
    """Compute the passband for each detector.

    Parameters
    ----------
    MCE : int
        The MCE to load
    obs_ids : list
        List of observation IDs
    file_paths : list
        List of file paths to raw data
    encoder_paths : list
        List of paths to encoder data
    OPD_FACTOR : int, default=2
        Factor to account for the fact that OPD = 2 * mirror displacement
    debug : bool, default=True
        If True, save debugging plots for each step
    run_log_path : str, default='run_failures.log'
        Path to a log file where per-observation failures are recorded
    """
    mmstep = 0.05e-3 * OPD_FACTOR  # optical path difference step in meters

    for obs_id, path, encoder_path in zip(obs_ids, file_paths, encoder_paths):
        ts = Timestream(path, mc=MCE, version='2024.dev.1', skip_last=False, mode='raw')

        out_path = str(obs_id) + 'mc_%s' % MCE
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        xd = np.vstack([np.load(a) for a in sorted(
            glob(encoder_path + '*.npy'),
            key=lambda x: float(x[len(encoder_path) + 8:-4])
        )]).T
        enc_t = xd[0]
        enc_pos = xd[1]
        interp_func = interp1d(enc_t, enc_pos)
        new_pos = interp_func(ts.t[ts.t < enc_t.max()])

        print(f"Completed loading data from {obs_id} : MCE{MCE}.")

        xf0_unique = sorted(set([xf[0] for xf in ts.header['xf_coords']]))

        fit_results = []

        fail_counts = {
            'no_scans': 0,
            'low_snr': 0,
            'bad_frequency': 0,
            'bad_fwhm': 0,
            'fit_failed': 0,
            'too_few_points': 0,
        }
        # Detailed per-detector failure log entries
        fail_details = []

        bpass_fig, bpass_axs = plt.subplots(1, figsize=(18, 6))
        bpass_axs.set_xlim(180, 330)

        for row in xf0_unique:
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))
            ax.set_xlabel('Frequency [GHz]')
            ax.set_ylabel('Normalized Power')

            xfs_row = [xf for xf in ts.header['xf_coords'] if xf[0] == row]

            for xf in xfs_row:
                figs, axs = plt.subplots(2)

                idx = ts.get_xf(xf[0], xf[1])

                new_pos_shifted = new_pos - np.nanmax(new_pos)
                ts_first_cut = ts.data[idx][ts.t < enc_t.max()]

                good = (new_pos_shifted > -995) & (new_pos_shifted < 0)
                pos_good = new_pos_shifted[good]
                inds = np.arange(len(ts_first_cut))[good]

                low_mask = np.isclose(pos_good, -995, atol=0.5)
                high_mask = np.isclose(pos_good, -5, atol=0.5)
                low_inds = prune(inds[low_mask])
                high_inds = prune(inds[high_mask])

                if len(low_inds) != len(high_inds):
                    msg = f'x={xf[0]} f={xf[1]}: {len(low_inds)} scan starts but {len(high_inds)} scan ends — truncating'
                    print(f'\tWarning: {msg}')
                    fail_details.append(f'[WARN] obs={obs_id} MCE={MCE} {msg}')

                scans_data, scans_pos = [], []
                for lo, hi in zip(low_inds, high_inds):
                    if lo < hi:
                        scans_data.append(ts_first_cut[lo:hi])
                        scans_pos.append(new_pos_shifted[lo:hi])

                if len(scans_data) == 0:
                    msg = f'x={xf[0]} f={xf[1]}: no valid scans'
                    print(f'\tWarning: {msg}')
                    fail_counts['no_scans'] += 1
                    fail_details.append(f'[FAIL:no_scans] obs={obs_id} MCE={MCE} {msg}')
                    plt.close(figs)
                    continue

                # DEBUG Step 1: raw interferogram scans + encoder position vs time
                if debug:
                    fig_raw, axs_raw = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

                    axs_raw[0].plot(enc_t, enc_pos, color='steelblue', linewidth=LW)
                    axs_raw[0].set_xlabel('Time (s)')
                    axs_raw[0].set_ylabel('Encoder position (mm)')
                    axs_raw[0].set_title('Raw encoder position vs time')
                    axs_raw[0].minorticks_on()
                    axs_raw[0].tick_params(which='both', direction='in', top=True, right=True)

                    for x, y in zip(scans_pos, scans_data):
                        axs_raw[1].plot(x, y, alpha=0.6, linewidth=LW)
                    axs_raw[1].set_xlabel('Mirror position (mm)')
                    axs_raw[1].set_ylabel('Amplitude')
                    axs_raw[1].set_title(f'Raw scans overlaid | x={xf[0]} f={xf[1]}')
                    axs_raw[1].minorticks_on()
                    axs_raw[1].tick_params(which='both', direction='in', top=True, right=True)

                    fig_raw.tight_layout()
                    fig_raw.savefig('%s/x%s_f%s_step1_raw.png' % (out_path, xf[0], xf[1]))
                    plt.close(fig_raw)

                # DEBUG Step 2: shifted position showing scan alignment
                if debug:
                    n_scans = len(scans_pos)
                    fig_shift, axs_shift = plt.subplots(n_scans, 1, figsize=(12, 3 * n_scans), sharex=True)
                    if n_scans == 1:
                        axs_shift = [axs_shift]
                    for i, x in enumerate(scans_pos):
                        x_shifted = x - np.max(x)
                        axs_shift[i].plot(x_shifted, alpha=0.8, linewidth=LW, color='steelblue')
                        axs_shift[i].axvline(0, color='black', linestyle='dashed', linewidth=LW, label='ZPD')
                        axs_shift[i].set_ylabel('Position (mm)')
                        axs_shift[i].set_title(f'Scan {i + 1}')
                        axs_shift[i].legend(fontsize=8)
                        axs_shift[i].minorticks_on()
                        axs_shift[i].tick_params(which='both', direction='in', top=True, right=True)
                    axs_shift[-1].set_xlabel('Sample index')
                    fig_shift.suptitle(f'Scan alignment | x={xf[0]} f={xf[1]}')
                    fig_shift.tight_layout()
                    fig_shift.savefig('%s/x%s_f%s_step2_alignment.png' % (out_path, xf[0], xf[1]))
                    plt.close(fig_shift)

                # Bandpass filter + detrend
                scans_pos_good = []
                scans_filtered = []
                detrended = []
                for x, y in zip(scans_pos, scans_data):
                    if len(x) <= 5:
                        msg = f'x={xf[0]} f={xf[1]}: skipping short scan with {len(x)} samples'
                        print(f'\tWarning: {msg}')
                        fail_details.append(f'[WARN] obs={obs_id} MCE={MCE} {msg}')
                        continue
                    raw_step = np.median(np.abs(np.diff(x))) * 1e-3
                    y_filtered = bandpass_filter(y, 150, 360, raw_step)
                    scans_pos_good.append(x)
                    scans_filtered.append(y_filtered)
                    detrended.append(y_filtered - np.polyval(np.polyfit(x, y_filtered, 4), x))

                if len(detrended) == 0:
                    fail_counts['no_scans'] += 1
                    fail_details.append(f'[FAIL:no_scans] obs={obs_id} MCE={MCE} x={xf[0]} f={xf[1]}: all scans too short after filtering')
                    plt.close(figs)
                    continue

                # DEBUG Step 3: raw vs filtered vs detrended
                # Median-subtract each scan so they are co-aligned vertically
                if debug:
                    n_scans = len(scans_pos_good)
                    fig_detrend, axs_detrend = plt.subplots(n_scans, 1, figsize=(12, 3 * n_scans), sharex=True)
                    if n_scans == 1:
                        axs_detrend = [axs_detrend]
                    for i, (x, y_raw, y_filt, y_det) in enumerate(zip(scans_pos_good, scans_data, scans_filtered, detrended)):
                        y_raw_plot  = y_raw  - np.nanmedian(y_raw)
                        y_filt_plot = y_filt - np.nanmedian(y_filt)
                        y_det_plot  = y_det  - np.nanmedian(y_det)
                        axs_detrend[i].plot(x, y_raw_plot,  alpha=0.8, linewidth=LW, color='steelblue', label='Raw (median-sub)')
                        axs_detrend[i].plot(x, y_filt_plot, alpha=0.8, linewidth=LW, color='orange',    label='Filtered (median-sub)')
                        axs_detrend[i].plot(x, y_det_plot,  alpha=0.8, linewidth=LW, color='green',     label='Detrended')
                        axs_detrend[i].set_ylabel('Amplitude')
                        axs_detrend[i].set_title(f'Scan {i + 1}')
                        axs_detrend[i].legend(fontsize=8)
                        axs_detrend[i].minorticks_on()
                        axs_detrend[i].tick_params(which='both', direction='in', top=True, right=True)
                    axs_detrend[-1].set_xlabel('Mirror position (mm)')
                    fig_detrend.suptitle(f'Filter and detrend | x={xf[0]} f={xf[1]}')
                    fig_detrend.tight_layout()
                    fig_detrend.savefig('%s/x%s_f%s_step3_detrend.png' % (out_path, xf[0], xf[1]))
                    plt.close(fig_detrend)

                bin_ys = []
                for x, y in zip(scans_pos_good, detrended):
                    yb, _, _ = binned_statistic(x, y, statistic='mean', bins=bin_edges)
                    bin_ys.append(yb)
                bin_ys = np.array(bin_ys)

                bin_xs = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                avg_bin_ys = np.nanmedian(bin_ys, axis=0)

                max_idx = np.nanargmax(avg_bin_ys)
                avg_bin_xs = bin_xs - bin_xs[max_idx]

                axs[0].plot(avg_bin_xs, avg_bin_ys)
                axs[0].axvline(0, color='black', linestyle='dashed', label='ZPD')
                axs[0].set_xlabel('Binned Position [mm]')
                axs[0].set_ylabel('Averaged Flux [arb]')

                mask = np.isnan(avg_bin_ys)
                if np.any(~mask):
                    interp = interp1d(
                        avg_bin_xs[~mask], avg_bin_ys[~mask],
                        bounds_error=False, fill_value=0
                    )
                    avg_bin_ys[mask] = interp(avg_bin_xs[mask])
                avg_bin_ys[np.isnan(avg_bin_ys)] = 0

                d_raw = avg_bin_ys.copy()

                symm_len = int(len(d_raw) // 2)
                asymm_len = len(d_raw) - symm_len
                final_len = log2ceil(asymm_len * 2)

                igram_asymm_ext = np.zeros(final_len)
                igram_asymm_ext[:asymm_len] = d_raw[symm_len:]

                try:
                    igram_asymm_ext[-symm_len:] = d_raw[:symm_len]
                except ValueError:
                    fail_counts['too_few_points'] += 1
                    fail_details.append(f'[FAIL:too_few_points] obs={obs_id} MCE={MCE} x={xf[0]} f={xf[1]}: ValueError on symmetric padding')
                    plt.close(figs)
                    continue

                igram_asymm_ext[-symm_len:] *= np.linspace(0, 0.5, symm_len)
                igram_asymm_ext[:symm_len] *= np.linspace(0.5, 1, symm_len)

                # DEBUG Step 4: zero-padded interferogram (full view)
                if debug:
                    fig_pad, ax_pad = plt.subplots(figsize=(12, 4))
                    ax_pad.plot(igram_asymm_ext, color='steelblue', linewidth=LW, label='Zero-padded interferogram')
                    ax_pad.axvline(asymm_len, color='red', linestyle='dashed', linewidth=LW, label='End of data')
                    ax_pad.axvline(final_len - symm_len, color='orange', linestyle='dashed', linewidth=LW, label='Start of symmetric part')
                    ax_pad.set_xlabel('Sample index')
                    ax_pad.set_ylabel('Amplitude')
                    ax_pad.set_title(f'Zero-padded interferogram | x={xf[0]} f={xf[1]}')
                    ax_pad.legend(fontsize=8)
                    ax_pad.minorticks_on()
                    ax_pad.tick_params(which='both', direction='in', top=True, right=True)
                    fig_pad.tight_layout()
                    fig_pad.savefig('%s/x%s_f%s_step4_zeropad.png' % (out_path, xf[0], xf[1]))
                    plt.close(fig_pad)

                # DEBUG Step 4b: ZPD zoom
                if debug:
                    zpd_zoom_half = 200  # samples either side of ZPD to show
                    zpd_idx = 0  # ZPD sits at index 0 after asymmetric padding
                    zoom_start = max(0, zpd_idx)
                    zoom_end   = min(final_len, zpd_idx + zpd_zoom_half)
                    # also look at the end of the array where the symmetric part lives
                    fig_zpd, axs_zpd = plt.subplots(1, 2, figsize=(12, 4))

                    # left: start of array (asymmetric side, contains ZPD)
                    axs_zpd[0].plot(np.arange(zoom_start, zoom_end),
                                    igram_asymm_ext[zoom_start:zoom_end],
                                    color='steelblue', linewidth=LW)
                    axs_zpd[0].axvline(zpd_idx, color='black', linestyle='dashed', linewidth=LW, label='ZPD (index 0)')
                    axs_zpd[0].set_xlabel('Sample index')
                    axs_zpd[0].set_ylabel('Amplitude')
                    axs_zpd[0].set_title('ZPD region (start of array)')
                    axs_zpd[0].legend(fontsize=8)
                    axs_zpd[0].minorticks_on()
                    axs_zpd[0].tick_params(which='both', direction='in', top=True, right=True)

                    # right: end of array (symmetric side)
                    sym_start = max(0, final_len - zpd_zoom_half)
                    axs_zpd[1].plot(np.arange(sym_start, final_len),
                                    igram_asymm_ext[sym_start:final_len],
                                    color='steelblue', linewidth=LW)
                    axs_zpd[1].axvline(final_len - symm_len, color='orange', linestyle='dashed', linewidth=LW, label='Start of symmetric part')
                    axs_zpd[1].set_xlabel('Sample index')
                    axs_zpd[1].set_ylabel('Amplitude')
                    axs_zpd[1].set_title('Symmetric region (end of array)')
                    axs_zpd[1].legend(fontsize=8)
                    axs_zpd[1].minorticks_on()
                    axs_zpd[1].tick_params(which='both', direction='in', top=True, right=True)

                    fig_zpd.suptitle(f'ZPD zoom | x={xf[0]} f={xf[1]}')
                    fig_zpd.tight_layout()
                    fig_zpd.savefig('%s/x%s_f%s_step4b_zpd_zoom.png' % (out_path, xf[0], xf[1]))
                    plt.close(fig_zpd)

                # Step 5: apply window function
                window = np.fft.ifftshift(np.blackman(final_len))
                igram_asymm_ext *= window

                # DEBUG Step 5: window alignment check
                if debug:
                    fig_win, ax_win = plt.subplots(figsize=(12, 4))
                    ax_win_twin = ax_win.twinx()
                    ax_win.plot(igram_asymm_ext, label='Windowed interferogram', color='steelblue', linewidth=LW)
                    ax_win_twin.plot(np.fft.ifftshift(np.blackman(final_len)), label='Window function', color='orange', alpha=0.7, linewidth=LW)
                    ax_win.set_ylabel('Interferogram amplitude', color='steelblue')
                    ax_win_twin.set_ylabel('Window amplitude', color='orange')
                    ax_win.set_xlabel('Sample index')
                    ax_win.minorticks_on()
                    ax_win.tick_params(which='both', direction='in', top=True, right=True)
                    fig_win.suptitle(f'Window alignment check | x={xf[0]} f={xf[1]}')
                    fig_win.legend(loc='upper right')
                    fig_win.tight_layout()
                    fig_win.savefig('%s/x%s_f%s_step5_window.png' % (out_path, xf[0], xf[1]))
                    plt.close(fig_win)

                fft_vals = np.fft.fftshift(np.fft.fft(igram_asymm_ext))
                freq = np.fft.fftshift(np.fft.fftfreq(final_len, d=mmstep)) * 3e8 / 1e9

                cs_coadd = np.abs(fft_vals) ** 2

                mask_all = (freq > FMIN) & (freq < FMAX)
                if not np.any(mask_all):
                    plt.close(figs)
                    continue

                cs_smooth = uniform_filter1d(cs_coadd[mask_all], size=20)
                noise_std = np.std(cs_coadd[mask_all])
                iii = np.argmax(cs_smooth)
                peak_height = cs_coadd[mask_all][iii]

                if peak_height < 5 * noise_std:
                    fail_counts['low_snr'] += 1
                    fail_details.append(f'[FAIL:low_snr] obs={obs_id} MCE={MCE} x={xf[0]} f={xf[1]}: peak {peak_height:.2f} < 5*noise {5*noise_std:.2f}')
                    plt.close(figs)
                    continue

                spec_peak_val = freq[mask_all][iii]
                mask_tight = (freq > spec_peak_val - FRANGE) & (freq < spec_peak_val + FRANGE)

                if np.sum(mask_tight) < 5:
                    fail_counts['too_few_points'] += 1
                    fail_details.append(f'[FAIL:too_few_points] obs={obs_id} MCE={MCE} x={xf[0]} f={xf[1]}: only {np.sum(mask_tight)} points in tight mask')
                    plt.close(figs)
                    continue

                cs_coadd_raw = cs_coadd.copy()

                mask_phase = mask_tight & (cs_coadd > 0.1 * peak_height)
                if np.any(mask_phase):
                    phase_orig = np.unwrap(np.angle(fft_vals))
                    p = np.polyfit(freq[mask_phase], phase_orig[mask_phase], deg=1)
                    phase_corr = np.polyval(p, freq)
                    fft_vals *= np.exp(-1j * phase_corr)
                    cs_coadd = np.abs(fft_vals) ** 2
                    peak_height = np.max(cs_coadd[mask_tight])

                # DEBUG Step 6: raw vs phase corrected spectrum + phase fit
                if debug:
                    fig_phase, axs_phase = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

                    axs_phase[0].plot(freq[mask_all], cs_coadd_raw[mask_all], color='steelblue',
                                      linewidth=LW, label='Before phase correction')
                    axs_phase[0].plot(freq[mask_all], cs_coadd[mask_all], color='orange',
                                      linewidth=LW, label='After phase correction')
                    axs_phase[0].axvline(spec_peak_val, color='black', linestyle='dashed',
                                         linewidth=LW, label=f'Peak = {spec_peak_val:.2f} GHz')
                    axs_phase[0].set_ylabel('Power')
                    axs_phase[0].set_title(f'Phase correction | x={xf[0]} f={xf[1]}')
                    axs_phase[0].legend(fontsize=8)
                    axs_phase[0].minorticks_on()
                    axs_phase[0].tick_params(which='both', direction='in', top=True, right=True)

                    if np.any(mask_phase):
                        axs_phase[1].plot(freq[mask_phase], phase_orig[mask_phase], color='steelblue',
                                          linewidth=LW, label='Unwrapped phase')
                        axs_phase[1].plot(freq[mask_phase], np.polyval(p, freq[mask_phase]), color='red',
                                          linewidth=LW, linestyle='dashed', label='Linear fit')
                    axs_phase[1].set_xlabel('Frequency (GHz)')
                    axs_phase[1].set_ylabel('Phase (rad)')
                    axs_phase[1].set_title('Phase fit')
                    axs_phase[1].legend(fontsize=8)
                    axs_phase[1].minorticks_on()
                    axs_phase[1].tick_params(which='both', direction='in', top=True, right=True)

                    fig_phase.tight_layout()
                    fig_phase.savefig('%s/x%s_f%s_step6_phase.png' % (out_path, xf[0], xf[1]))
                    plt.close(fig_phase)

                df_freq = freq[1] - freq[0]
                p0 = [df_freq, spec_peak_val, peak_height]
                try:
                    popt, _ = curve_fit(
                        gaussian,
                        freq[mask_tight],
                        cs_coadd[mask_tight],
                        p0=p0,
                        maxfev=1000
                    )
                except RuntimeError:
                    fail_counts['fit_failed'] += 1
                    fail_details.append(f'[FAIL:fit_failed] obs={obs_id} MCE={MCE} x={xf[0]} f={xf[1]}: curve_fit did not converge')
                    plt.close(figs)
                    continue

                fit_sigma, fit_f0, fit_amp = popt
                spec_f0_val = fit_f0
                spec_fwhm_val = gaussian_fwhm(fit_sigma)

                if not (FMIN < spec_f0_val < FMAX):
                    fail_counts['bad_frequency'] += 1
                    fail_details.append(f'[FAIL:bad_frequency] obs={obs_id} MCE={MCE} x={xf[0]} f={xf[1]}: fitted center {spec_f0_val:.2f} outside [{FMIN},{FMAX}]')
                    plt.close(figs)
                    continue

                mask_peak = (freq < fit_f0 - 3 * fit_sigma) | (freq > fit_f0 + 3 * fit_sigma)
                residual = cs_coadd[mask_peak]
                snr = fit_amp / np.std(residual)

                if snr < SNR:
                    fail_counts['low_snr'] += 1
                    fail_details.append(f'[FAIL:low_snr] obs={obs_id} MCE={MCE} x={xf[0]} f={xf[1]}: SNR={snr:.1f} < {SNR}')
                    plt.close(figs)
                    continue

                if not (expected_spec_f0[xf[1]] - 10) <= spec_f0_val <= (expected_spec_f0[xf[1]] + 10):
                    fail_counts['bad_frequency'] += 1
                    fail_details.append(f'[FAIL:bad_frequency] obs={obs_id} MCE={MCE} x={xf[0]} f={xf[1]}: center {spec_f0_val:.2f} not within 10 GHz of expected {expected_spec_f0[xf[1]]:.2f}')
                    plt.close(figs)
                    continue

                if spec_fwhm_val > 12:
                    fail_counts['bad_fwhm'] += 1
                    fail_details.append(f'[FAIL:bad_fwhm] obs={obs_id} MCE={MCE} x={xf[0]} f={xf[1]}: FWHM={spec_fwhm_val:.2f} > 12')
                    plt.close(figs)
                    continue

                mask_res = (freq >= FMIN) & (freq <= FMAX)
                freq_plot = freq[mask_res]
                spec_plot = cs_coadd[mask_res]

                axs[1].plot(freq_plot, spec_plot)
                axs[1].set_xlabel('Frequency [GHz]')
                axs[1].set_ylabel('Power [arb]')
                axs[1].set_xlim(FMIN, FMAX)
                figs.tight_layout()
                figs.savefig('%s/x%s_f%s_bandpass.png' % (out_path, xf[0], xf[1]))
                plt.close(figs)

                # DEBUG Step 7: Gaussian fit overlaid on final spectrum
                if debug:
                    fig_fit, ax_fit = plt.subplots(figsize=(10, 4))
                    ax_fit.plot(freq_plot, spec_plot, color='steelblue', linewidth=LW, label='Spectrum')
                    ax_fit.plot(freq[mask_tight], gaussian(freq[mask_tight], *popt), color='red',
                                linewidth=LW, linestyle='dashed',
                                label=f'Gaussian fit\ncenter={fit_f0:.2f} GHz\nFWHM={spec_fwhm_val:.2f} GHz\nSNR={snr:.1f}')
                    ax_fit.axvline(fit_f0, color='black', linestyle='dashed', linewidth=LW)
                    ax_fit.axvline(expected_spec_f0[xf[1]], color='green', linestyle='dashed',
                                   linewidth=LW, label=f'Expected center={expected_spec_f0[xf[1]]:.2f} GHz')
                    ax_fit.set_xlabel('Frequency (GHz)')
                    ax_fit.set_ylabel('Power')
                    ax_fit.set_title(f'Gaussian fit | x={xf[0]} f={xf[1]}')
                    ax_fit.legend(fontsize=8)
                    ax_fit.minorticks_on()
                    ax_fit.tick_params(which='both', direction='in', top=True, right=True)
                    fig_fit.tight_layout()
                    fig_fit.savefig('%s/x%s_f%s_step7_fit.png' % (out_path, xf[0], xf[1]))
                    plt.close(fig_fit)

                fit_results.append({
                    'xf0': row,
                    'xf1': xf[1],
                    'center': spec_f0_val,
                    'expected_center': expected_spec_f0[xf[1]],
                    'fwhm': spec_fwhm_val,
                })

                norm = np.nanmax(cs_coadd)
                if not np.isfinite(norm) or norm == 0:
                    fail_counts['bad_frequency'] += 1
                    fail_details.append(f'[FAIL:bad_frequency] obs={obs_id} MCE={MCE} x={xf[0]} f={xf[1]}: norm is zero or non-finite')
                    continue

                ax.plot(freq, cs_coadd / norm, label=f'xf1={xf[1]}')
                bpass_axs.plot(freq, cs_coadd / norm)

            ax.set_xlim(FMIN, FMAX)
            ax.set_title(f'Bandpasses for xf[0] = {row}')
            fig.tight_layout()
            fig.savefig(os.path.join(out_path, f'xf_{row}_bandpasses.png'))
            plt.close(fig)

        bpass_axs.set_xlabel('Frequency [GHz]')
        bpass_axs.set_ylabel('Normalized Power')
        bpass_fig.savefig('%s/Curated_Band_passes.png' % out_path)
        plt.close(bpass_fig)
        print("Finished generating bandpass figures.")

        df_out = pd.DataFrame(fit_results)
        csv_path = os.path.join(out_path, "bandstats.csv")
        df_out.to_csv(csv_path, index=False)
        print("Finished saving band stats.")

        # Print and save failure breakdown
        total = sum(fail_counts.values())
        print(f"Obs_ID: {obs_id} MCE={MCE}  |  Total failures: {total}")
        log_lines = [f"\n{'='*60}\nObs_ID: {obs_id} MCE={MCE}  |  Total failures: {total}\n"]
        for reason, count in fail_counts.items():
            if count > 0:
                print(f"\t{reason}: {count}")
                log_lines.append(f"\t{reason}: {count}\n")

        # Per-detector detail lines
        if fail_details:
            log_lines.append('\nPer-detector details:\n')
            for line in fail_details:
                log_lines.append(f'  {line}\n')

        obs_log_path = os.path.join(out_path, "failure_log.txt")
        with open(obs_log_path, 'w') as f:
            f.writelines(log_lines)
        print(f"Failure log saved to {obs_log_path}")

        # Also append to the global run log
        with open(run_log_path, 'a') as f:
            f.writelines(log_lines)


def bin_bandstats(data_paths, f0_tolerance, fwhm_tolerance):
    center_dict = defaultdict(list)
    fwhm_dict = defaultdict(list)
    for path in data_paths:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (row['xf0'], row['xf1'])
            if ((row['center'] <= row['expected_center'] + f0_tolerance) and
                    (row['center'] >= row['expected_center'] - f0_tolerance)):
                if row['fwhm'] <= fwhm_tolerance:
                    center_dict[key].append(row['center'])
                    fwhm_dict[key].append(row['fwhm'])
    return center_dict, fwhm_dict


def dict_to_df(center_dict, fwhm_dict):
    rows = []
    keys = set(center_dict.keys())
    for (xf0, xf1) in keys:
        centers = center_dict[(xf0, xf1)]
        fwhms = fwhm_dict[(xf0, xf1)]
        center_mean = np.mean(centers)
        center_err = np.std(centers) / np.sqrt(len(centers))
        fwhm_quad = np.sqrt(np.mean(np.square(fwhms)))
        rows.append({
            'xf0': xf0,
            'xf1': xf1,
            'center': center_mean,
            'center_err': center_err,
            'fwhm': fwhm_quad
        })
    return pd.DataFrame(rows)


def df_mapping(filepath, f0_lim, df_lim):
    center, fwhm = bin_bandstats(filepath, f0_lim, df_lim)
    df = dict_to_df(center, fwhm)
    df['expected_center'] = df['xf1'].astype(int).map(lambda x: expected_spec_f0[x])
    return df


if __name__ == '__main__':

    # -------------------------------------------------------
    # Observation list: (obs_id, encoder_run_id)
    # Entries with no obs_id or known bad data are commented out
    # -------------------------------------------------------
    base_data_path = '/data/time/2025_data/time-missy'
    base_encoder_path = '/data/time/2025_data/time-missy/encoder_npy_files'

    obs_list = [
        ('1772474133', 'run_20260302_175522'),   # testing 2x speed
        ('1772407333', 'run_20260301_232157'),   # high bias long scan
        # ('1772412258', 'run_20260301_004304'), # med bias — no sync data
        ('1772418469', 'run_20260302_022744'),   # low bias long scan
        ('1772469294', 'run_20260302_163536'),   # med bias take 2
        ('1772484062', 'run_20260302_204132'),   # med bias again
        ('1772494384', 'run_20260302_233300'),   # low bias again
        ('1772497449', 'run_20260303_002407'),   # low bias short scan
        # ('1772501770', None),                  # noise — no encoder
        ('1772547051', 'run_20260303_140942'),   # bias 2000
        ('1772550458', 'run_20260303_150533'),   # bias 1750
        ('1772554146', 'run_20260303_160821'),   # bias 1500
        ('1772557975', 'run_20260303_170958'),   # bias 1500, 1.2 mm/s
        ('1772564862', 'run_20260303_190614'),   # bias 1250
        ('1772567791', 'run_20260303_195458'),   # bias 1000
        ('1772571200', 'run_20260303_205241'),   # bias 900
        ('1772573571', 'run_20260303_213220'),   # bias 800
        ('1772577226', 'run_20260303_223206'),   # bias 700
        ('1772583659', 'run_20260304_001934'),   # bias 600
        ('1772587232', 'run_20260304_011910'),   # bias 500
        ('1772633076', 'run_20260304_140355'),   # bias 1750, 1.2 mm/s
        ('1772638487', 'run_20260304_153348'),   # bias 1000, 1.2 mm/s
        ('1772651942', 'run_20260304_180255'),   # bias 400
        ('1772656093', 'run_20260304_202704'),   # bias 300 redo
        ('1772659774', 'run_20260304_212853'),   # bias 200
        ('1772665933', 'run_20260304_231010'),   # bias 200 after check rms
        ('1772668050', 'run_20260304_234546'),   # bias 200, 1.2 mm/s
        ('1772710021', 'run_20260305_112519'),   # medium bias
        ('1772718343', 'run_20260305_134521'),   # medium low
        ('1772726042', 'run_20260305_155332'),   # low
    ]

    run_log = 'run_failures.log'
    # Clear the run log at the start of each full run
    with open(run_log, 'w') as f:
        f.write(f'FTS passband run log\n{"="*60}\n')

    for MCE in [0, 1]:
        print(f'\n{"="*60}')
        print(f'Processing MCE{MCE} (Spec {"A" if MCE == 0 else "B"})')
        print(f'{"="*60}')
        for obs_id, encoder_run_id in obs_list:
            file_path = f'{base_data_path}/{obs_id}'
            encoder_path = f'{base_encoder_path}/{encoder_run_id}/'
            print(f'\nProcessing obs_id={obs_id}, MCE={MCE}')
            try:
                passband_compute(
                    MCE,
                    [obs_id],
                    [file_path],
                    [encoder_path],
                    OPD_FACTOR=2,
                    debug=True,
                    run_log_path=run_log
                )
            except Exception as e:
                msg = f'[EXCEPTION] obs={obs_id} MCE={MCE}: {type(e).__name__}: {e}\n{traceback.format_exc()}'
                print(f'\tFAILED: {msg}')
                with open(run_log, 'a') as f:
                    f.write(f'\n{msg}\n')
                continue