from timesoft.timestream.timestream_tools import Timestream 
import numpy as np
import matplotlib.pyplot as plt 
from glob import glob
import os 
import pandas as pd


from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
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
SNR = 4

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

def passband_compute(MCE, obs_ids, file_paths, encoder_paths, V_MM):
    mmstep = 0.05e-3 * V_MM 
    
    for obs_id, path, encoder_path in zip(obs_ids, file_paths, encoder_paths):
        ts = Timestream(path, mc = MCE, version='2024.dev.1', skip_last = False, mode = 'raw')

        path = str(obs_id) + 'mc_%s' % MCE
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
    
        xd = np.vstack([np.load(a) for a in sorted(glob(encoder_path+'*.npy'),key=lambda x:float(x[len(encoder_path)+8:-4]))]).T
        enc_t = xd[0]; pos = xd[1]
        interp_func = interp1d(enc_t, pos)
        new_pos = interp_func(ts.t[ts.t < enc_t.max()]) 

        print(f"Completed loading data from {obs_id} : MCE{MCE}.")

        xf0_unique = sorted(set([xf[0] for xf in ts.header['xf_coords']]))

        fit_results = []
        fail_count  = 0  
    
        bpass_fig, bpass_axs = plt.subplots(1, figsize=(18,6))
        bpass_axs.set_xlim(180,330)

        for row in xf0_unique:
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))
            ax.set_xlabel('Frequency [GHz]')
            ax.set_ylabel('Normalized Amplitude')

            xfs_row = [xf for xf in ts.header['xf_coords'] if xf[0] == row]

            for xf in xfs_row:
                figs, axs = plt.subplots(2)

                idx = ts.get_xf(xf[0], xf[1])

                new_pos_shifted = new_pos - np.nanmax(new_pos)
                ts_first_cut = ts.data[idx][ts.t < enc_t.max()]

                good = (new_pos_shifted > -995) & (new_pos_shifted < 0)
                pos = new_pos_shifted[good]
                inds = np.arange(len(ts_first_cut))[good]

                low_mask = np.isclose(pos, -995, atol=0.5)
                high_mask = np.isclose(pos, -5, atol=0.5)
                low_inds = prune(inds[low_mask])
                high_inds = prune(inds[high_mask])

                scans_data, scans_pos = [], []
                for lo, hi in zip(low_inds, high_inds):
                    if lo < hi:
                        scans_data.append(ts_first_cut[lo:hi])
                        scans_pos.append(new_pos_shifted[lo:hi])

                if len(scans_data) == 0:
                    plt.close(figs)
                    continue
                
                detrended = [
                    y - np.polyval(np.polyfit(x, y, 4), x)
                    for x, y in zip(scans_pos, scans_data)
                    if len(x) > 5
                ]

                bin_ys = []
                for x, y in zip(scans_pos, detrended):
                    yb, _, _ = binned_statistic(x, y, statistic='mean', bins=bin_edges)
                    bin_ys.append(yb)
                bin_ys = np.array(bin_ys)

                """bin_drift = []
                for x, y in zip(scans_pos, scans_data):
                    yb, _, _ = binned_statistic(x, y, statistic='mean', bins=bin_edges)
                    bin_drift.append(yb)
                bin_drift = np.array(bin_drift)
                avg_bin_drift = np.nanmedian(bin_drift, axis = 0)"""

                bin_xs = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                avg_bin_ys = np.nanmedian(bin_ys, axis=0)

                max_idx = np.nanargmax(avg_bin_ys)
                avg_bin_xs = bin_xs - bin_xs[max_idx]

                """fg, axx = plt.subplots(1, figsize=(8,5))
                axx.plot(avg_bin_xs, avg_bin_drift)
                axx.axvline(0, color='black', linestyle='dashed', label='ZPD')
                axx.set_xlabel('Binned Position [mm]')
                axx.set_ylabel('Averaged Flux [arb]')
                fg.tight_layout()
                fg.legend()
                fg.savefig('%s/x%s_f%s_raw' % (path,xf[0],xf[1]))
                plt.close(fg)"""

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
                    fail_count += 1
                    plt.close(figs)
                    continue

                igram_asymm_ext[-symm_len:] *= np.linspace(0, 0.5, symm_len)
                igram_asymm_ext[:symm_len] *= np.linspace(0.5, 1, symm_len)

                window = np.fft.ifftshift(np.blackman(final_len))
                igram_asymm_ext *= window

                fft_vals = np.fft.fftshift(np.fft.fft(igram_asymm_ext))
                freq = np.fft.fftshift(np.fft.fftfreq(final_len, d=mmstep)) * 3e8 / 1e9

                cs_coadd = np.abs(fft_vals)

                mask_all = (freq > FMIN) & (freq < FMAX)
                if not np.any(mask_all):
                    plt.close(figs)
                    continue
                
                iii = np.argmax(cs_coadd[mask_all])
                spec_peak_val = freq[mask_all][iii]
                peak_height = cs_coadd[mask_all][iii]

                mask_tight = (freq > spec_peak_val - FRANGE) & (freq < spec_peak_val + FRANGE)
                if np.sum(mask_tight) < 5:
                    plt.close(figs)
                    continue

                mask_phase = mask_tight & (cs_coadd > 0.1 * peak_height)
                if np.any(mask_phase):
                    phase_orig = np.unwrap(np.angle(fft_vals))     
                    p = np.polyfit(freq[mask_phase], phase_orig[mask_phase], deg=1)
                    phase_corr = np.polyval(p, freq)
                    fft_vals *= np.exp(-1j * phase_corr)
                    cs_coadd = np.abs(fft_vals)                  
                    peak_height = np.max(cs_coadd[mask_tight])

                df = freq[1] - freq[0]
                p0 = [df, spec_peak_val, peak_height]
                try:
                    popt, _ = curve_fit(
                        gaussian,
                        freq[mask_tight],
                        cs_coadd[mask_tight],
                        p0=p0,
                        maxfev=1000
                    )
                except RuntimeError:
                    fail_count += 1
                    plt.close(figs)
                    continue

                fit_sigma, fit_f0, fit_amp = popt
                spec_f0_val = fit_f0
                spec_fwhm_val = gaussian_fwhm(fit_sigma)

                if not (FMIN < spec_f0_val < FMAX):
                    fail_count += 1
                    plt.close(figs)
                    continue

                mask_peak = (freq < fit_f0 - 3 * fit_sigma) | (freq > fit_f0 + 3 * fit_sigma)
                residual = cs_coadd[mask_peak]
                snr = fit_amp / np.std(residual)

                if snr < SNR:
                    fail_count += 1
                    plt.close(figs)
                    continue  
            
                if not (expected_spec_f0[xf[1]] - 10) <= spec_f0_val <= (expected_spec_f0[xf[1]] + 10):
                    fail_count += 1
                    plt.close(figs)
                    continue

                if (spec_fwhm_val > 12):
                    fail_count += 1
                    plt.close(figs)
                    continue

                mask_res = (freq >= FMIN) & (freq <= FMAX)
                freq_plot = freq[mask_res]
                spec_plot = cs_coadd[mask_res]

                axs[1].plot(freq_plot, spec_plot)
                axs[1].set_xlabel('Frequency [GHz]')
                axs[1].set_ylabel('FFT [arb]')
                axs[1].set_xlim(FMIN, FMAX)
                """axs[1].set_xlim(180, fit_f0 + 10)"""
                figs.tight_layout()
                figs.savefig('%s/x%s_f%s_bandpass' % (path,xf[0],xf[1]))
                plt.close(figs)

                fit_results.append({
                    'xf0': row,
                    'xf1': xf[1],
                    'center': spec_f0_val,
                    'expected_center': expected_spec_f0[xf[1]],
                    'fwhm': spec_fwhm_val,
                })

                norm = np.nanmax(cs_coadd)

                if not np.isfinite(norm) or norm == 0:
                    fail_count += 1
                    continue
            
                ax.plot(freq, cs_coadd / norm, label=f'xf1={xf[1]}')
                bpass_axs.plot(freq, cs_coadd / norm)

            ax.set_xlim(FMIN, FMAX)
            ax.set_title(f'Bandpasses for xf[0] = {row}')
            fig.tight_layout()
            fig.savefig(os.path.join(path, f'xf_{row}_bandpasses.png'))
            plt.close(fig)
    
        bpass_axs.set_xlabel('Frequency [GHz]')
        bpass_axs.set_ylabel('Normalized Amplitude')
        bpass_fig.savefig('%s/Curated_Band_passes' % path)
        plt.close(bpass_fig)
        print("Finished generating bandpass figures.")

        df_out = pd.DataFrame(fit_results)
        csv_path = os.path.join(path, "bandstats.csv")
        df_out.to_csv(csv_path, index=False)
        print("Finished saving band stats.")

        print(f"Obs_ID: {obs_id}  |  Failures: {fail_count}")

def bin_bandstats(data_paths, f0_tolerance, fwhm_tolerance):
    center_dict = defaultdict(list)
    fwhm_dict = defaultdict(list)
    for path in data_paths:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (row['xf0'], row['xf1'])
            if ((row['center'] <= row['expected_center'] + f0_tolerance) and (row['center'] >= row['expected_center'] - f0_tolerance)):
                if (row['fwhm'] <= fwhm_tolerance):
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