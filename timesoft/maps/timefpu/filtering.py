from __future__ import division, print_function, unicode_literals, absolute_import

import logging
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal

from timefpu.utils import num_suffix

try:
	from scipy.signal import find_peaks
except ImportError:
	print("You need a newer version of scipy to use this library (>=1.1.0)")
	exit()
	
# Subtract a polynomial of order k from a single timestream
def polysub(data, k=4):
	
	t = list(range(len(data)))
	func = np.poly1d(np.polyfit(t, data, k))
	
	return (data - func(t))
	
# Save the filter response plot
def plot_filter(filt_order, filt_omega, filt_h, filt_f_cutoff, f_nyq, title=None, fname_out=None):
	fig = plt.figure()
	if title is None:
		plt.title(str(filt_order) + num_suffix(filt_order) + ' Order Filter')
	else:
		plt.title(title)
	ax1 = fig.add_subplot(111)
	plt.plot(filt_omega*f_nyq/np.pi, 20 * np.log10(abs(filt_h)), 'b', label='Transfer (Magnitude)')
	plt.ylabel('Amplitude [dB]')
	plt.xlabel('Frequency [Hz]')
	ax2 = ax1.twinx()
	plt.plot(filt_omega[30:]*f_nyq/np.pi, np.unwrap(np.angle(filt_h))[30:], 'r', label='Transfer (Phase)')
	plt.ylabel('Phase [radians]')
	ax1.grid()
	ax1.set_ylim([-40,3])
	ax1.set_xlim([0, filt_f_cutoff*2.1])
	plt.axvline(filt_f_cutoff, color='k', ls='dashed')
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2, loc="lower right", framealpha=1)
	if fname_out:
		plt.savefig(fname_out)
		plt.close(fig)
	else:
		plt.show()

# Apply a highpass filter to a timesteam d.  Cuttoff is at f_cut.
def highpass(d, f_cut, f_sample=1, plot=False):
	return _highlowfilt_helper('highpass', d, f_cut, f_sample, plot)

# Apply a lowpass filter to a timesteam d.  Cuttoff is at f_cut.
def lowpass(d, f_cut, f_sample=1, plot=False):
	return _highlowfilt_helper('lowpass', d, f_cut, f_sample, plot)
	
# Helper function implementing lowpass() and highpass()
_filters_highlowpass = {}
def _highlowfilt_helper(ftype, d, f_cut, f_sample=1, plot=False):
	
	if len(d) < 1:
		raise ValueError("No data provided")
		
	f_nyq = f_sample / 2
	f_cut_rel = f_cut/f_nyq
	
	if f_cut_rel >= 1:
		raise ValueError("Filter band includes the nyquist frequency.")
		
	if f_cut_rel < 0.0002:
		raise ValueError("Fractional cutoff frequency too low, filter would be unstable")
	
	lookup_key = (ftype,f_cut_rel)
	hlpf = _filters_highlowpass.get(lookup_key, None)
	
	# Create the filter if we have never used it
	if hlpf is None:
		filt_loss_pass = 0.1 #dB
		filt_loss_stop = 40 #dB
		filt_norm_pass = [f_cut_rel]
		filt_norm_stop = [2*f_cut_rel]
		(filt_order, Wn) = scipy.signal.buttord(wp=filt_norm_pass, ws=filt_norm_stop, gpass=filt_loss_pass, gstop=filt_loss_stop, analog=False)
		filt_order = min(filt_order, 8)
		if f_cut_rel < 0.002:
			filt_order = min(filt_order, 4)
		elif f_cut_rel < 0.02:
			filt_order = min(filt_order, 6)
		logging.debug("Building " + str(filt_order) + num_suffix(filt_order) + " order Butterworth " + ftype + " filter")
		filt_b, filt_a = scipy.signal.butter(filt_order, Wn, btype=ftype, analog=False)
		_filters_highlowpass[lookup_key] = (filt_b, filt_a, filt_order)
	else:
		filt_b, filt_a, filt_order = hlpf
		
	if plot:
		filt_omega, filt_h = scipy.signal.freqz(filt_b, filt_a, worN=2048*16) # Filter response
		plot_filter(filt_order, filt_omega, filt_h, f_cut, f_nyq)
	
	return scipy.signal.filtfilt(filt_b, filt_a, d, padlen=len(d)//20, padtype='even')

# Apply a bandpass filter to a timesteam d.  Centered on f_center,
# with a fractional bandwidth of frac_bw.
def bandpass(d, f_center, frac_bw, f_sample=1, plot=False):
	return _bandfilt_helper('bandpass', d, f_center, frac_bw, f_sample, plot)
	
# Apply a bandstop filter to a timesteam d.  Centered on f_center,
# with a fractional bandwidth of frac_bw.
def bandstop(d, f_center, frac_bw, f_sample=1, plot=False):
	return _bandfilt_helper('bandstop', d, f_center, frac_bw, f_sample, plot)

# Helper function implementing bandpass() and bandstop()
_filters_band = {}
def _bandfilt_helper(ftype, d, f_center, frac_bw, f_sample, plot):
	
	if len(d) < 1:
		raise ValueError("No data provided")
		
	f_nyq = f_sample / 2
	df = frac_bw * f_center
	f_min = f_center - df/2.0
	f_max = f_center + df/2.0

	f_center_rel = f_center/f_nyq
	f_min_rel = f_min/f_nyq
	f_max_rel = f_max/f_nyq
	
	if f_min_rel >= 1 or f_max_rel >= 1:
		raise ValueError("The " + ftype + " range includes the nyquist frequency.")
	
	if not f_min_rel > 0:
		raise ValueError("Frequency minimum extends below 0. Should this be a low/high pass?")
	
	if f_center_rel < 0.002:
		raise ValueError("Center frequency too low, filter would be unstable")
	
	lookup_key = (ftype,f_center_rel,frac_bw)
	filt = _filters_band.get(lookup_key, None)
	
	# Create the filter if we have never used it
	if filt is None:
		filt_loss_pass = 0.1 #dB
		filt_loss_stop = 40 #dB
		filt_norm_pass = [f_min_rel, f_max_rel]
		filt_norm_stop = [filt_norm_pass[0]/2, filt_norm_pass[1]*2]	
		(filt_order, Wn) = scipy.signal.buttord(wp=filt_norm_pass, ws=filt_norm_stop, gpass=filt_loss_pass, gstop=filt_loss_stop, analog=False)
		filt_order = min(filt_order, 4)
		if f_center_rel < 0.02:
			filt_order = min(filt_order, 2)
		elif f_center_rel < 0.2:
			filt_order = min(filt_order, 3)
		logging.debug("Building " + str(filt_order) + num_suffix(filt_order) + " order Butterworth " + str(ftype) + " filter")
		filt_b, filt_a = scipy.signal.butter(filt_order, Wn, btype=ftype, analog=False)
		_filters_band[lookup_key] = (filt_b, filt_a, filt_order)
	else:
		filt_b, filt_a, filt_order = filt
	
	if plot:
		filt_omega, filt_h = scipy.signal.freqz(filt_b, filt_a, worN=2048*16*8) # Filter response
		plot_filter(filt_order, filt_omega, filt_h, f_center, f_nyq)
		
	return scipy.signal.filtfilt(filt_b, filt_a, d, padlen=len(d)//20, padtype='even')
		
# Remove narrow glitches from a timestream.  Removes at most max_glitches.
# The parameters min_spacing and max_width descibe glitch spacing and width
# and are in units of samples. min_height is the height in whatever units
# x is in.  Any peaks with a height less than max_frac_height times the
# tallest peak are ignored.
def deglitch(x, return_count=False, max_glitches=10, max_frac_height=0.2, min_height=None, min_spacing=30, max_width=5, flatten=True):
	
	if flatten:
		# Subtract a polynomial to flatten the data, then re-add it at the end
		timeproxy = list(range(len(x)))
		polyfunc = np.poly1d(np.polyfit(timeproxy, x, deg=4))
		x = x - polyfunc(timeproxy)

	if min_height is None:
		min_height = 7 * np.std(x)
	
	peaks, props = find_peaks(abs(x-np.mean(x)), height=min_height, distance=min_spacing, width=(0,max_width))
	
	if flatten:
		# Re-add the low frequency drift we subtracted earlier
		x = x + polyfunc(timeproxy)
		
	if len(peaks) < 1:
		if return_count:
			return x, 0
		else:
			return x
	
	# Sort by height
	mask = np.argsort(props['peak_heights'])[::-1]
	heights = props['peak_heights'][mask]
	widths = props['widths'][mask]
	peaks = peaks[mask]
	
	# Limit the dynamic range over which peaks are detected
	peaks = peaks[heights > max_frac_height * heights[0]]
	
	# Snip out the peaks
	n_remove = min(len(peaks), max_glitches)
	for i in range(n_remove):
		p = peaks[i]
		fwhm = max(1, int(np.ceil(widths[i])))
		imin = max(0, p-2*fwhm)
		imax = min(len(x), p+2*fwhm+1)
		x[imin:imax] *= np.nan
	
	# Use a linear interpolation to remove any NaNs 
	sample = np.arange(len(x))
	mask = np.isfinite(x)
	x[~mask] = np.interp(sample[~mask], sample[mask], x[mask])
	
	if n_remove > 0:
		logging.info("Glitches removed: %i" % n_remove)
	
	if return_count:
		return x, n_remove
	else:
		return x

if __name__ == '__main__':
	
	print("Filter testing")
	
	f_sample = 1000 # Hz
	x = np.random.normal(size=int(2**21))
	
	def plot_psd(label, data):
		f, Pxx = scipy.signal.welch(data, fs=f_sample, nperseg=int(2**14))
		plt.loglog(f, Pxx, label=label)
	
	plt.figure(0)
	plot_psd('Raw', x)
	for f_cut in [1, 10, 100]:
		plot_psd("Lowpass %g Hz" % f_cut, lowpass(x, f_cut, f_sample))
		plot_psd("Highpass %g Hz" % f_cut, highpass(x, f_cut, f_sample))
	
	plt.figure(1)
	plot_psd('Raw', x)
	for f_center in [3, 30]:
		for frac_bw in [0.1, 0.5]:
			plot_psd("Bandpass %g Hz, Frac BW %g" % (f_center, frac_bw), bandpass(x, f_center, frac_bw, f_sample))
			plot_psd("Bandstop %g Hz, Frax BW %g" % (f_center, frac_bw), bandstop(x, f_center, frac_bw, f_sample))
	
	for i in range(2):
		plt.figure(i)
		plt.grid()
		plt.legend()
		plt.xlabel("Frequency [Hz]")
		plt.ylabel("PSD [V^2/Hz]")
		plt.title("Filter Testing")
	plt.show()
