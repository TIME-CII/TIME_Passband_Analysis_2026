from __future__ import division, print_function, unicode_literals, absolute_import

import time
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, hilbert

import timefpu.utils as utils
import timefpu.filtering as filt

# Bandpass filters a raw reference signal (usually the aux channel input)
# to find the I reference and phase shifts by 90 deg to find the Q reference.
# Returns the I and Q references.  'f_center' is the center of the bandpass
# filter, or None to infer the frequency from the signal itself.
def get_ref(ref_i_raw, f_sample, f_center=None, frac_bw=0.3, plot=False):

	if (np.ptp(ref_i_raw) == 0):
		raise ValueError("No ref i channel data detected!")
	
	ref_i_raw -= np.mean(ref_i_raw)
	
	# Check the peak in the data
	f, pxx = welch(ref_i_raw, fs=f_sample, nperseg=2048)
	f_peak = f[np.argmax(pxx)]
	
	if f_center is None:
		f_center = f_peak
		logging.debug("Extracted ref center frequency of %0.2f" % f_center)
	else:
		# Check the peak they gave is reasonable
		if abs(f_peak - f_center)/f_center > frac_bw:
			raise ValueError("Ref I channel peak frequency (%0.1f Hz) is outside the band pass filter." % (f_peak))
	
	ref_i = filt.bandpass(ref_i_raw, f_center=f_center, frac_bw=frac_bw, f_sample=f_sample)
	
	# Shift by 90 deg
	ref_q = -np.imag(hilbert(ref_i))
	
	if plot:
		plt.figure()
		t = np.arange(len(ref_i_raw)) / f_sample
		plt.plot(t, ref_i_raw, label="Reference I Raw")
		plt.plot(t, ref_i, label="Reference I")
		plt.plot(t, ref_q, label="Reference Q")
		maxt = max(t)
		plt.xlim(max(0,maxt-10/f_center),maxt)
		plt.xlabel("Time [sec]")
		plt.ylabel("Relative Signal")
		plt.title("Reference I and Q")
		plt.legend(loc='lower right')
		plt.show()
	
	return ref_i, ref_q, f_center

# Takes a [r][c][nsample] array as well as a raw reference signal of shape
# [nsample].  Returns an array of complex numbers of shape [r][c], where
# each value is the demodulation using the reference (I and Q being the 
# real and imaginary parts of the output).
def collapse_iq(data, ref_i_raw, f_sample, f_center=None, frac_bw=0.3):
	
	logging.debug("Demodulating...")
	
	ref_i, ref_q, f_center = get_ref(ref_i_raw, f_sample, f_center=f_center, frac_bw=frac_bw, plot=False)
	
	results = np.zeros(data.shape[:-1], dtype=np.csingle)
	
	for r in range(data.shape[0]):
		for c in range(data.shape[1]):
			
			# Remove thermal drift
			signal = filt.polysub(data[r,c], k=4)
			
			# Filter out other harmonics
			signal = filt.bandpass(signal, f_center=f_center, frac_bw=frac_bw, f_sample=f_sample)
			
			# Demodulate (filter down to DC)
			result_i = np.mean(np.multiply(ref_i, signal))
			result_q = np.mean(np.multiply(ref_q, signal))
			
			# Save the result as a complex number
			results[r,c] = result_i + 1.0j * result_q
				
	logging.debug("Demodulation Complete!")
	
	return results
