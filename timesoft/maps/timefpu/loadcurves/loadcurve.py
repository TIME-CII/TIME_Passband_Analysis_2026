from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import logging

try:
	from functools import lru_cache
except:
	# If this fails, run "sudo pip install functools32"
	from functools32 import lru_cache

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, argrelmax

sys.path.append('./TIME-rx/timefpu/')
import mce_data, params
import coordinates as coords

# Loads one set of load curves from a given folder. Returns a 2d array
# [col,row] of LoadCurve objects.  r_ser_override is a dictionary
# of r_ser override values indexed as (col,row).
def load_loadcurves_muxcr(foldername, calib, r_ser_override={}, partial=False):

	if not os.path.isdir(foldername):
		raise ValueError("Specified load curve folder doesn't exist: " + str(foldername))

	filename = os.path.join(foldername, os.path.basename(foldername))
	# biasfilename = filename + '_bias.npy'
	biasfilename = filename + '.bias'
	#print(biasfilename)

	if not os.path.isfile(filename) or not os.path.isfile(biasfilename):
		raise ValueError("Expected files not present in load curve folder " + str(foldername))

	if r_ser_override is None:
		r_ser_override = {}

	f = mce_data.MCEFile(filename)

	tes_raw = -1.0*f.Read(row_col=True, unfilter='DC', unwrap=True).data
	#tes_raw = -1.0*f.Read(row_col=True, unfilter='DC').data
	bias_raw = np.loadtxt(biasfilename, skiprows=1)

	# Store LC with ascending bias
	reverse = False
	if bias_raw[1] < bias_raw[0]:
		bias_raw = bias_raw[::-1]
		reverse = True
	# for i in range(len(bias_raw)):
	# 	if bias_raw[i,1] < bias_raw[i,0]:
	# 		bias_raw[i] = np.flip(bias_raw[i])
	# 	reverse = True

	# ~ assert(np.all(np.diff(bias_raw) > 0))

	try:
		fluxjmp =  f.Read(row_col=True, field='fj').data
	except KeyError:
		fluxjmp = None

	assert tes_raw.shape[0] == params.N_MUX_ROWS, "Invalid row count"

	lcall = []
	for c in range(tes_raw.shape[1]): # Allow fewer cols if needed

		lccol = []
		for r in range(params.N_MUX_ROWS):

			if fluxjmp is None:
				fj = None
			else:
				fj = fluxjmp[r,c]

			tr = tes_raw[r,c]

			if reverse:
				tr = tr[::-1]
				if fluxjmp is not None:
					fj = fj[::-1]

			# Note that the mce_data is r,c but we are returning c,r
			# for this function
			lc = LoadCurve(bias_raw, tr, calib, fj, r_ser_override.get((c,r),None), partial)
			# lc = LoadCurve(bias_raw[c,:], tr, calib, fj, r_ser_override.get((c,r),None), partial)

			lccol.append(lc)

		lcall.append(lccol)

	return lcall

class LoadCurve:

	# didi_noise_thresh: The RMS noise threshold for segments of dI_tes/dI_bias
	def __init__(self, bias_dac, tes_dac, calib, fluxjmp=None, r_ser_override=None, partial=False, didi_noise_thresh=0.1):

		assert(len(bias_dac) == len(tes_dac))

		self.didi_noise_thresh = didi_noise_thresh

		self.partial = partial
		self.r_ser_override = r_ser_override
		self.fluxjmp = fluxjmp
		self.bias_dac = bias_dac
		self.tes_dac = tes_dac
		self.calib = calib

	# Bias current in Amps
	@property
	@lru_cache()
	def bias_i(self):
		return self.calib.CAL_TES_BIAS_I * self.bias_dac

	# TES current in Amps
	@property
	@lru_cache()
	def tes_i(self):
		return self.calib.CAL_SQ1_FB_I * self.tes_dac

	# TES current, corrected for the Ti normal slope to cross 0
	@property
	@lru_cache()
	def tes_i0(self):
		return self.tes_i - self.ti_rn_fit_offset

	# TES power with an attempt at masking the superconducting slope
	@property
	@lru_cache()
	def tes_p_masked(self):
		p = np.copy(self.tes_p)
		p[0:self.approx_transition_peak] = np.nan #float('nan')
		return p

	# TES power
	@property
	@lru_cache()
	def tes_p(self):
		return self.tes_r * np.power(self.tes_i0, 2)

	# TES resistance
	@property
	@lru_cache()
	def tes_r(self):
		return self.calib.R_SH * (self.bias_i/self.tes_i0 - 1) - self.r_ser

	# Series resistance
	@property
	@lru_cache()
	def r_ser(self):
		if self.r_ser_override is not None:
			return self.r_ser_override
		else:
			m = self.r_ser_fit_slope
			rser = self.calib.R_SH * ((1 / m) - 1)
			rser = max(0, rser)
			return rser

	# Ti TES normal resistance
	@property
	@lru_cache()
	def ti_rn(self):
		m = self.ti_rn_fit_slope
		return self.calib.R_SH * ((1 / m) - 1) - self.r_ser

	@property
	@lru_cache()
	def ti_rn_fit_mask(self):
		imin = self.calib.IMIN_BIAS_TI_RN
		imax = self.calib.IMAX_BIAS_TI_RN
		return self.generate_bias_mask(imin, imax, mask_gradient=True)

	@property
	@lru_cache()
	def r_ser_fit_mask(self):
		imin = self.calib.IMIN_BIAS_SC
		imax = min(self.calib.IMAX_BIAS_SC, self.bias_i[self.approx_transition_peak-3])
		return self.generate_bias_mask(imin, imax, mask_fj=True, mask_noise=True)
	
	# Return a mask over the given current ranges that, if possible,
	# does not include any SQ1 flux jumps.
	def generate_bias_mask(self, imin, imax, mask_noise=False, mask_fj=False, mask_gradient=False):

		bias = self.bias_i
		mask = np.logical_and(bias > imin, bias < imax)
		
		if mask_fj:
			if self.fluxjmp is not None and not np.all(self.fluxjmp == self.fluxjmp[0]) and np.any(mask):
				# Pick a section of constant flux jump.  Search for a good range from the
				# bottom of the transition.
				for ctr in range(int(max(self.fluxjmp[mask])), int(min(self.fluxjmp[mask]))-1, -1):
					newmask = np.logical_and(mask, np.equal(self.fluxjmp, ctr))
					vals = self.tes_dac[newmask]
					if len(vals) >= 3 and np.ptp(vals)/len(vals) < 2000:
						mask = newmask
						break
			else:
				# Can't do it with fj, use gradient instead
				mask_gradient = True
		
		if mask_gradient and np.any(mask):
			# Find the largest region of positive slope
			grad = np.gradient(self.tes_dac)[mask]
			borders = [0] + list(np.where(grad < 0)[0]) + [len(grad)-1]
			npoints = np.diff(borders)
			j = np.argmax(npoints)
			if npoints[j] > 3:
				bias_max = self.bias_i[mask][borders[j+1]]
				bias_min = self.bias_i[mask][borders[j]]
				if npoints[j] > 50:
					# If we have points to spare, prefer the top of the 
					# range where the slope is more consistent
					# ~ bias_min = bias_max - (bias_max - bias_min) * (40/npoints[j])
					bias_min = bias_max - (bias_max - bias_min) * 0.2
					bias_max = bias_max - (bias_max - bias_min) * 0.1
				newmask = np.logical_and(mask, self.bias_i <= bias_max)
				newmask = np.logical_and(newmask, self.bias_i >= bias_min)
				if np.count_nonzero(newmask) >= 3:
					mask = newmask

		if mask_noise and np.any(mask):

			# Mask off noisy areas in the range we found
			binsize = max(len(self.tes_dac[mask]) // 10, 5)
			noise = np.asarray([np.ptp(self.tes_dac[max(0,i-binsize):i+binsize]) for i in range(len(self.tes_dac))])
			thresh = min(noise[mask]) * 1.5
			newmask = np.logical_and(mask, noise < thresh)
			if len(self.tes_dac[newmask]) >= 3:
				mask = newmask

		if np.count_nonzero(mask) < 3:
			#~ logging.debug("Overriding short r_ser fit mask")
			mask = np.logical_and(bias > imin, bias < imax)

		return mask

	# The array of bias points used to fit for the Ti normal resistance
	@property
	def ti_rn_fit_bias(self):
		return self.bias_i[self.ti_rn_fit_mask]

	# The array of bias points used to fit for the series resistance
	@property
	def r_ser_fit_bias(self):
		return self.bias_i[self.r_ser_fit_mask]

	# The values of the offset TES current used to fit for Ti normal resistance
	@property
	@lru_cache()
	def r_ser_fit_i0(self):
		return self.r_ser_fit_slope * self.r_ser_fit_bias + (self.r_ser_fit_offset - self.ti_rn_fit_offset)

	# The values of the offset TES current used to fit for series resistance
	@property
	@lru_cache()
	def ti_rn_fit_i0(self):
		return self.ti_rn_fit_slope * self.ti_rn_fit_bias

	# The slope of the fit for Ti normal resistance in units [A/A]
	@property
	@lru_cache()
	def ti_rn_fit_slope(self):
		return self._find_slope(self.ti_rn_fit_mask)

	# The slope of the fit for Ti normal resistance in units [A] for
	# the non-offset TES current
	@property
	@lru_cache()
	def ti_rn_fit_offset(self):
		mask = self.ti_rn_fit_mask
		bias = np.mean(self.bias_i[mask])
		tes = np.mean(self.tes_i[mask])
		return tes - self.ti_rn_fit_slope * bias

	# The slope of the fit for TES series resistance resistance in units [A/A]
	@property
	@lru_cache()
	def r_ser_fit_slope(self):
		return self._find_slope(self.r_ser_fit_mask)

	# The slope of the fit for TES series resistance in units [A] for
	# the non-offset TES current
	@property
	@lru_cache()
	def r_ser_fit_offset(self):
		mask = self.r_ser_fit_mask
		bias = np.mean(self.bias_i[mask])
		tes = np.mean(self.tes_i[mask])
		return tes - self.r_ser_fit_slope * bias

	#~ def find_p_sat(self, r_frac_min=0.6, r_frac_max=0.9, dpdr_max=(0.1e-12/10e-3)):

		#~ # Mask from just above the peak
		#~ peak = self.approx_transition_peak + 3

		#~ # Starting at r_frac_max, return the first point where P vs R is
		#~ # sufficiently flat.  We don't necessarily want the flattest
		#~ # section, we want want a fixed r_frac (for optical efficiency)
		#~ # but with a repeatable and automatic way of adjusting the
		#~ # set point if we are on a nasty slope.
		#~ dpdr = np.gradient(self.tes_p) / np.gradient(self.tes_r)
		#~ r_max = r_frac_max * self.ti_rn
		#~ r_min = r_frac_min * self.ti_rn
		#~ i = peak + np.argmin(np.abs(self.tes_r[peak:] - r_frac_max*self.ti_rn))
		#~ while i > peak:
			#~ if (abs(dpdr[i]) < dpdr_max) and (self.tes_r[i] < r_max) and (self.tes_r[i] > r_min):
			#~ return self.tes_p[i]
			#~ i -= 1

		#~ logging.warning("Failed to find flat section for saturation power")

		#~ # Pick the middle of the range
		#~ r_frac = (r_frac_min + r_frac_max) / 2
		#~ i = peak + np.argmin(np.abs(self.tes_r[peak:] - r_frac*self.ti_rn))
		#~ return self.tes_p[i]

	def find_p_sat(self, r_frac=0.7, override_partial=False):
		if self.partial and not override_partial:
		# Generally a useful partial load curve is one which enters
		# the transition but does not latch.  Thus we pick the lowest
		# (last) bias and return that.
			return self.tes_p[np.argmin(self.bias_dac)]

		# Mask from just above the peak
		peak = self.approx_transition_peak + 2
		i = peak + np.argmin(np.abs(self.tes_r[peak:] - r_frac*self.ti_rn))
		return self.tes_p[i]


	# The TES saturation power
	@property
	@lru_cache()
	def p_sat(self):
		return self.find_p_sat()

	# Return the index of the approx transition peak
	@property
	@lru_cache()
	def approx_transition_peak(self):

		# For partial load curves, the lowest point should be in the transition
		if self.partial:
			return 0

		mask = (self.bias_i < self.calib.IMIN_BIAS_TI_RN)
		tes_i_lowbias = self.tes_i0[mask]

		try:
			order = len(tes_i_lowbias) // 20
			index_peak = max(argrelmax(tes_i_lowbias, order=order)[0])
		except ValueError:
			#~ logging.debug("Backup peak finder")
			index_peak = np.argmax(tes_i_lowbias)

		return index_peak

	# ~ # Form a heavily smoothed version of the load curve
	# ~ @property
	# ~ def tes_i_smoothed(self):
		# ~ window_length = max(len(self.tes_i) // 15, 15)
		# ~ return savgol_filter(self.tes_i, window_length, polyorder=2)

	# Return the slope of the masked region
	def _find_slope(self, mask):
		d_bias = np.mean(np.diff(self.bias_i[mask]))
		d_tes = np.mean(np.diff(self.tes_i[mask]))
		return d_tes / d_bias

	# Dimensionless slope dI_tes / dI_bias
	@property
	@lru_cache()
	def didi(self):
		return np.gradient(self.tes_i) / np.median(np.gradient(self.bias_i))

	# Median dimensionless slope dI_tes / dI_bias in the superconducting region
	@property
	@lru_cache()
	def didi_sc(self):
		return np.median(self.didi[self.r_ser_fit_mask])

	# Median dimensionless slope dI_tes / dI_bias in the Ti normal region
	@property
	@lru_cache()
	def didi_rn(self):
		return np.median(self.didi[self.ti_rn_fit_mask])

	# Returns a boolean with the best guess of whether or not the mux
	# channel is on
	@lru_cache()
	def is_on(self):

		if (np.ptp(self.tes_dac) < 0.1):
			return False

		return True

	# Returns a boolean with the best guess of whether or not the channel
	# looks like a normal resistor
	@lru_cache()
	def is_resistor(self):
		
		grad = np.gradient(self.tes_dac)
		g0 = np.median(grad)
		metric = np.max(np.abs(grad - g0)) / g0

		if (metric > 0.3):
			return False

		return True

	# Returns a boolean with the best guess of whether or not a TES
	# is present on this channel (inferred from the load curve)
	def is_present(self):

		if not self.is_on():
			return False

		if not self.valid_ti_rn():
			return False

		if not self.valid_r_ser():
			return False

		if self.is_resistor():
			return False

		return True

	# Returns a boolean with the best guess of whether or the Ti normal
	# resistance is sensible
	def valid_ti_rn(self):

		if self.ti_rn < 0.030 or self.ti_rn > 0.6:
			return False

		# Noisy
		if np.std(self.didi[self.ti_rn_fit_mask]) > self.didi_noise_thresh:
			return False

		return True

	# Returns a boolean with the best guess of whether or the series
	# resistance is sensible
	def valid_r_ser(self):

		if self.r_ser < 0 or self.r_ser > 0.02:
			return False

		# Noisy
		if np.std(self.didi[self.r_ser_fit_mask]) > self.didi_noise_thresh:
			return False

		return True

	# Return a string describing the overall load curve Valid outputs
	# are as listed below:
	# "present" - load curve is present and sane
	# "off" - channel mux is off
	# "unstable" - has a normal state, but superconducting region is poorly behaved
	# "huge-rser" - has a large r_ser value
	# "resistor" - behaves as a resistor, no transition
	# "short" - behaves as a short, no transition
	def classify(self):

		if not self.is_on():
			return 'off'

		if self.is_present():
			return 'present'

		rms_sc = np.std(self.grad_tes_dac[self.r_ser_fit_mask])
		rms_tirn = np.std(self.grad_tes_dac[self.ti_rn_fit_mask])
		rms_good = 200
		rms_bad = 500

		if self.r_ser >= 0 and rms_sc < rms_good and rms_tirn < rms_good:
			slope_frac = abs(self.didi_rn - self.didi_sc) / self.didi_rn
			if slope_frac < 0.3:
				if self.r_ser > 0.030:
					return 'resistor'
				else:
					return 'short'

		if rms_sc < rms_good and self.r_ser > 0.03:
			return 'huge-rser'

		if rms_tirn < rms_good:
			if (rms_sc > rms_bad) or (not self.valid_r_ser()):
				return 'unstable'

		logging.info("Failed to classify detector")
		return 'unknown'

	# Return a string describing the load curve state at the given
	# bias value (in DAC units).  Valid outputs are as listed below:
	# "superconducting", "normal-ti", "normal-al", "transition-ti",
	# "transition-al", "unknown", "bad", "off"
	@lru_cache()
	def state(self, biasval_dac):

		biasval_dac = abs(biasval_dac)

		if (np.ptp(self.tes_dac) < 1):
			return 'off'

		if not self.is_present():
			return 'bad'

		peak = self.bias_dac[self.approx_transition_peak]
		if abs(peak - biasval_dac) < 20 and not self.partial:
			return 'unknown'

		# Pull out the bias range near the given bias
		ipos = np.argmin(abs(self.bias_dac - biasval_dac + 20))
		ineg = np.argmin(abs(self.bias_dac - biasval_dac - 20))
		if ipos < ineg:
			ipos, ineg = ineg, ipos
		ineg = max(0, ineg)
		didi_near = self.didi[ineg:ipos]

		if np.std(didi_near) > self.didi_noise_thresh:
			return 'unknown'

		# Compare the local slope to the slope of the superconducting
		# and normal sections

		i = np.argmin(abs(self.bias_dac - biasval_dac))
		didi_local = self.didi[i]

		if abs(didi_local - self.didi_sc) / self.didi_sc < 0.2:
			return 'superconducting'

		if abs(didi_local - self.didi_rn) / self.didi_rn < 0.2:
			return 'normal-ti'

		if didi_local < -0.01:
			return 'transition-ti'

		return 'unknown'
