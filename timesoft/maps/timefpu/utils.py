from __future__ import division, print_function, unicode_literals, absolute_import

import logging
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.signal
from scipy.stats.mstats import gmean
	
import timefpu.coordinates as coords
import timefpu.params as params
import timefpu.mce_data as mce_data

# Configures the logging level to show debug messages. Can be run 
# multiple times safely.
def debug_on(mpl_debug_off=True):
	logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')
	if mpl_debug_off:
		# Turn off matplotlib debug messages
		logging.getLogger(matplotlib.__name__).setLevel(logging.WARNING) 

def assert_data_muxrc(data_muxrc):
	msg = "Expecting muxrc style data of shape [row,col,time]"
	assert len(data_muxrc.shape)==3, msg
	assert data_muxrc.shape[0] == params.N_MUX_ROWS, msg
	assert data_muxrc.shape[1] == params.N_MUX_COLS, msg
	assert data_muxrc.shape[2] > 0, msg
	
def load_mcerun(fname):
	
	assert(os.path.exists(fname))
	assert(os.path.isfile(fname))
		
	logging.info("Loading in data file...")
	
	f = mce_data.MCEFile(fname)
	data = f.Read(row_col=True, unfilter='DC')
	
	logging.info("Data loaded!")
	
	return data

# Coadd the data by spectrometer.  Assumes data is [mux_r,mux_c,time].
#
# Returns data of the shape [det_x,time] (regardless of det_x_list).
#
# If return_mask is True, also returns a mask of the shape [det_x, det_f]
# indicating whether or not that channel was included in the coadd for det_x.
#
# p2p_threshold and stdev_threshold determine how the standard deviation
# and peak to peak should be used to exlude data from the coadd.
# If they are None, they are not used.
# If they are a number, anything larger than that value is ejected.
# If they are a tuple of length 2, the values represent the min and max
# 	accepted values.
#
# If exclude_const is True, any channels that are constant across all
# time are ejected.
def coadd_by_x(data_muxrc, det_x_list=None, stdev_threshold=None, p2p_threshold=None, return_mask=False, exclude_const=True):
	
	assert_data_muxrc(data_muxrc)
	
	t = data_muxrc.shape[2]
	
	data_x_all = []
	mask_x_all = []
	
	try:
		p2p_min, p2p_max = p2p_threshold
	except TypeError:
		p2p_min = None
		p2p_max = p2p_threshold
		
	try:
		stdev_min, stdev_max = stdev_threshold
	except TypeError:
		stdev_min = None
		stdev_max = stdev_threshold
	
	for det_x in range(params.N_CHAN_SPATIAL):
		
		data_x = np.zeros(t)
		mask_x = [False for i in range(params.N_CHAN_SPECTRAL)]
		
		if det_x_list is None or det_x in det_x_list:

			for det_f in range(params.N_CHAN_SPECTRAL):
				
				try:
					c,r = coords.xf_to_muxcr(det_x, det_f)
				except NotImplementedError:
					continue
					
				d = data_muxrc[r,c,:]
				
				ptp = np.ptp(d)
				std = np.std(d)
				
				if stdev_max is not None and std > stdev_max:
					logging.info("Ejecting x%02if%02i (c%02ir%02i) for exceeding stdev maximum" % (det_x, det_f, c, r))
					continue
				if p2p_max is not None and ptp > p2p_max:
					logging.info("Ejecting x%02if%02i (c%02ir%02i) for exceeding peak to peak maximum" % (det_x, det_f, c, r))
					continue
				if stdev_min is not None and std < stdev_min:
					logging.info("Ejecting x%02if%02i (c%02ir%02i) for failing to meet stdev minimum" % (det_x, det_f, c, r))
					continue
				if p2p_min is not None and ptp < p2p_min:
					logging.info("Ejecting x%02if%02i (c%02ir%02i) for failing to meet peak to peak minimum" % (det_x, det_f, c, r))
					continue
				if exclude_const and not (ptp > 0):
					logging.info("Ejecting x%02if%02i (c%02ir%02i) for remaining constant over time" % (det_x, det_f, c, r))
					continue
					
				data_x += d
				mask_x[det_f] = True
		
		data_x_all.append(data_x)
		mask_x_all.append(mask_x)

	if return_mask:
		return data_x_all, mask_x_all
	else:
		return data_x_all

# 'outfile' is the file name to save the plot in, or None to plt.show()
# 'data' is a dict with index (det_x, det_f) giving the number plotted on the color scale.
#		Can be None if you are using markers only and wish to hide the color bar.
# 'markers' is a dict with index (det_x, det_f) giving the matplotlib color string ('r', 'yellow', etc)
#		for that pixel.  This overrides the color scale and can be used to indicate
#		special cases like bad detectors.
# 'title' is the plot title
# 'clabel' is the color scale axis label
# 'cmin' is the color scale lower limit, or None to autoscale
# 'cmax' is the color scale upper limit, or None to autoscale
# 'scale' is a number to scale all of the data values by (for unit conversion)
# 'missing_color' is the color to plot when no data or markers are present
# 'mux_space' will, if True, change 'data' and 'markers' to be indexed
#		as (mux_c, mux_r) instead of (x,f) and will produce a plot
#		in mux space instead of xf space.
# 'marker_labels' is a dict of {'color_str':'legend_label'} for some or all
#		of the colors strings in 'markers'
# 
# Ex: utils.plot_map(outfile=None, data={(5,8):1.1,(3,7):2.2}, markers={(4,5):"r",(3,5):"m"})
#
def plot_map(outfile, data=None, markers=None, title='',
	clabel='', cmin=None, cmax=None, scale=1.0, missing_color='grey',
	cmap='rainbow', mux_space=False, marker_labels=None, 
	cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
	logscale=False):
		
	val_a = []
	val_b = []
	val_c = []
	
	marker_legend = (marker_labels is not None)
	colorbar = (data is not None)

	if mux_space:

		mshape = 's'
		msize = 70

		# a, b = mux_c, mux_r
		a_range = list(range(params.N_MUX_COLS))
		b_range = list(range(params.N_MUX_ROWS))
		transpose = False
		
		if colorbar and marker_legend:
			figsize = (10,7.5)
			legend_bbox = (1.18, 0.5)
			subplot_right = 0.7
		elif colorbar:
			figsize = (7,7.5)
		elif marker_legend:
			figsize = (9,7.5)
			legend_bbox = (1.02, 0.5)
			subplot_right = 0.75
		else:
			figsize = (8,7.5)
			
	else:
		
		# Slightly rectangular for reasonable aspect ratio
		mshape = [(-0.55,-1),(-0.55,1),(0.55,1),(0.55,-1)]
		msize = 120
		
		# a, b = det_x, det_f
		a_range = list(range(params.N_CHAN_SPATIAL))
		b_range = list(range(params.N_CHAN_SPECTRAL))
		transpose = True
		
		if colorbar and marker_legend:
			figsize = (12.1,4.95)
			legend_bbox = (1.12, 0.5)
			subplot_right = 0.785
		elif colorbar:
			figsize = (10.42,4.95)
		elif marker_legend:
			figsize = (11.7,4.95)
			legend_bbox = (1.01, 0.5)
			subplot_right = 0.84
		else:
			figsize = (9.6,4.95) 
			
		#~ if colorbar and marker_legend:
			#~ figsize = (15,5)
			#~ legend_bbox = (1.10, 0.5)
			#~ subplot_right = 0.83
		#~ elif colorbar:
			#~ figsize = (13.5,5)
		#~ elif marker_legend:
			#~ figsize = (15,5)
			#~ legend_bbox = (1.01, 0.5)
			#~ subplot_right = 0.85
		#~ else:
			#~ figsize = (13,5)

	plt.figure(figsize=figsize)
	
	markers_a = {}
	markers_b = {}

	for a in a_range:
		for b in b_range:

			if markers is not None and (a, b) in markers:
				mc = markers[(a, b)]	
			elif data is not None and (a, b) in data:
				val_a.append(a)
				val_b.append(b)
				val_c.append(scale * data[(a, b)])
				continue
			else:
				mc = missing_color
			
			# This is a marker, not a value
			if mc not in markers_a.keys():
				markers_a[mc] = []
				markers_b[mc] = []
			markers_a[mc].append(a)
			markers_b[mc].append(b)
	
	if transpose:
		val_a, val_b = val_b, val_a
		markers_a, markers_b = markers_b, markers_a
	
	# Show markers
	for mc in markers_a.keys():
		plt.scatter(markers_a[mc], markers_b[mc], marker=mshape, s=msize, c=mc)

	cnorm = None
	if logscale:
		cnorm = matplotlib.colors.LogNorm()
	
	# Show values
	plt.scatter(val_a, val_b, marker=mshape, s=msize, c=val_c, cmap=cmap, norm=cnorm)
	
	plt.title(title, fontsize=titlesize)
	
	if mux_space:
			
		plt.xlim(-1, params.N_MUX_COLS)
		plt.ylim(-1, params.N_MUX_ROWS)
		plt.xlabel("Multiplexing Column", fontsize=fontsize)
		plt.ylabel("Multiplexing Row", fontsize=fontsize)
	
	else:
		
		plt.xlim(-1, params.N_CHAN_SPECTRAL)
		plt.ylim(-1, params.N_CHAN_SPATIAL)
		plt.xlabel("Frequency Index ($f$ Coordinate, Lowest Frequency at $f=0$)", fontsize=fontsize)
		plt.ylabel("Spatial Index ($x$ Coordinate)", fontsize=fontsize)
		
		for f in [0,8,16,24,36,48,60]:
			plt.axvline(f - 0.5, color = 'k', lw=0.6)
		for m in range(0,17,4):
			plt.axhline(m - 0.5, color = 'k', lw=0.6)

	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	
	if marker_legend:
		# Plot dummy scatter plots for the labels
		for key, val in marker_labels.items():
			plt.scatter([100], [100], marker=mshape, s=msize, c=key, label=val)
		plt.legend(loc='center left', bbox_to_anchor=legend_bbox, fontsize=legendsize)
		
	if colorbar:
		divider = make_axes_locatable(plt.gca())
		cax = divider.append_axes("right", "2%", pad="2%")
		cb = plt.colorbar(cax=cax, ticks=cticks)
		if cticklabels is not None:
			cb.set_ticklabels(cticklabels)
			cb.ax.tick_params(labelsize=fontsize) 
		cb.set_label(clabel, fontsize=fontsize)
		plt.clim(cmin, cmax)
	
	plt.tight_layout()
	
	if marker_legend:
		plt.subplots_adjust(right=subplot_right)

	if outfile is None:
		plt.show()
	else:
		plt.savefig(outfile)

# Equivalent to the output of scipy.signal.coherence, except that the
# data is binned in log space according to 'scale' (passed to psd_logbin).
# This is done because binning doesn't work properly in the final
# coherence units, so we bin the spectra before computing coherence.
def coherence_logbinned(x, y, fs, nperseg=16384, scale=1.01):
	
	f, Pxx = scipy.signal.welch(x, fs=fs, nperseg=nperseg)
	_, Pyy = scipy.signal.welch(y, fs=fs, nperseg=nperseg)
	_, Pxy = scipy.signal.csd(x, y, fs=fs, nperseg=nperseg)
	
	# Be careful not to take the absolute value of Pxy before you
	# bin, that would bias the result!
	_, Pxx = psd_logbin(f, Pxx, scale=scale)
	_, Pyy = psd_logbin(f, Pyy, scale=scale)
	f, Pxy = psd_logbin(f, Pxy, scale=scale)
	
	Cxy = np.abs(Pxy)**2 / Pxx / Pyy

	return f, Cxy

# Return the alias of frequency f when sampling at f_sample
def f_alias(f, f_sample):
	f_nyq = f_sample / 2
	direction = (f // f_nyq) % 2 # 0 or 1
	f_alias = direction*f_nyq + (-1)**direction * (f % f_nyq)
	return f_alias

# Returns a PSD with aliasing after a change in sample frequency.
# Assumes f is linearly sampled and Pxx is in power units (not amplitude).
def psd_alias(f, Pxx, f_sample):
	
	f_nyq = f_sample / 2

	f_out = f[f <= f_nyq].copy()
	Pxx_out = np.zeros_like(f_out)
	
	# Find the aliases
	f_aliased = f_alias(f, f_sample)
	i_out = np.digitize(f_aliased, f_out)
	
	# Adjust the output of np.digitize to match f_out
	i_out -= 1
	i_out[i_out >= len(f_out)] = len(f_out) - 1
	
	# Aliases add linearly in power space (quadrature in amplitude space).
	for j in range(len(i_out)):
		Pxx_out[i_out[j]] += Pxx[j]

	return f_out, Pxx_out	

# Return a spectrum rebinned for log spacing (preserves low frequency
# tail but averages down high freqency noise)
def psd_logbin(f, Pxx, scale=1.01):
	
	fbin = f.copy()
	Pxxbin = Pxx.copy()
	
	finsert = 0
	fwork = 0
	fsize = 1
	while fwork < len(fbin):
		fmax = min(len(fbin), int(fwork + fsize))
		fbin[finsert] = gmean(fbin[fwork:fmax])
		Pxxbin[finsert] = np.mean(Pxxbin[fwork:fmax])
		finsert += 1
		fwork = fmax
		fsize *= scale  # Scale up the next bin size
	fbin = fbin[:finsert]
	Pxxbin = Pxxbin[:finsert]
	
	return fbin, Pxxbin

# Return the suffix used for writing a number (21st, 6th, etc).
_num_suffix = ["th","st","nd","rd"] + ['th']*6
def num_suffix(num):
	n = abs(int(num))
	if (n // 10) % 10 == 1:
		return 'th' # *10-*19
	else:
		return _num_suffix[n % 10]

# Return the first integer power of 2 that is larger than x
def log2ceil(x):
	return int(2**np.ceil(np.log2(x)))

# Wrap phases onto the range -pi to pi
def wrap_phase(phase):
	return (phase + np.pi) % (2 * np.pi) - np.pi
	
# Returns a string with the most relevant SI units.
# Prints "precision"+1 digits, with at least "precision" guaranteed
# to be significant digits for non-extreme values.
# "min_decade" will clip to use prefixes above ~10^min_decade, allowing
# you to essentially set a noise floor for the reported data.
_si_prefix = ['a','f','p','n','u','m','','k','M','G','T','P','E']
_si_decade = [-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18]
_si_str = "%0.*f %s%s"
def pretty_units(val, base_unit, precision = 3, min_decade = None):
	
	min_i = 0
	if min_decade is not None:
		min_i = int(round((min_decade - _si_decade[0]) / 3))
		min_i = np.clip(min_i, 0, len(_si_prefix)-1)
		
	if val == 0:
		return _si_str % (precision, 0, _si_prefix[min_i], str(base_unit))
	elif not np.isfinite(val):
		return str(val) + " " + str(base_unit)
		
	# Look for the prefix that fits best
	for i in range(min_i, len(_si_prefix)):
		if (np.log10(abs(val)) < _si_decade[i] + 2.5) or (i == len(_si_prefix)-1):
			v = val*(10**-_si_decade[i])
			p = precision - max(0, int(np.log10(abs(v))))
			return _si_str % (p, v, _si_prefix[i], str(base_unit))

# Return a Gaussian
def gaussian(x, sigma, x0=1, A=1):
	return A * np.exp(-0.5 * ((x-x0)/sigma)**2)

# Find the FWHM of a Gaussian given the standard deviation
def gaussian_fwhm(sigma):
	return 2*np.sqrt(2*np.log(2))*sigma

# Automatically enable logging on import
debug_on()

if __name__ == '__main__':
	
	print("Test of pretty_units")
	assert len(_si_prefix) == len(_si_decade)
	tocheck = np.append(np.logspace(-20,20,100), [0,float('inf'),float('nan')])
	for i in tocheck:
		stand = "%0.8e" % i
		print(stand, 
			  pretty_units(i,'W', precision=5), 
			  pretty_units(i,'W'), 
			  pretty_units(-i,'W'), 
			  pretty_units(i,'W', min_decade=-6), 
			  pretty_units(i,'W', min_decade=8), 
			  pretty_units(i,'W', min_decade=-30), 
			  pretty_units(i,'W',min_decade=80),
			  pretty_units(i,'W', precision=2), 
			  )
