#!/usr/bin/env python3

# Plots raw IV curves, both per-channel and per-column.
# Takes a folder with the IV curves as an argument.
# Based on the B3 showiv.py

import sys, os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mce_data as mce_data
import coordinates as coords
from colors import default_colors
import loadcurve as loadcurve
import calib.time202001 as calib

folder = sys.argv[1]
# name = os.path.split(folder)[1]
fn = os.path.join(folder,folder)

biasfn = fn + '_bias.npy'
f = mce_data.MCEFile(fn)
dname = os.path.split(fn)[0]
bias = np.load(biasfn) * calib.CAL_TES_BIAS_I * 1e6 #coverting bias dac to mirco amps
# all_lc = (loadcurve.load_loadcurves_muxcr(folder, calib,partial=True))
y = -1.0*f.Read(row_col=True, unfilter='DC').data

index_maxbias = 2000 #np.argmax(bias)

linestyles = ['solid', (0, (3, 1)), (0, (3, 1, 1, 1)), 'dotted']

for col in range(32):

	print("Column", col)

	plt.figure(1, figsize=(13,9))
	plt.clf()

	for row in range(33):

		plt.figure(2)

		muxcr_str = 'c%02ir%02i' % (col,row)

		chan_str = muxcr_str
		if row == 32:
			chan_str += ' Dark SQ1'
		else:
			try:
				det_x, det_f = coords.muxcr_to_xf(col, row)
				chan_str += ' x%02if%02i' % (det_x, det_f)
			except:
				pass

		d = y[row,col] * calib.CAL_SQ1_FB_I * 1e6 # mirco amps

		''' Bias has to be indexed by column because each one has different bias step '''
		plt.plot(bias[:][col], d, label= muxcr_str)

		plt.figure(1)
		# d -= d[index_maxbias] # Roughly line up normal slopes
		color = default_colors[row % 10]
		ls = linestyles[row // 10]
		if row == 32:
			ls = 'solid'
			color = 'k'
		plt.plot(bias[:][col], d, label=chan_str, color=color, ls=ls, alpha=0.8)

	plt.figure(2)
	plt.title(f'Load Curve for Column {col}')
	plt.ylabel(r"SQ1 Feedback [$\mu$A]")
	plt.xlabel(r"Bias Current [$\mu$A]")
	plt.grid()
	fn_s = os.path.join(dname, f'iv_col{col}.png')
	#plt.legend(loc='center left', bbox_to_anchor=(1.03, 0.5))
	plt.legend()
	plt.savefig(fn_s)
	plt.clf()

	plt.figure(1)
	plt.title("Column %02i Zoomed Load Curves" % col)
	plt.ylabel(r"Offset SQ1 Feedback [$\mu$A]")
	plt.xlabel(r"Bias Current [$\mu$A]")
	plt.grid()
	#plt.subplots_adjust(right=0.80)
	#plt.legend(loc='center left', bbox_to_anchor=(1.03, 0.5))
	plt.legend()
	plt.ylim(-50, 0)
	fn_s = os.path.join(dname, 'iv_zoomcol%02i.png' % col)
	plt.savefig(fn_s)
	plt.clf()
