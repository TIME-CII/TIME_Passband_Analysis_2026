from __future__ import division, print_function

import sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import calib.time202001 as calib
import coordinates as coords
import params as params
import loadcurve as loadcurve

def read_load_curve(fnames,temperatures,det_x,det_f):

    bias_start = np.zeros((33,32))
    bias_stop = np.zeros((33,32))

    # Load in everything
    all_lc = []
    for k in range(len(temperatures)):
        all_lc.append(loadcurve.load_loadcurves_muxcr(fnames[k], calib))

    for i in range(len(det_x)):
        mux_c, mux_r = coords.xf_to_muxcr(det_x[i], det_f[i])
        print('col:',mux_c,'row:',mux_r)

        for j in range(len(temperatures)):
            lc = all_lc[j][mux_c][mux_r]
            # plt.plot(lc.tes_r, lc.tes_p*1e12, label="%i K" % temperatures[i])
            plt.plot(lc.tes_r, lc.tes_p_masked*1e12, label="%i K" % temperatures[j])
            m_chng = turn(lc.tes_p_masked*1e12)
            plt.scatter(lc.tes_r[m_chng],(lc.tes_p_masked*1e12)[m_chng])

        plt.ylabel("TES Power [pW]")
        plt.xlabel("Resistance [ohm]")
        plt.title("Detector x%02if%02i (c%02ir%02i)" % (det_x[i],det_f[i],mux_c,mux_r))
        plt.savefig('r_vs_p_r:%i_c:%i.png' %(mux_r,mux_c))
        plt.clf()
