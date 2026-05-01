from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('xx-small')
import calib.time202109 as calib
import coordinates as coords
import params as params
import loadcurve as loadcurve
from math import *
from termcolor import colored

fname = '/Volumes/KINGSTON/20231019/20231019a_iv'

det_x = np.arange(32)
det_f = np.arange(33)

all_lc = (loadcurve.load_loadcurves_muxcr(fname, calib))
    # define plot colors
    # colors = plt.cm.jet(np.linspace(0,1,33))
for i in range(len(det_x)): #detector position
    for k in range(len(det_f)): #detector frequency
        try :
            mux_c, mux_r = coords.xf_to_muxcr(det_x[i], det_f[k])
            print(colored('col: %s , row: %s' %(mux_c,mux_r),'green'))

            lc = all_lc[mux_c][mux_r]
            if len(lc.tes_r) != 0 :
                plt.figure(1)
                plt.plot(lc.tes_r, lc.tes_p_masked*1e12, label='Det # %s'%(mux_r))

            else :
                print('temp %s for det r = %s , c = %s doesnt exist' %(temp,mux_r,mux_c))

            plt.figure()
            plt.plot(lc.tes_r, lc.tes_p_masked*1e12)
            plt.ylabel("TES Power [pW]")
            plt.xlabel("Resistance [ohm]")
            plt.title("Detector x%02if%02i (c%02ir%02i)" % (det_x[i],det_f[k],mux_c,mux_r))
            plt.savefig('/Volumes/KINGSTON/20231019/20231019a_iv/plots/rvp_r:%i_c:%i.png' %(mux_r,mux_c))
            plt.clf()

            # plt.figure(2)
            # plt.ylabel("TES Current [$\mu$A]")
            # plt.xlabel("Bias Current [$\mu$A]")
            # plt.title("Detector x%02if%02i (c%02ir%02i)" % (det_x[i],det_f[k],mux_c,mux_r))
            # plt.savefig('/Volumes/KINGSTON/20231019/20231019a_iv/plots/test_bias_vs_tes_r:%i_c:%i.png' %(mux_r,mux_c))
            # plt.clf()

        except NotImplementedError :
            print('NotImplementedError')