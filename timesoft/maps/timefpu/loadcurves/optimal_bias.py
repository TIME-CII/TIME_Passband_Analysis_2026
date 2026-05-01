from __future__ import division, print_function
import sys,os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
fontP = FontProperties()
fontP.set_size('xx-small')
from PIL import Image
import calib.time202001 as calib
import coordinates as coords
import params as params
import loadcurve as loadcurve
from scipy.stats import linregress
import pdb
from math import *
from termcolor import colored

class optimal_bias_finder():

    def __init__(self,fname,cols,rows,saveplots=0,temp=0,diagnostic=0):

        self.fname = fname
        self.cols = cols 
        self.rows = rows 
        self.saveplots = saveplots 
        self.temp = temp
        self.diagnostic = diagnostic
        self.good_count = 0
        self.colors = plt.cm.jet(np.linspace(0,1,33))
        self.flags = np.zeros((33,32,5)) # To catch errors
        self.good_det = np.zeros((33,32))
        self.trans_limits = np.zeros((33,32,2))
        '''
        FLAG DEFINITIONS :
            #1 = try/except ... there was no detector
            #2 = len of the array was zero
            #3 = turn data mean of zero
            #4 = turn data error with slope
            #5 = not enough data to interpolate
        '''
        # read in .bias and .run files using /timefpu
        self.all_lc = (loadcurve.load_loadcurves_muxcr(self.fname, calib))
        lc = self.all_lc[0][0]
        self.tes_p = np.zeros((33,32,len(lc.tes_p_masked)))
        self.tes_r = np.zeros((33,32,len(lc.tes_p_masked)))
        self.bias_i = np.zeros((33,32,len(lc.tes_p_masked)))
        self.tes_i = np.zeros((33,32,len(lc.tes_p_masked)))
        # self.tes_p = np.zeros((33,32,len(lc.tes_p_masked)))

        if self.diagnostic :
            self.series_slopes = np.zeros((33,32))
            self.normal_slopes = np.zeros((33,32))

        for c in self.cols:
            for r in self.rows:
                print(colored(f'col:{c} , row:{r}','yellow'))
                # grab the data
                self.read_in_data(c,r)
                # remove crap data
                err = self.cull_det(c,r)
                if not err :
                    # find the transition region
                    self.find_transition(c,r) #grabs data where transition is

            self.col_transition(c)

        print(colored(f'number of detectors with transition: {self.good_count}','magenta'))
        np.save(fname + '/plots/good_det.npy', self.good_det)

        # make a plot of normal/series slopes distributions
        if self.diagnostic :
            s_counts,s_bins = np.histogram(self.series_slopes,bins=50)
            n_counts,n_bins = np.histogram(self.normal_slopes,bins=50)

            fig, ax = plt.subplots(1,2)
            ax[0].hist(s_bins[:-1], s_bins, weights=s_counts, label=f'mean = {np.mean(s_bins):.3f}')
            ax[0].legend()
            ax[0].set_title('Series Slopes')
            ax[0].set_ylabel("Det Counts")
            ax[0].set_xlabel("Slope")
            ax[0].set_yscale('log')
            ax[0].set_xscale('log')

            ax[1].hist(n_bins[:-1], n_bins, weights=n_counts, label=f'mean = {np.mean(n_bins):.3f}')
            ax[1].legend()
            ax[1].set_title('Normal Slopes')
            ax[1].set_ylabel("Det Counts")
            ax[1].set_xlabel("Slope")
            ax[0].set_yscale('log')
            ax[0].set_xscale('log')
            plt.show()
            plt.close()

    def read_in_data(self,c,r):

         # grab single loadcurve
        self.lc = self.all_lc[c][r]

        # masking data less than zero on the y axis RP curve
        self.tes_p[c,r,:] = self.lc.tes_p_masked * 1e12
        self.tes_r[c,r,:] = self.lc.tes_r
        self.bias_i[c,r,:] = self.lc.bias_i * 1e6
        self.tes_i[c,r,:] = self.lc.tes_i * 1e6

        return True

    def cull_det(self,c,r,):

        # let's mask the end of the power and bias data because sometimes
        # there is a second transition there...
        if np.max(self.bias_i[c,r,:]) > 1400.0 :
            self.bias_i[c,r,-1000:] = np.nan
            self.tes_i[c,r,-1000:] = np.nan
            self.tes_r[c,r,-1000:] = np.nan
            self.tes_p[c,r,-1000:] = np.nan

        # fit for slope before and after transition region in IV curve
        first_tes = self.tes_i[c,r,0:100]
        last_tes = self.tes_i[c,r,-2500:-2000]
        first_bias = self.bias_i[c,r,0:100]
        last_bias = self.bias_i[c,r,-2500:-2000]

        line_first = linregress(first_bias,first_tes)
        line_last = linregress(last_bias,last_tes)
        if self.diagnostic :
            self.normal_slopes[c,r] = line_first.slope
            self.series_slopes[c,r] = line_last.slope

        # get rid of if the slope is zero
        if line_first.slope == 0.0 or line_last.slope == 0.0 :
            self.tes_p[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
            self.tes_r[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
            self.bias_i[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
            self.tes_i[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
            return True

        '''
           ####################################################
                The values for the slopes were chosen from 
                visual inspection of the load curves, and may 
                not work on new detector modules.
           ####################################################
        '''

        # get rid of if normal slope is not between these values
        if line_first.slope < 0.15 or line_first.slope > 0.6 :
            print('bad normal slope')
            self.tes_p[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
            self.tes_r[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
            self.bias_i[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
            self.tes_i[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
            return True

        # elif line_last.slope < 0.03 or line_last.slope > 0.1:
        #     print('bad series slope (too low')
        #     self.tes_p[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
        #     self.tes_r[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
        #     self.bias_i[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
        #     self.tes_i[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
        #     return True

        else :
            if self.saveplots:
                fig, ax = plt.subplots(1,2, figsize=(12,10), dpi=300)
                ax[0].plot(self.bias_i[c,r,:],self.tes_i[c,r,:])
                ax[0].scatter(first_bias,first_tes, color='black')
                ax[0].plot(first_bias,line_first.intercept + (line_first.slope*first_bias), 'green',label=f'normal slope = {line_first.slope:.3f}')
                ax[0].scatter(last_bias,last_tes, color='black')
                ax[0].plot(last_bias,line_last.intercept + (line_last.slope*last_bias), 'r',label=f'series slope = {line_last.slope:.3f}')
                ax[0].legend(loc='best')
                ax[0].set_title(f'Slope Fits IV c{c},r{r}')
                ax[0].set_ylabel("TES Current [$\mu$A]")
                ax[0].set_xlabel("Bias Current [$\mu$A]")

                ax[1].plot(self.tes_r[c,r,:],self.tes_p[c,r,:])
                ax[1].set_title(f'Slope Fits RP c{c},r{r}')
                ax[1].set_ylabel("TES Power [pW]")
                ax[1].set_xlabel("Resistance [ohm]")
                # plt.show() 
                plt.savefig(fname + f'/plots/slope_fit_c{c}_r{r}.png')
                plt.clf()
                
            return False

    def find_transition(self,c,r):

        tes_i = self.tes_i[c,r,:]
        bias_i = self.bias_i[c,r,:]
        tes_p = self.tes_p[c,r,:]
        tes_r = self.tes_r[c,r,:]

        # if we haven't already culled the data earlier from weird slopes... 
        # check the slope of the power data
        if not np.isnan(np.nanmean(tes_p)):
            d_r = np.nanmean(np.diff(tes_r))
            d_p = np.nanmean(np.diff(tes_p))
            m = d_p / d_r
            print(colored(f"RP slope : {m}",'red'))
            if m > 10000 :
                print(colored(f'c{c},r{r}, questionable power data...','red'))

            ###################################################
            # i'm trimming the first several points because 
            # sometimes there are strange features that look
            # like transitions in the TES I slope
            ##################################################
            bias_i = self.bias_i[c,r,50:]
            tes_i = self.tes_i[c,r,50:]
            diff_tes = np.diff(tes_i)
            tes_r = self.tes_r[c,r,50:]
            tes_p = self.tes_p[c,r,50:]

            # specify outliers from linear fit
            std = np.nanstd(diff_tes)
            mean = np.nanmean(diff_tes)
            lower = mean - (3 * std)
            upper = mean + (3 * std)

            inb  = np.where(np.logical_or(diff_tes < lower, diff_tes > upper))[0]

            if len(inb) != 0:

                self.good_count += 1
                self.good_det[c,r] = 1

                self.trans_limits[c,r,0] = np.nanmin(bias_i[:-1][inb])
                self.trans_limits[c,r,1] = np.nanmax(bias_i[:-1][inb])

                fig, ax = plt.subplots(1,2,figsize=(12,10), dpi=300)
                ax[0].scatter(bias_i[inb],np.diff(tes_i)[inb],color='black')
                ax[0].plot(bias_i[:-1],np.diff(tes_i))
                ax[0].set_title('Transition Region from IV Slope')
                ax[0].set_ylabel("Delta TES_I")
                ax[0].set_xlabel("Bias Current [$\mu$A]")

                ax[1].plot(bias_i,tes_i)
                # ax[1].scatter(bias_i[:-1][inb],tes_i[:-1][inb],color='black')
                # ax[1].axvspan(np.nanmin(bias_i[inb]), np.nanmax(bias_i[inb]), alpha = 0.5, label='transition region',color='blue')
                ax[1].axvline(np.max(bias_i[inb]), color = 'r')
                ax[1].set_ylabel("TES Current [$\mu$A]")
                ax[1].set_xlabel("Bias Current [$\mu$A]")
                ax[1].set_title(f'Original IV Curve, c{c},r{r}')
                # ax[1].legend(loc='best')
                plt.savefig(fname + f'/plots/transition_c{c}_r{r}.png')
                # plt.show()
                plt.clf()

            else :
                print('no transition')
                # setting the bad data to nans so we don't see it on plots
                self.tes_p[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
                self.tes_r[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
                self.bias_i[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)
                self.tes_i[c,r,:] = np.full([len(self.lc.tes_p_masked)], np.nan)

    def col_transition(self,c):

        data = self.trans_limits[c,:,1]
        mid = np.mean(data)

        for r in range(32):
               if not np.all(np.isnan(self.bias_i[c,r,:])) :
                    plt.plot(self.bias_i[c,r,:],self.tes_i[c,r,:],color=self.colors[r],label=f'c{c}r{r}')
        
        plt.axvline(mid, color = 'black',label=f'{mid:.1f}',linestyle='dashed')
        plt.ylabel("TES Current [$\mu$A]")
        plt.xlabel("Bias Current [$\mu$A]")
        plt.grid(which='major', color='dimgrey', linewidth=0.8)
        plt.grid(which='minor', color='darkgrey', linestyle=':', linewidth=0.5)
        plt.minorticks_on()
        plt.title(f'Column {c} Optimal Bias')
        plt.legend(loc='best',ncol=2)
        plt.savefig(fname + f'/plots/opt_bias_col{c}.png')
        plt.clf()
            


if __name__ == '__main__':
    # fname = '/Volumes/KINGSTON/20231019/20231019a_iv'
    # fname = '/Volumes/KINGSTON/20231213a_iv'
    # fname = 'sample-load-curves/iv_opteff_77K_run1'
    fname = '/Users/butler/Documents/20231221a_iv'

    if not os.path.isdir(fname + '/plots'):
        os.mkdir(fname + '/plots')

    # cols = [0]
    # rows = [14,15,0,19,11,10]
    cols = np.arange(31)
    rows = np.arange(32)

    optimal_bias_finder(fname,cols,rows,saveplots = 1, temp= 77, diagnostic=0)