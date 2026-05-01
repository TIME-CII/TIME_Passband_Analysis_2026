rom __future__ import division, print_function
import sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
from scipy import interpolate
import pdb
from math import *
from termcolor import colored
# sys.path.append('/path/to/file')
from iv_curve_test import main
import mce_data as mce_data

def turn(y,x,bias_y,bias_x):
    # checking for nans in masked power array
    x = np.ma.array(x, mask=np.isnan(x))
    y = np.ma.array(y, mask=np.isnan(y))
    bias_x = np.ma.array(bias_x, mask=np.isnan(bias_x))
    bias_y = np.ma.array(bias_y, mask=np.isnan(bias_y))

    if np.mean(y) != 0.0 and np.mean(bias_y) != 0.0 : #gets rid of unecessary data
        #finds what kind of slope is in the transition data
        z, cov = np.polyfit(list(bias_x), list(bias_y), 1, cov=True) # I want to know if my data is a line
        e = np.sqrt(np.diag(cov))
        intercept_error = e[1]
        slope_error = e[0]

        if intercept_error > 1.5 : # if my data isn't a line, how different from a line is it?
        # I made that number up from looking at data (1.5)
            m = np.diff(y) #finding the slope
            out = np.where(np.logical_and(np.greater(m,0.0),np.less(m,0.06))) # where is the slope between those two numbers?
            out = [list(k) for k in out][0]

            if len(out) > 20 :
                return out[0:19], None #trims data array to only 20 points
            else :
                return out, None
        else :
            print('error with slope')
            return None, 4
    else :
        print('mean of zero')
        return None, 3



def detector(fn,mce_data,calib):
    biasfn = fn + '_bias.npy'
    f = mce_data.MCEFile(fn)
    dname = os.path.split(fn)[0]
    bias = np.load(biasfn) * calib.CAL_TES_BIAS_I * 1e6 #coverting bias dac to mirco amps
    tes = np.load(biasfn) * calib.CAL_SQ1_FB_I #converting to TES
    # all_lc = (loadcurve.load_loadcurves_muxcr(folder, calib,partial=True))
    y = -1.0*f.Read(row_col=True, unfilter='DC').data


if __name__ == '__main__':
    # this is where your code starts running
    transition = False
    i = 0
    init_bias = np.load('bias_list_293.npy',allow_pickle=True)
    # we have to tell the script which column we are working with
    bias_min = init_bias[:,1] # = [good_count,bias_min,power_min,opt_num]
    # check and see how many detectors we need to reach = opt_num
    # good_count = how many total working detectors in the column
    # bias_min = starting bias
    # power_min = starting power
    '''
    while transition == False :

        if i != 0 :
            if transition == False :
                # tell the user to provide what the new bias level should be
                user_input = int(input('provide new bias limit :'))
                bias_min = bias_min - user_input
            else :
                # end the while loop
                transition = True
        else :
            # first time through, just use the file
            bias_min = init_bias[:,1]
            np.save('new_bias_list_293.npy',[init_bias[:,0],bias_min,init_bias[:,2],init_bias[:,3]],allow_pickle=True)
            main('partial_load_test','293', calib)
            '''
        # check the output of your new bias level
        # this is where you would put plotting stuff
        folder = 'partial_load_test'
        # name = os.path.split(folder)[1]
        fn = os.path.join(folder,folder)
        
        detector(fn,mce_data,calib)

        resp = bool(input('If you are happy with the number of transitions, enter 1 for yes and 0 for no'))
        transition = resp

        i += 1



good_count, power_mean, bias_mean, opt_num = read_load_curve(fnames[0],det_x,det_f,saveplots=saveplots,col=7,temp=temperatures[0])
