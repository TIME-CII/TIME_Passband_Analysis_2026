from __future__ import division, print_function
import sys,os
import numpy as np
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

def read_load_curve(fname,det_x,det_f,saveplots=0,col=0,temp=0):
    bias_x = np.zeros((33,32,20))
    bias_y = np.zeros((33,32,20))
    current_x = np.zeros((33,32,20))
    current_y = np.zeros((33,32,20))
    flags = np.zeros((33,32,5)) # To catch errors
    '''
    FLAG DEFINITIONS :
        #1 = try/except ... there was no detector
        #2 = len of the array was zero
        #3 = turn data mean of zero
        #4 = turn data error with slope
        #5 = not enough data to interpolate
    '''
    # keep track of how many good detectors per column
    good_count = 0
    # Load in everything
    all_lc = (loadcurve.load_loadcurves_muxcr(fname, calib))
    # define plot colors
    colors = plt.cm.jet(np.linspace(0,1,33))
    for i in range(len(det_x)): #detector position
        for k in range(len(det_f)): #detector frequency
            try :
                mux_c, mux_r = coords.xf_to_muxcr(det_x[i], det_f[k])
                print('col:',mux_c,'row:',mux_r)

                lc = all_lc[mux_c][mux_r]
                if len(lc.tes_r) != 0 :
                    if saveplots :
                        plt.figure(1)
                        plt.plot(lc.tes_r, lc.tes_p_masked*1e12, label='Det # %s'%(mux_r),c=colors[mux_r])

                    # provide both sets of y data for filtering useless data
                    m_chng, err = turn(lc.tes_p_masked*1e12,lc.tes_r,lc.tes_i*1e6,lc.bias_i*1e6) #grabs data where transition is
                    if err != None :
                        flags[int(mux_r),int(mux_c),int(err-1)] = err

                    else :
                        good_count += 1
                        # plots points in transition
                        if saveplots :
                            plt.scatter(lc.tes_r[m_chng],(lc.tes_p_masked*1e12)[m_chng],color=colors[mux_r]) #figure 1
                            plt.figure(2)
                            plt.plot(lc.bias_i*1e6, lc.tes_i*1e6, label='Det # %s'%(mux_r),c=(colors[mux_r])) # this is the original data array (full load curve)
                            plt.scatter((lc.bias_i*1e6)[m_chng],(lc.tes_i*1e6)[m_chng],color=(colors[mux_r]))

                        if len(m_chng) == 20 :
                            #assigning each data point to its corresponding row and colum numbers
                            bias_x[int(mux_r),int(mux_c),:] = lc.tes_r[m_chng]
                            bias_y[int(mux_r),int(mux_c),:] = (lc.tes_p_masked*1e12)[m_chng]
                            current_x[int(mux_r),int(mux_c),:] = (lc.bias_i*1e6)[m_chng]
                            current_y[int(mux_r),int(mux_c),:] = (lc.tes_i*1e6)[m_chng]
                        else :
                            if len(m_chng) < 2 :
                                print('not enough data to interpolate!')
                                flags[int(mux_r),int(mux_c),4] = 5
                            else :
                                #fill in the data arrays if there are <20 data points
                                f = interpolate.interp1d((lc.tes_r)[m_chng], (lc.tes_p_masked*1e12)[m_chng])
                                xnew = np.linspace((lc.tes_r)[m_chng][0],(lc.tes_r)[m_chng][-1] , num = 20)
                                ynew = f(xnew)
                                #saving the new 20 point data array for later
                                bias_x[int(mux_r),int(mux_c),:] = xnew
                                bias_y[int(mux_r),int(mux_c),:] = ynew

                                t = interpolate.interp1d((lc.bias_i*1e6)[m_chng], (lc.tes_i*1e6)[m_chng])
                                cxnew = np.linspace((lc.bias_i*1e6)[m_chng][0],(lc.bias_i*1e6)[m_chng][-1] , num = 20)
                                cynew = t(cxnew)
                                current_x[int(mux_r),int(mux_c),:] = cxnew
                                current_y[int(mux_r),int(mux_c),:] = cynew

                else :
                    print('temp %s for det r = %s , c = %s doesnt exist' %(temp,mux_r,mux_c))
                    flags[int(mux_r),int(mux_c),1] = 2

                if int(mux_c) == col :
                    plt.figure(100+col)
                    if np.mean(lc.tes_i*1e6) != 0.0 :
                        plt.plot(lc.bias_i*1e6, lc.tes_i*1e6, color=colors[mux_r])
                    plt.figure(200+col)
                    plt.plot(lc.tes_r, lc.tes_p_masked*1e12, color=colors[mux_r])

                if saveplots :
                    plt.figure(1)
                    plt.ylabel("TES Power [pW]")
                    plt.xlabel("Resistance [ohm]")
                    plt.title("Detector x%02if%02i (c%02ir%02i)" % (det_x[i],det_f[k],mux_c,mux_r))
                    plt.legend(loc='best')
                    plt.savefig('plots/test_res_vs_power_r:%i_c:%i.png' %(mux_r,mux_c))
                    plt.clf()

                    plt.figure(2)
                    plt.ylabel("TES Current [$\mu$A]")
                    plt.xlabel("Bias Current [$\mu$A]")
                    plt.title("Detector x%02if%02i (c%02ir%02i)" % (det_x[i],det_f[k],mux_c,mux_r))
                    plt.legend(loc='best')
                    plt.savefig('plots/test_bias_vs_tes_r:%i_c:%i.png' %(mux_r,mux_c))
                    plt.clf()

            except NotImplementedError :
                flags[int(mux_r),int(mux_c),0] = 1

    powerx,powery,biasx,biasy = find_mode_limits(bias_x[:,col,:],bias_y[:,col,:],current_x[:,col,:],current_y[:,col,:],col=col,temp=temp)

    #assigning and labeling Detectors to different colors in one column
    plt.figure(100+col)
    for i in range(0, 33):
        if np.mean(current_x[i,col,:]) != 0.0 :
            plt.scatter(current_x[i,col,:], current_y[i,col,:], s=50.0, marker='*',color=colors[i],label='Det # %s'%(i))

    plt.ylabel("TES Current [$\mu$A]")
    plt.xlabel("Bias Current [$\mu$A]")
    plt.title('Column %s Detectors @ %s K (Bias)'%(col,temp))
    plt.ylim(min(biasy)-15,max(biasy)+15)
    plt.xlim(min(biasx)-85,max(biasx)+85)
    plt.legend(ncol=1, loc='upper left', bbox_to_anchor=(0.98, 1.03), labelspacing=0.0, prop=fontP, shadow=True)
    plt.savefig('plots/final_bias_color_%s_%sK.png'%(col,temp))
    plt.clf()

    plt.figure(200+col)
    for i in range(0, 33):
        plt.scatter(bias_x[i,col,:], bias_y[i,col,:], s=50.0, marker='*',color=colors[i],label='Det # %s'%(i))

    plt.ylabel("TES Power [pW]")
    plt.xlabel("Resistance [ohm]")
    plt.title('Column %s Detectors @ %s K (Power)'%(col,temp))
    plt.ylim(min(powery)-10.0,max(powery)+10)
    plt.xlim(min(powerx)-0.05,max(powerx)+0.05)
    plt.legend(ncol=1, loc='upper left', bbox_to_anchor=(0.98, 1.03), labelspacing=0.0, prop=fontP, shadow=True)
    plt.savefig('plots/final_power_color_%s_%sK.png'%(col,temp))
    plt.clf()

    # save the start and stop of bias transition to numpy file
    # np.save('bias_start_list.npy',bias_list,allow_pickle=True) # Note: units are in micro Amperes
    # np.save('bias_flags.npy',flags)
    return good_count

 # finding the limits of the optimal x values:
def find_mode_limits(xvals,yvals,cxvals,cyvals,col=None,temp=0):

    xvalues = []
    yvalues = []
    cxvalues = []
    cyvalues = []
    count = 0
    try:
        for i in range(len(xvals)):
            if xvals[i][0] != 0.0 and xvals[i][-1] != 0.0 :
                xvalues.extend(list(xvals[i]))
                yvalues.extend(list(yvals[i]))
                cxvalues.extend(list(cxvals[i]))
                cyvalues.extend(list(cyvals[i]))
                count += 1
            else :
                 xvals[i][xvals[i] == 0] = np.nan
                 cxvals[i][cxvals[i] == 0] = np.nan

        print('num of good det : ' ,count)

        # removing outliers
        mean = np.mean(yvalues)
        std = np.std(yvalues)
        dist_from_mean = abs(yvalues - mean)
        max_dev = 3
        not_outlier = dist_from_mean < max_dev * std

        for k in range(len(yvalues)):
            if not_outlier[k] == False :
                yvalues[k] = np.nan
                xvalues[k] = np.nan

        x_sampler = np.linspace(np.nanmin(xvalues),np.nanmax(xvalues),num=100)

        modes = []
        maxcount = 0
        counts = [0]*100

        for j in range(len(x_sampler)):
            for l in range(len(xvals)):
                if xvals[l][0] != np.nan:
                    if x_sampler[j] > np.nanmin(xvals[l]) and x_sampler[j] < np.nanmax(xvals[l]): #xvals[l] each detector
                        counts[j] += 1 #counting how many intersections
                    if counts[j] > maxcount:
                        maxcount = counts[j]

        for k in counts:
            if k == maxcount:
                 modes.append(k)

        min_x = np.nanmin(xvalues)
        max_x = np.nanmax(xvalues)

        num_det = np.max(counts)
        mask = np.where(counts == num_det)[0]
        limits = x_sampler[mask]

        n_mask = np.logical_and(xvalues < limits[-1],xvalues > limits[0]) # saves the indices from xvals that match the boolean statements
        cxvalues = np.array(cxvalues)[n_mask]
        cyvalues = np.array(cyvalues)[n_mask]

        print ("First and Last optimal values: " +  str(limits[0]) + str(limits[-1]))
        print ("Central optimal value: " + str((limits[0] + limits[-1])/2))
        # plotting the actual points as scatter plot
        plt.figure(4)
        plt.scatter(xvalues, yvalues, color = "m", marker = "o", s = 30)
        #creates region of optimal x values
        plt.axvspan(limits[0], limits[-1], alpha = 0.5)
        plt.axvline((limits[0] + limits[-1])/2, color = 'r')
        plt.xlabel('Resistance [ohm]')
        plt.ylabel('TES Power [pW]')
        plt.title('Optimal Transition Column %s' %(col))
        plt.savefig('plots/optimal_intersect_%s_%sK.png'%(col,temp))
        plt.clf()

    except ValueError:
        print('Empty Arrays')

    return np.array(xvalues)[n_mask],np.array(yvalues)[n_mask],cxvalues,cyvalues #powerx,powery,biasx,biasy

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

def save_final_im(good_count,dir=None,type='power'):

    rem = good_count % 10
    num_rows = int(good_count / 10)

    list_im = []
    if type == 'bias' :
        images = 'test_bias_vs_tes'
    else :
        images = 'test_res_vs_power'

    for file in os.listdir(dir):
        if images in file :
            list_im.append(file)

    if rem != 0 :
        # makes dummy white graphs for filler when we don't have even rows
        test_im = Image.open(list_im[0])
        plt.gca().set_axis_off()
        cmap = plt.cm.OrRd
        cmap.set_under(color='white')
        plt.imshow(np.zeros(test_im.size),cmap=cmap,vmin=0.1,vmax=1.0)
        for i in range(10 - rem):
            plt.savefig(dir + 'test' + type + '_fake_%s.png'%(i))
            plt.clf()
            list_im.append(dir + 'test' + type + '_fake_%s.png'%(i))

    l = 0
    imgs_horz = []
    for j in range(0,len(list_im),10):
        imgs = [Image.open(i) for i in list_im[j:j+10]]

        # pick the image which is the smallest, and resize the others to match it
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[-1][1]
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

        # save that beautiful picture
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(dir + 'test_' + type + '_comp_%s.png'%(l))
        imgs_horz.append(dir + 'test_' + type + '_comp_%s.png'%(l))
        l += 1

    # vertically stack the horizontal saved images we made
    imgs_h = [Image.open(i) for i in imgs_horz]
    max_shapeh = sorted([(np.sum(i.size), i.size) for i in imgs_h])[-1][1]
    imgs_combh = np.vstack(imgs_h)
    imgs_combh = Image.fromarray(imgs_combh)
    imgs_combh.save(dir + 'test_' + type + '_final.png')

    # go through and delete all of the files that we no longer need
    for s in imgs_horz:
        os.remove(s)

    #delete originals
    for m in list_im :
        os.remove(m)

if __name__ == '__main__' :
    temperatures = [77,293]
    fnames = ['/home/time_user/time_analysis/py/timefpu/sample-load-curves/iv_opteff_77K_run1','/home/time_user/time_analysis/py/timefpu/sample-load-curves/iv_opteff_300K_run1']
    det_x = np.arange(32)
    det_f = np.arange(33)
    saveplots = 0
    #for temp in range(len(temperatures)):
        # for col in range(32):
        #     good_count = read_load_curve(fnames,det_x,det_f,saveplots=saveplots,col=col,temp=temp)

    ''' ########################## THIS IS FOR TESTING 1 COL AT A TIME ##############################'''
    # you have to change fnames to the same index as temps
    good_count = read_load_curve(fnames[1],det_x,det_f,saveplots=saveplots,col=23,temp=temperatures[1])
    ''' ############################################################################################# '''

    if saveplots :
        save_final_im(good_count,dir='/home/time_user/time_analysis/py/timefpu/plots',type='bias')
        save_final_im(good_count,dir='/home/time_user/time_analysis/py/timefpu/plots',type='power')
