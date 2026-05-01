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
from scipy import interpolate
import pdb
from math import *
from termcolor import colored

def read_load_curve(fname,cols,rows,saveplots=0,temp=0):

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
    for col in cols:
        # generate figure objects to add each detector to
        # fig3 = plt.figure(100 + col, figsize=(20,8)) # I-V plot
        # ax3 = fig3.add_subplot(111)
        # fig2 = plt.figure(200 + col, figsize=(20,8)) # R-P plot
        # ax2 = fig2.add_subplot(111)

        for row in rows:
            try :
                print(colored(f'col:{col} , row:{row}','yellow'))
                lc = all_lc[col][row]
                # masking data less than zero on the y axis RP curve
                tes_p = np.array(lc.tes_p_masked * 1e12)
                tes_r = np.array(lc.tes_r)[tes_p > 0]
                bias_i = np.array(lc.bias_i * 1e6)[tes_p > 0]
                tes_i = np.array(lc.tes_i * 1e6)[tes_p > 0]
                tes_p = tes_p[tes_p > 0]

                if len(tes_r) != 0 :
                    # provide both sets of y data for filtering useless data
                    m_chng, err = turn(tes_p,tes_r,tes_i,bias_i,c=col,r=row,color=colors) #grabs data where transition is

                    if m_chng == None :
                        flags[int(row),int(col),int(err-1)] = err

                    else :
                        good_count += 1
                        # individual detector plots
                        if saveplots :
                            # fig = plt.figure(num = 'r-p')
                            # ax = fig.add_subplot(111)
                            # ax.set_title(f'R-P c{col}r{row}, Temp {temp}')
                            # ax.plot(tes_r, tes_p, label='Det # %s'%(row),color='black')
                            # ax.scatter(tes_r[m_chng],(tes_p)[m_chng],color='green')
                            # ax.set_ylabel("TES Power [pW]")
                            # ax.set_xlabel("Resistance [ohm]")
                            # ax.legend(loc='best')
                            # fig.savefig(fname + f'/plots/res_vs_power_r{row}_c{col}.png')
                            # # plt.show()
                            # plt.close(fig)

                            if np.mean(tes_i) != 0.0 :
                                print('good detector')
                                # ax3.plot(bias_i, tes_i, color=colors[row]) # plot cummulative figure first

                                # fig1 = plt.figure(num = 'i-v')
                                # ax1 = fig1.add_subplot(111)
                                # ax1.set_title(f'I-V c{col}r{row}, Temp {temp}')
                                # ax1.plot(bias_i, tes_i, label='Det # %s'%(col),color='black')
                                # ax1.scatter((bias_i)[m_chng],(tes_i*1e6)[m_chng],color='green')
                                # # plt.ylim(-500,50)
                                # ax1.set_ylabel("TES Current [$\mu$A]")
                                # ax1.set_xlabel("Bias Current [$\mu$A]")
                                # ax1.legend(loc='best')
                                # fig1.savefig(fname + f'/plots/bias_vs_tes_r{row}_c{col}.png')
                                # # plt.show()
                                # plt.close(fig1)

                        # ax2.plot(tes_r, tes_p, color=colors[row])

                        if len(m_chng) == 20 :
                            #assigning each data point to its corresponding row and colum numbers
                            bias_x[int(row),int(col),:] = (tes_r)[m_chng]
                            bias_y[int(row),int(col),:] = (tes_p)[m_chng]
                            current_x[int(row),int(col),:] = (bias_i)[m_chng]
                            current_y[int(row),int(col),:] = (tes_i)[m_chng]
                        else :
                            if len(m_chng) < 2 :
                                print('not enough data to interpolate!')
                                flags[int(row),int(col),4] = 5
                            else :
                                #fill in the data arrays if there are <20 data points
                                f = interpolate.interp1d((tes_r)[m_chng], (tes_p*1e12)[m_chng])
                                xnew = np.linspace((tes_r)[m_chng][0],(tes_r)[m_chng][-1] , num = 20)
                                ynew = f(xnew)
                                #saving the new 20 point data array for later
                                bias_x[int(row),int(col),:] = xnew
                                bias_y[int(row),int(col),:] = ynew

                                t = interpolate.interp1d((bias_i)[m_chng], (tes_i)[m_chng])
                                cxnew = np.linspace((bias_i)[m_chng][0],(bias_i)[m_chng][-1] , num = 20)
                                cynew = t(cxnew)
                                current_x[int(row),int(col),:] = cxnew
                                current_y[int(row),int(col),:] = cynew

                else :
                    print('temp %s for det r = %s , c = %s doesnt exist' %(temp,row,col))
                    flags[int(row),int(col),1] = 2

            except NotImplementedError :
                print('Not Implemented Error')
                flags[int(row),int(col),0] = 1

        powerx,powery,biasx,biasy,opt_num = find_mode_limits(bias_x[:,col,:],bias_y[:,col,:],current_x[:,col,:],current_y[:,col,:],colors,col=col,row=rows,temp=temp)

        if powerx == 'Error':
            print('No good detectors found, moving on to the next column...')
            return 0,0,0,0

        else :
            print ("Range of optimal CURRENT values: " +  str(min(biasx)) + " and " + str(max(biasx)))
            print ("Optimal CURRENT value: " + str(max(biasx)))
            print(colored('good count : %s'%(good_count),'magenta'))

            # if saveplots :

            #     #assigning and labeling Detectors to different colors in one column
            #     for i in range(0, 33):
            #         if np.mean(current_x[i,col,:]) != 0.0 :
            #             ax3.scatter(current_x[i,col,:], current_y[i,col,:], s=50.0, marker='*',color=colors[i],label='Det # %s'%(i))
            #             ax2.scatter(bias_x[i,col,:], bias_y[i,col,:], s=50.0, marker='*',color=colors[i],label='Det # %s'%(i))

            #     ax3.set_ylabel("TES Current [$\mu$A]")
            #     ax3.set_xlabel("Bias Current [$\mu$A]")
            #     ax3.set_title('Column %s Detectors @ %s K (Bias)'%(col,temp))
            #     ax3.set_ylim(np.min(biasy)-15,np.max(biasy)+15)
            #     ax3.set_xlim(np.min(biasx)-85,np.max(biasx)+85)
            #     ax3.axvspan(np.min(biasx), np.max(biasx), alpha = 0.5)
            #     ax3.axvline(np.max(biasx), color = 'r')
            #     ax3.legend(ncol=2,loc='upper left', labelspacing=0.0, shadow=True)
            #     fig3.savefig(fname + f'/plots/final_bias_color_{col}_{temp}K.png')

            #     ax2.set_ylabel("TES Power [pW]")
            #     ax2.set_xlabel("Resistance [ohm]")
            #     ax2.set_title('Column %s Detectors @ %s K (Power)'%(col,temp))
            #     ax2.set_ylim(min(powery)-10.0,max(powery)+10)
            #     ax2.set_xlim(min(powerx)-0.05,max(powerx)+0.05)
            #     ax2.axvspan(min(powerx), max(powerx), alpha = 0.5)
            #     ax2.axvline(np.mean(powerx), color = 'r')
            #     ax2.legend(ncol=2,loc='upper left', labelspacing=0.0, shadow=True)
            #     fig2.savefig(fname + f'/plots/final_power_color_{col}_{temp}K.png')
            #     # plt.show()
            #     fig2.clf()
            #     fig3.clf()
            #     plt.close()

    return #good_count, np.max(powerx), np.max(biasx), opt_num

 # finding the limits of the optimal x values:
def find_mode_limits(xvals,yvals,cxvals,cyvals,colors,col=None,row=None,temp=0):

    count = 0
    try: # figure out how many good detectors there are with actual data
        for i in range(len(xvals)):
            if np.mean(xvals[i]) == 0.0 :
                xvals[i][:] = np.nan
                yvals[i][:] = np.nan
                cxvals[i][:] = np.nan
                cyvals[i][:] = np.nan
            else:
                count += 1

        print('num of good det : ' ,count)

        if count == 0 :
            return 'Error','Error','Error','Error', 'Error' # 5 error statements for powerx,powery,biasx,biasy,opt_num outputs

        else : # if we have a good detector ... 

            # removing outliers
            mean = np.nanmean(yvals)
            std = np.nanstd(yvals)
            dist_from_mean = abs(yvals - mean)
            max_dev = 1
            not_outlier = dist_from_mean < max_dev * std

            for k in range(len(yvals)):
                for l in range(len(yvals[0])):
                    if not_outlier[k,l] == False :
                        yvals[k,l] = np.nan
                        xvals[k,l] = np.nan

            x_sampler = np.linspace(np.nanmin(xvals),np.nanmax(xvals),num=100) #same data set just outliers removed

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

            min_x = np.nanmin(xvals)
            max_x = np.nanmax(xvals)

            num_det = np.max(counts)
            mask = np.where(counts == num_det)[0]
            limits = x_sampler[mask]
            n_mask = np.logical_and(xvals.flatten() < limits[-1],xvals.flatten() > limits[0]) # saves the indices from xvals that match the boolean statements
            n_mask = n_mask.reshape(33,20)
            cxvals = np.array(cxvals)[n_mask]
            cyvals = np.array(cyvals)[n_mask]


            opt_num = 0
            for h in range(len(xvals)):
                if np.any(n_mask[h][:]) == True :
                    opt_num += 1

            print ("Range of optimal values: " +  str(limits[0] ) + str(limits[-1]))
            print ("Optimal value: " + str(limits[-1]))

            # plotting the actual points as scatter plot
            # m_fig = plt.figure(num = 'power limits')
            for i in row:
                if xvals[i][0] != np.nan:
                    plt.plot(xvals[i,:],yvals[i,:],color=colors[i] ,label=f'c{col}:r{i}')
            plt.scatter(cxvals, cyvals, color = "m", marker = "o", s = 30,label='fitted transition')
            # #creates region of optimal x values
            # plt.axvspan(limits[0], limits[-1], alpha = 0.5)
            # plt.axvline((limits[-1]), color = 'r')
            plt.xlabel('Resistance [ohm]')
            plt.ylabel('TES Power [pW]')
            plt.legend(ncol=2,loc='best', labelspacing=0.0, shadow=True)
            plt.title('Optimal Transition Column %s' %(col))
            plt.show()
            plt.clf()
            # m_fig.savefig('/Volumes/KINGSTON/20231019/20231019a_iv/plots/optimal_intersect_%s_%sK.png'%(col,temp))

            print(colored('opt_num: %s'%(opt_num), 'magenta'))
            return np.array(xvals)[n_mask],np.array(yvals)[n_mask],cxvals,cyvals,opt_num #powerx,powery,biasx,biasy

    except ValueError:
        print('Empty Arrays')

def turn(y,x,bias_y,bias_x,c=0,r=0,color=None): #checks to see if loadcurve data looks reasonable
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
        print(e[1])
        slope_error = e[0]
        
        plt.plot(xvals[i,:],yvals[i,:],color=colors[i] ,label=f'c{c}:r{r}')
        plt.scatter()
        plt.plot(x,y,color=color[r])

        if intercept_error > 1.0 : # if my data isn't a line, how different from a line is it?
            # I made that number up from looking at data
            m = np.diff(y) #finding the slope
            out = np.where(np.logical_and(np.greater(m,1.0),np.less(m,2.0))) # where is the slope between those two numbers?
            out = [list(k) for k in out][0]

            if len(out) > 20 :
                return out[0:19], None #trims data array to only 20 points
            else :
                return out, None
        else :
            print('power data not a line')
            return None, 4
    else :
        print('mean of zero')
        return None, 3

def save_final_im(good_count,dir=None,type='power'):

    rem = good_count % 10
    num_rows = int(good_count / 10)

    list_im = []
    if type == 'bias' :
        images = 'bias_vs_tes'
    else :
        images = 'res_vs_power'

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
            plt.savefig(dir + type + '_fake_%s.png'%(i))
            plt.clf()
            list_im.append(dir + type + '_fake_%s.png'%(i))

    l = 0
    imgs_horz = []
    for j in range(0,len(list_im),10):
        imgs = [Image.open(i) for i in list_im[j:j+10]]

        # pick the image which is the smallest, and resize the others to match it
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[-1][1]
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

        # save that beautiful picture
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(dir + type + '_comp_%s.png'%(l))
        imgs_horz.append(dir + type + '_comp_%s.png'%(l))
        l += 1

    # vertically stack the horizontal saved images we made
    imgs_h = [Image.open(i) for i in imgs_horz]
    max_shapeh = sorted([(np.sum(i.size), i.size) for i in imgs_h])[-1][1]
    imgs_combh = np.vstack(imgs_h)
    imgs_combh = Image.fromarray(imgs_combh)
    imgs_combh.save(dir + type + '_final.png')

    # go through and delete all of the files that we no longer need
    for s in imgs_horz:
        os.remove(s)

    #delete originals
    for m in list_im :
        os.remove(m)

if __name__ == '__main__' :
    import calib.time202109 as calib
    bias_list = np.zeros((32,4))
    temperatures = [77]
    # fnames = ['/home/time_user/time_analysis/py/timefpu/sample-load-curves/iv_opteff_77K_run1','/home/time_user/time_analysis/py/timefpu/sample-load-curves/iv_opteff_300K_run1']
    fnames = ['/Volumes/KINGSTON/20231019/20231019a_iv']
    saveplots = 1

    # rcs = [(4,14),(4,15),(4,0),(4,19),(4,11),(4,10),(5,4),(5,5),(5,11),(5,12),(13,1),(13,12),\
    #     (13,14),(14,11),(14,24),(14,20),(14,12),(14,15),(14,5),(14,17),(14,4),(14,1),(15,25),\
    #         (15,17),(15,11),(15,15),(15,5),(15,12),(15,4),(15,1),(15,16),(16,20),(20,16),(20,17),\
    #             (20,22),(20,21),(20,23),(21,17),(21,24),(21,20),(21,11),(21,16),(21,18),(21,12),(22,23),\
    #                 (22,20),(22,12),(22,22),(24,18),(24,16),(24,24)]

    cols = [4]
    rows = [14,15,0,19,11,10]

    for t in range(len(temperatures)):
        read_load_curve(fnames[0],cols,rows,saveplots=saveplots,temp=temperatures[t])
        # good_count, power_max, bias_max, opt_num = read_load_curve(fnames[t],cols,rows,saveplots=saveplots,temp=temperatures[t])
        # bias_list[col,:] = [good_count,bias_max/1e6/calib.CAL_TES_BIAS_I,power_max, opt_num]
        # np.save('bias_list_%s.npy'%(temperatures[t]),bias_list,allow_pickle=True)

    # if saveplots :
    #     save_final_im(good_count,dir='/Volumes/KINGSTON/20231019/20231019a_iv/plots',type='bias')
    #     save_final_im(good_count,dir='/Volumes/KINGSTON/20231019/20231019a_iv/plots',type='power')
