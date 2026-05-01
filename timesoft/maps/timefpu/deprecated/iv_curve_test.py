#!/usr/bin/env python

import os,sys
import argparse
import time

import numpy as np
import pylab as pl

from pymce import MCE

DEBUG = False

class MCEWrap():
    def __init__(self):
        self.m = MCE()
    def read(self,x,y):
        v = self.m.read(x,y)
        print "rb %s %s = %s"%(x,y,str(v))
        return v
    def write(self,x,y,v):
        print "wb %s %s %s"%(x,y,str(v))
        if not DEBUG:
            self.m.write(x,y,v)

def main(d,temp,calib,bias_cols=None,bias_start=2000,bias_step=-2,bias_count=1001,zap_bias=30000,zap_time=1.0,settle_time=10.0,settle_bias=2000.0,bias_pause=0.1,bias_final=0,data_mode=1): # add starting arguments as variables in function call

    data_name = d #iv_test0 or whatever file name you want to call it
    m = MCEWrap()

    mas_data = os.environ['MAS_DATA'] #  /data/cryo/current_data
    dname = os.path.join(mas_data,data_name)
    fname = os.path.join(dname,data_name)
    if not os.path.exists(dname):
        os.mkdir(dname)
        print "created directory "+dname
    else:
        print "directory already exists, aborting"
        exit(1)

    flog = os.path.join(dname,'lc_ramp_tes_bias_log.txt')
    print "logging ramp parameters to "+flog
    f = open(flog,'w')
    print>>f,"Parameters for this run"
    print>>f,"bias_start=",bias_start
    print>>f,"bias_step=",bias_step
    print>>f,"bias_count=",bias_count
    print>>f,"zap_bias=",zap_bias
    print>>f,"zap_time=",zap_time
    print>>f,"settle_time=",settle_time
    print>>f,"bias_pause=",bias_pause
    print>>f,"bias_final=",bias_final
    print>>f,"data_mode=",data_mode
    f.close()

    print "Setting up MCE mode"
    m.write('rca','data_mode',data_mode)
    m.write('rca','en_fb_jump',1)
    m.write('rca','flx_lp_init',1)
    ncol = len(m.read('tes','bias'))

    if bias_cols is None:
        bias_cols = np.arange(ncol) # this has been making columns list by default probably
        # make a list from 0 to len(live columns)

    bias_mask = np.zeros(ncol,dtype=np.int32)
    for c in bias_cols:
        bias_mask[c] = 1

    runfile = fname + '.run'
    biasfile = fname + '.bias' # tells MCE what to set the bias levels to
    logfile = fname + '.log'

    ''' Jon what do these commands do? '''
    os.system('mce_status > '+runfile)
    os.system('frameacq_stamp %s %s %d >> %s'%('s',fname,bias_count,runfile))
    biasf = open(biasfile,'w')
    print>>biasf,"<tes_bias>"

    print "zapping"
    zap_arr = bias_mask * zap_bias # 1*30,000 or a 0
    ''' #################################### '''
    # bias_start_arr = bias_mask * bias_start
    ''' #################################### '''
    m.write('tes','bias',zap_arr)
    time.sleep(zap_time) # 0.1 seconds to settle down
    print "settling"
    m.write('tes','bias',bias_mask*settle_bias) # settle bias = 2000
    time.sleep(settle_time) # wait 10 seconds...


    ''' What is the point of starting the bias at 2000 before looping through the array?'''

    m.write('tes','bias',bias_mask*bias_start) # starts at 2000 , steps down by 2
    time.sleep(0.1)
    m.write('rca','flx_lp_init',1) # dunno what this is for either...
    time.sleep(0.1)

    bscript = os.path.join(dname,'bias_script.scr')
    b = open(bscript,'w')
    print>>b,'acq_config %s rcs'%fname

    ''' read in numpy array of bias start/stop values
    Note : Bias values are in Amperes
    '''
    #read_load_curve file...
    bias_file = np.load('/home/time/time_analysis/py/timefpu/bias_list_%s.npy'%(temp),allow_pickle=True) # shape = (32,3)
    # [good_count,bias_max,resistance_max] => for each col
    bias_file = bias_file/calib.CAL_SQ1_FB_I/1e6
    print(bias_file)
    exit()

    #This is where we actually start step biasing
    bias_inc = np.zeros((32,bias_count))
    for j in range(32):
        if bias_mask[j] == 1 and np.sum(bias_file[j]) != 0: # don't bias masked detectors
            stop = bias_file[j][1]
            bias_inc[j,:] = np.linspace(bias_start,stop,bias_count)
        else :
            bias_inc[j,:] = np.linspace(bias_start,bias_start,bias_count)

    for i in range(bias_count):
        bias = bias_inc[:,i] # grab every column for which step we are on
        print>>biasf,list(bias)
        bias_str = ' '.join([str(x) for x in bias*bias_mask])
        print>>b,'wb tes bias '+bias_str
        print>>b,'sleep %d'%(bias_pause*1e6) #0.1 seconds * 1e6 = 100,000 so close to 3 hours!
        print>>b,'acq_go 1'

    # for i in range(bias_count):
    #     bias = bias_step*i + bias_start # -2 * 0 ... -2 * 1 ... -2 * 2 ... (-0, -2, -4 ...)
        # 2000, 1998, 1996 ...
    #     print>>biasf,bias
    #     bias_str = ' '.join([str(x) for x in bias*bias_mask])
    #     print>>b,'wb tes bias '+bias_str
    #     print>>b,'sleep %d'%(bias_pause*1e6) #0.1 seconds * 1e6 = 100,000 so close to 3 hours!
    #     print>>b,'acq_go 1'

    biasf.close()
    b.close()
    t0 = time.time()
    print "executing bias ramp"
    if not DEBUG:
        os.system('mce_cmd -qf '+bscript)
    t1 = time.time()
    print "ramp finished in %.2f seconds"%(t1-t0)

    m.write('tes','bias',bias_mask*bias_final)
    m.write('rca','flx_lp_init',1)

if __name__=='__main__':
    import calib.time202001 as calib
    main('partial_load_test','293', calib)
