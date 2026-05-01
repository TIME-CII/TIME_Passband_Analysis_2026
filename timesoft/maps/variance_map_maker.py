import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from scipy.optimize import curve_fit
from astropy.modeling.models import BlackBody
from timesoft.timestream.timestream_tools import Timestream
from timesoft.maps.absolute_cal import absolute_calibration
from timesoft.maps.map_tools import Map
from timesoft.calibration import Offsets
from timesoft.calibration import setup_simple_atm_model
from astropy.coordinates import get_body
from astropy.coordinates import SkyCoord, EarthLocation, TETE, ICRS, AltAz
from astropy.time import Time
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.coordinates as apcoord
import sys 
from timesoft.Utilities.plot_map import plot_maps
import astropy.units as u
from timesoft.calibration import DetectorConstants
from timesoft.helpers.nominal_frequencies import *
from matplotlib import cm
import matplotlib.colors as colors
import os
import datetime
from timesoft.helpers.get_tau_forecast import get_tau
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import get_cmap
import matplotlib
from astropy.coordinates import solar_system_ephemeris
solar_system_ephemeris.set('jpl')
matplotlib.use('Agg')

class Weighted_Maps_Handler():
    def __init__(self, path, obj_override='None', scan_override = 'None', save_path='/data/vaughan/inverse_variance_weighted_maps/maps_testing/',debugging_plots=False,thresh=100,n_edge_start=4,n_edge_finish=4,n_scan_start=50,n_scan_finish=50,filter_deg=5,pixel=(0.0055,0.0055),tau_path='/data/vaughan/tau_forecasts/', verbose=False, frame='J2000_hard', beamtype='Gauss', mc=0, version='2022.dev.1', good_xf=None):
        """ Initializer for handling weighted maps
        
        Parameters
        ----------
        path : str 
        The path to the netcdf files for the observation we want to analyze 
        obj_override : str
        over rides the object name of our observation, this is necessary for engineering data as meta 
        data failed to be properly uploaded to the netcdffiles. 
        scan_override : scan direction of the instrument e.g. RA or DEC, in the future this will be part of the meta data, but
        can be used for engineering data 
        save_path : pass to save data
        debugging plots : bool
        True saves debugging plots to the save_path directory (so make sure save_path exists!), default is False

        thresh, n_edge_start, n_edge_finish, n_scan_start, n_scan_finish, filter_deg, and pixel are all defined in detail in the timestream object.
        and are passed directly to a new instance of that object when initializing this one. 
        tau_path - the directory where the atmosphere opacity data is stored 
        verbose - depreciated 
        frame - coordinate system to transform to, example would be J200_hard or J200_soft, the soft version will transform only the data 
        the hard version will transform the data and the backup copy of the data that timestream makes as a save state. Note that the frame must be set to "hard"
        because setup_masked_filter() relies on reseting the data. 
        
        beamtype : what type of beam to fit to planet data 

        Returns
        -------
        None
        """
        # print(pixel, 'pixel')
        #### NOTE NEED TO CHANGE GOOD XF FROM NONE TO SOMETHING THAT DEFAULTS ALL 
        version = '2024.dev.1'
        print('HARD CODED TO 2024.dev.1')

        print(mc, 'mc')
        if good_xf is not None:
            self.ts = Timestream(path, mc=mc, store_copy=True,impose_frame=frame, version=version, xf=good_xf)# xf=[(0,5),(0,15),(0,45),(7,49)])
        else:
            self.ts = Timestream(path, mc=mc, store_copy=True,impose_frame=frame, version=version)#, xf=good_xf)# xf=[(0,5),(0,15),(0,45),(7,49)])

        # self.cross_power_all(self.ts.data, dt=1/100)

        # self.mean_cross_power(self.ts.data, dt=1/100)
        # print(self.ts.header)
        # exit()


        # print(good_xf)

        # print(self.ts.header['xf_coords'])

        nancheck = np.all(np.isnan(self.ts.data),axis=1)
        # print(nancheck, 'nan_check')
        zerocheck = np.all(self.ts.data==0,axis=1)
        idx = np.nonzero((~nancheck) & (~zerocheck))[0].astype('int')
        det_idx = self.ts.header['xf_coords'][idx]
        good_idx = []
        self.ts.restrict_detectors(det_idx, det_coord_mode='xf')

        # fig, axs = plt.subplots(1)
        # axs.scatter(self.ts.ra, self.ts.dec)
        # axs.set_xlabel('Right Ascension [deg]')
        # axs.set_ylabel('Declination [deg]')
        # fig.savefig(save_path + 'ra_dec')

        # if not os.path.isdir(save_path + 'raw_ts/'):
                # os.makedirs(save_path + 'raw_ts')
        # for xf in (self.ts.header['xf_coords']):
        #     print(xf)
        #     idx = self.ts.get_xf(xf[0],xf[1])
        #     data = self.ts.data[idx]
        #     std = np.nanstd(data)
        #     if std < 300:
        #         good_idx.append(idx)
        #         fig, axs = plt.subplots(1)
        #         time = self.ts.t
        #         # axs.plot(time,self.ts.ra)
        #         axs.plot(time, data)
        #         axs.set_xlabel('unix time')
        #         axs.set_ylabel('Det Counts')
        #         fig.savefig(save_path + 'raw_ts/x%s_f%s_p%s' % (xf[0],xf[1],mc))
        #         plt.close(fig)


        # fig, axs = plt.subplots(1,2,figsize=(12,4))
        # axes[0].set_xlabel('t')
        # axes[0].set_ylabel('az')
        # axes[1].set_xlabel('t')
        # axes[1].set_ylabel('el')
        # axes[0].plot(ts.t, ts.az, label='telescope')
        # axes[1].plot(ts.t, ts.el, label='telescope')
        # from astropy.coordinates import EarthLocation, AltAz, Latitude, Longitude, get_body
        # import astropy.units as u 
        # from astropy.time import Time
        # kitt_peak = EarthLocation(latitude=-111-35/60-48/3600*u.deg,longitude=31 + 57 /60 + 30/3600 )
            # else: 
            #     pass


        # exit()
        # det_idx = self.ts.header['xf_coords'][np.array(good_idx)]
        # self.ts.restrict_detectors(det_idx, det_coord_mode='xf')



        # fig, axs = plt.subplots(1)
        # print(time.shape, self.ts.data.shape)
        
        # print(self.ts.data.T.shape)
        # detrend = np.nanmedian(self.ts.data.T,axis=0)
        # axs.plot(time, self.ts.data.T - detrend, 'k-')
        # axs.set_xlabel('Unix Time')
        # axs.set_ylabel('Det Counts')
        # fig.savefig(save_path + 'raw_ts/All Dets_p%s' % mc)
        # plt.close(fig)
        # exit()

        # plt.close('all')
        self.mc = str(int(mc)) 

        # self.ts = Timestream(path, mc=mc, store_copy=True,impose_frame=frame, version=version, xf=[(0,5),(0,15),(0,45),(7,49)])
        
        if obj_override != 'None':
            self.ts.header['object'] = obj_override.casefold()

        ### use the filename, which is a unix timestamp, to estimate when the observation took place and then 
        ### interpolate the atmospheric opacity during that time 
        timestamp = int(os.path.basename(path))
        dtime = datetime.datetime.fromtimestamp(timestamp)
        date_str = dtime.strftime("%Y%m%d")
        self.ts.header['date'] = date_str
        self.ts.header['timestamp'] = timestamp
        timestamp = int(os.path.basename(path))
        dtime = datetime.datetime.fromtimestamp(timestamp)
        date_str = dtime.strftime("%Y%m%d"); time_str = dtime.strftime("%H:%M:%S")
        # try:
        # tau_override = get_tau(date_str, time_str,tau_path, version='2024.dev.1')
        # self.ts.t = 234234232342
        try:
            tau_override = get_tau(None, self.ts.t, None, version='2024.dev.1')
        except ValueError:
            print('Please Update /home/benvaughan/TIME-analysis/tau_measurements.txt from http://modelo.as.arizona.edu:8080/export/export_data')
            tau_override = 0.75         

        # print('TAU OVERRIDE HARD CODED TO version = 2024.dev.1')
        # tau_override = 0.105 
        # print('tau hardcoded to 0.105')
        # print('--------------------------------')

        ### enforce that tau is set to our prediction from the previous block of code and use it to generate a model of the atmosphere 
        self.ts.header['tau'] = tau_override
        self.ts.set_tau(tau_override) 
        self.ts.tau_copy = self.ts.tau
        self.tau_override = tau_override
        self.atm_function = setup_simple_atm_model()

        ### correct for atmospheric attenuation 
        self.ts.correct_tau(self.atm_function)


        if scan_override != 'None':
            self.ts.header['scan_pars']['direction'] = scan_override
        #store map making parameters for later 
        self.save_path = save_path 
        self.debugging_plots = debugging_plots
        self.thresh = thresh; self.n_edge_start = n_edge_start; self.n_edge_finish=n_edge_finish 
        self.n_scan_start = n_scan_start; self.n_scan_finish = n_scan_finish; self.filter_deg = filter_deg
        self.pixel = pixel 
        self.planet_list = ['Jupiter','uranus','mars','jupiter','Mars', 'Uranus', 'Saturn', 'saturn']
        self.quasar_list = ['3C454.3','3C273','3C279','3C84']
        self.beamtype = beamtype
             
    def apply_first_det_cuts(self, verbose = False):
        """ This function doesn't do anything special, it might not even need to be a function but what it does is it removes
        all images that are filled with zeros or filled with nans. This first step saves a lot of computation time by not having to deal
        with detector data that is unpopulated. It also saves the good detectors for later use. 
        
        Parameters
        ----------
        None
        Returns
        -------
        None

        """
        # if self.ts.data.shape[0] == 16:
        #     self.ts.data = self.ts.data.reshape(16 * 60, self.ts.data.shape[2])
        ### first set of data cuts
        nancheck = np.all(np.isnan(self.ts.data),axis=1)
        # print(nancheck, 'nan_check')
        zerocheck = np.all(self.ts.data==0,axis=1)
        idx = np.nonzero((~nancheck) & (~zerocheck))[0].astype('int')
        det_idx = self.ts.header['xf_coords'][idx]
        self.ts.restrict_detectors(det_idx, det_coord_mode='xf')
        ### store a list of good_xf coordinates to be passed onto to other datasets if necessary, e.g. in the case where we are 
        ### using both a calibrating source as well as a science source. 
        self.good_x = [xf[0] for xf in det_idx]; self.good_f = [xf[1] for xf in det_idx]
        self.good_xf = [(x,f) for x,f in zip(self.good_x, self.good_f)]

    # def compute_power_spectrum():
    #     f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)), detrend=False, scaling='density')

    def cross_power_all(self,timestreams, dt=1/100):
        """
        Returns C_ij(f) = X_i(f) X_j*(f)
        """
        timestreams = np.asarray(timestreams)

        # remove mean per timestream (important)
        timestreams = timestreams - np.nanmean(timestreams, axis=1, keepdims=True)

        N_streams, N = timestreams.shape
        freqs = np.fft.rfftfreq(N, d=dt)

        # FFT once
        F = np.fft.rfft(timestreams, axis=1)

        # Allocate cross-power matrix
        C = np.zeros((N_streams, N_streams, len(freqs)), dtype=np.complex64)

        for i in range(N_streams):
            for j in range(i, N_streams):
                C[i, j] = F[i] * np.conj(F[j])
                C[j, i] = np.conj(C[i, j])


        plt.figure(figsize=(7,4))
        plt.plot(freqs, np.abs(Pxy))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('|P_xy(f)|')
        plt.title('Cross Power Spectrum')
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_path + 'cross_power_test2')

    def mean_cross_power(self,timestreams, dt=1/100):
        timestreams = np.asarray(timestreams)
        timestreams = timestreams - np.mean(timestreams, axis=1, keepdims=True)

        N_streams, N = timestreams.shape
        freqs = np.fft.rfftfreq(N, d=dt)

        F = np.fft.rfft(timestreams, axis=1)

        cross_sum = np.zeros(len(freqs), dtype=np.complex128)
        count = 0

        for i in range(N_streams):
            for j in range(i+1, N_streams):
                cross_sum += F[i] * np.conj(F[j])
                count += 1

                
        plt.figure(figsize=(7,4))
        plt.plot(freqs, np.abs(Pxy))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('|P_xy(f)|')
        plt.title('Cross Power Spectrum')
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_path + 'cross_power_test')

        return freqs, cross_sum / count

    def apply_unmasked_filtering(self):
        """ Creates un-weighted error maps with a generic peak filtering algorithm

        Parameters, note that there are no parameters that get passed to this function, they are provided as inputs to the variance_maps_handler class. 
        below is the list of parameters that can be passed through 
        ----------
        thresh : int
        the minimum number of data points in a scan before it is considered "short" and removed from the data set.
        This is to remove spurious data that may contaminate our results 
        n_edge_start : int 
        The number of scans to remove scans from the bottom of the map 
        n_edge_finish : int 
        The number of scans to remove scans from the top of the map
        n_scan_start: int 
        The number of scans to remove from the left side of the map 
        n_scan_finish: int 
        The number of scans to remove from the right side of the map 
        filter_deg : int 
        The polynomial degree used for atmospheric filtering 
        pixel : tuple 
        the x and y pixelsizes in arcseconds default to (0.006, 0.006)
        Returns
        -------
        None
        """
        
        ### do filtering         
        self.ts.remove_obs_flag() # Get rid of points where the telescope wasn't in an "observing" mode
        self.ts.flag_scans() # Identify individual scans across the map
        self.ts.remove_scan_flag() # If data didn't appear to belong to a scan, drop it
        self.ts.remove_short_scans(thresh=self.thresh) # Remove anything that appears to short to be a real scan
        self.ts.remove_end_scans(n_start=self.n_edge_start,n_finish=self.n_edge_finish) # Remove the first few scans and last few scans (ie top and bottom of the map)
        self.ts.remove_scan_edge(n_start=self.n_scan_start, n_finish=self.n_scan_finish) # Remove a few data points from the edges of the scan, where the telescope may still be accelerating
        if not os.path.isdir(self.save_path + 'first_pass_ts/'):
            print('MAKING', self.save_path + 'first_pass_ts')
            os.makedirs(self.save_path + 'first_pass_ts')
        self.ts.filter_scan(n=self.filter_deg, save_dir=self.save_path + 'first_pass_ts/')

        ### before correcting for feed offset, do commmon mode removal
        #if remove_common_mode:
        #    self.ts.remove_common_mode_noise(save_path = self.save_path,pixel=self.pixel)

        if self.ts.header['object'] in self.planet_list:
            use_offsets = False
        elif self.ts.header['object'].upper() in self.quasar_list:
            use_offsets=False
        else:
            use_offsets=True           
        print('made naive map with offsets %s' % use_offsets) 
        self.ts.make_map(pixel=self.pixel, use_offsets=use_offsets)
        map_head = self.ts.Maps.header['map_pars']   
        self.cr = map_head['x_center']; self.cd = map_head['y_center'] ### cra is center ra, cd is center dec

    def aprio_guess_at_feed_offsets(self):

        if not os.path.isdir(self.save_path + 'mask_tests/'):
            os.mkdir(self.save_path+'mask_tests')
        ra_off = np.load('/home/benvaughan/TIME-analysis/bens_fabulous_0_angle_relative_ra_offsets.npy')
        dec_off = np.load('/home/benvaughan/TIME-analysis/bens_fabulous_0_angle_relative_dec_offsets.npy')

        theta = self.ts.header['scan_pars']['map_angle_offset']
        theta_rad = -1 * np.deg2rad(theta)

        ra_rot = -ra_off * np.cos(theta_rad) - np.sin(theta_rad) * dec_off 
        dec_rot = +ra_off * np.sin(theta_rad) - np.cos(theta_rad) * dec_off

        timestamp = self.ts.header['med_time']
        t = Time(timestamp, format='unix')
        loc = EarthLocation.from_geodetic(lat=(31 + 57/60 + 10.8 / 3600)*u.deg, lon=(-111 - 36/60 - 54.0 / 3600)*u.deg, height=1897.3*u.m)

        planet_eph = get_body(self.ts.header['object'], t, loc)
        # planet_eph = planet_eph.transform_to('icrs')   
        earth_eph = get_body('earth',t, loc)              
        altaz_system = AltAz(obstime=t,location=loc)
        planet_ra = float(planet_eph.ra.to_string(decimal=True))
        planet_dec = float(planet_eph.dec.to_string(decimal=True))
        # planet_az_el = planet_eph.transform_to(apcoord.AltAz(obstime=t,location=loc))
        # planet_az = float(planet_az_el.az.to_string(decimal=True))
        # planet_el = float(planet_az_el.alt.to_string(decimal=True))  

        def make_mask(ra_rot, dec_rot, planet_ra, planet_dec, scale=90/3600):
            ndets = len(self.ts.header['xf_coords'])
            xx,yy = np.meshgrid(self.ts.Maps.x_center,self.ts.Maps.y_center)
            ra_bins = np.zeros((ndets,xx.shape[1]))
            dec_bins = np.zeros((ndets,xx.shape[0]))
            ra_dec_mask = np.zeros((ndets,xx.shape[0], xx.shape[1])).astype('bool') 
            idx = 0 
            for xf in self.ts.header['xf_coords']:
                x = xf[0]; f = xf[1]
                ra_off = ra_rot[x]
                dec_off = dec_rot[x]
                p_ra = planet_ra + ra_off
                p_dec = planet_dec + dec_off
                idx = self.ts.Maps.get_xf(x,f)
                rr = np.sqrt((xx - p_ra)**2 + (yy - p_dec)**2)
                circ_mask = rr < (scale)
                ra_dec_mask[idx,circ_mask] = True
                idx += 1 
            self.ra_dec_mask = ra_dec_mask 
            return self.ra_dec_mask
        
        ra_dec_mask = make_mask(ra_rot,dec_rot,planet_ra,planet_dec)
        cr = self.ts.Maps.x_center; cd = self.ts.Maps.y_center


        for idx in range(len(self.ts.header['xf_coords'])):

            ### use mask function to create the mask 
            xf = self.ts.header['xf_coords'][idx]
            x = xf[0]; f = xf[1]
            ra_bins = cr; dec_bins = cd
            # ra, dec = self.ts.offset_pos(self.ts.header['xf_coords'][idx][0])
            ra,dec = self.ts.ra, self.ts.dec
            ra_mask_bin_inds = np.digitize(ra,ra_bins)
            dec_mask_bin_inds = np.digitize(dec,dec_bins)
            mask = ra_dec_mask[idx]
            big_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2))
            big_mask[1:-1,1:-1] = mask
            mask_flag = big_mask[tuple(dec_mask_bin_inds),tuple(ra_mask_bin_inds)]
            if self.debugging_plots:
                fig, axs = plt.subplots(2,1)
                axs[0].scatter(ra, dec, c=self.ts.data[idx])
                self.ts.data[idx][mask_flag==1] = np.nan
                axs[1].scatter(ra, dec, c=self.ts.data[idx])
                axs[0].set_title('Data')
                axs[1].set_title('Mask')
                # axs[0].set_aspect('equal')
                # axs[1].set_aspect('equal')
                fig.savefig(self.save_path + 'mask_tests/mask_check_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')
                plt.close(fig)

        ### Revert to the raw data and re-apply all these corrections 
        ### then we do the new masked version of the planet 
        ### in the case of other observations, this is the first time data gets processed so the reset doesn't actually
        ### have an effect 
        self.ts.reset()
        self.ts.correct_tau(self.atm_function)
        self.ts.restrict_detectors(self.good_xf,det_coord_mode='xf') ### throw out bad fits again
        self.ts.remove_obs_flag() # Get rid of points where the telescope wasn't in an "observing" mode
        self.ts.flag_scans() # Identify individual scans across the map
        self.ts.remove_scan_flag() # If data didn't appear to belong to a scan, drop it
        self.ts.remove_short_scans(thresh=self.thresh) # Remove anything that appears to short to be a real scan
        self.ts.remove_end_scans(n_start=self.n_edge_start,n_finish=self.n_edge_finish) # Remove the first few scans and last few scans (ie top and bottom of the map)
        self.ts.remove_scan_edge(n_start=self.n_scan_start, n_finish=self.n_scan_finish) # Remove a few data points from the edges of the scan, where the telescope may still be accelerating

        self.ts.filter_scan(mode='mask', n=self.filter_deg,nscans=0,mask_ra_bins=ra_bins, mask_dec_bins=dec_bins, mask=ra_dec_mask, make_stds=True, save_dir=self.save_path + 'timestreams/')

        print('MAKING MAPS!!!!')
        self.ts.make_map(pixel=self.pixel)



    def apply_beam_fits(self, ident=None, sigma=False, verbose=False, amp_cut=10, snr_cut=0, rms_cut = 0.8, rad_scale=2, NaN_ratio=0.5, mask_size=None):
        """ Wrapper that fits a gaussian function to each detector, this finds the center of the planet, which can be used to backout
        offset corrections. It first fits to the data then cuts detectors with bad fits, then refits the data. It then does some cuts based off of reduced chi squared and signal to noise ratios.
        Parameters
        ----------
        ident : (str) an identifier for the debugging plots that gets appended to the saved maps directory
        sigma : (bool) a boolean operator that indicates whether or not to use error weighting in the fits or not.
        amp_cut: (float) a minimum threshold that the best fit amplitude has to be above
        snr_cut: (float) a minimum threshold for the SNR
        rms_cut: (float) ratio of rms before and after subtracting planet model, determines if there is something significant fit to the data
        rad_scale: (float) for SNR and out of bounds exclusion mode, a multiplicative factor applied to the fitted fwhm that is used to mask
        out pixels for estimating SNR and to determine the boundaries for the out of bounds exclusion zone. 
        Returns
        -------
        fits : (array) (n_det, n_param) an array of the best fit parameters
        covs : (array) (n_det, n_param,n_param) covariance matrices for the best fit parameters 
        """

        if not sigma:
            self.ts.Maps.write(self.save_path + 'Cal_maps_no_offset_p%s' % self.mc)

        if self.debugging_plots:
        #     xf_coords = self.ts.header['xf_coords']
            if not os.path.isdir(self.save_path + 'pre_filtering/'):
                os.makedirs(self.save_path + 'pre_filtering')
        #     for xf in xf_coords:
        #         good_flag = False
        #         x = xf[0]; f = xf[1]
        #         ### create models and compute chi squared / SNR 
        #         idx = self.ts.Maps.get_xf(x,f)
        #         detector_map = self.ts.Maps.maps[idx]
        #         # detector_map = map_copy[idx]
        #         ### set up some plotting parameters for debugging plots.
        #         fig,axes = plt.subplots(1,1,figsize=(9,9),sharex=True,sharey=True)
        #         fig.patch.set_facecolor('w')
        #         w = .8
        #         h = .8
        #         fig.subplots_adjust(hspace=0,wspace=0,left=.1,right=.1+w,top=.1+h,bottom=.1)
        #         ax_cb1 = fig.add_axes([.95,.1+h/3+.01*h/3,.03,1.99*h/3])
        #         # ax_cb2 = fig.add_axes([.95,.1,.03,.99*h/3])
        #         # axes[1].set(ylabel='Y Offset from Center [arcsec]',aspect='equal')
        #         # axes[2].set(xlabel='X Offset from Center [arcsec]',aspect='equal')
        #         plt.setp(axes.get_xticklabels()+axes.get_xticklabels(),visible=False)

        #         axes.set(aspect='equal')
        #         (vmin,vmax,cmap) = (np.nanmin(detector_map),np.nanmax(detector_map),'hot')
        #         cm1 = axes.imshow(detector_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        #         # axes[1].imshow(model_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        #         # (vmin,vmax,cmap) = (-np.nanmax(np.abs(detector_map-model_map)),np.nanmax(np.abs(detector_map-model_map)),'coolwarm')
        #         # cm2 = axes[2].imshow(detector_map-model_map, origin='lower', cmap=cmap)
        #         # axes[0].text(.025,.8,'Data',fontweight='bold',color='w',transform=axes[0].transAxes)
        #         # axes[1].text(.025,.8,'Model',fontweight='bold',color='w',transform=axes[1].transAxes)
        #         # axes[2].text(.025,.8,'Difference',fontweight='bold',color='k',transform=axes[2].transAxes)
        #         fig.colorbar(cm1, cax=ax_cb1, label='Counts [Arb]')
        #         # fig.colorbar(cm2, cax=ax_cb2, label='Residuals [Arb]')
        #         if sigma:
        #             pass
        #             plt.close(fig)
        #         else:
        #             fig.savefig(self.save_path + 'pre_filtering/qual_assurance_' + ident + '_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')  
        #             plt.close(fig)      

        ### now fit the beam parameters
        self.fit_function,self.fits,covs = self.ts.Maps.repurposed_beam_fit(nofit_handling='allow', beamtype=self.beamtype)



        ### failed fits will have NaNs as the best fit parameters, throw these out.
        bad_fit_inds = np.nonzero(~np.isnan(self.fits[:,0]))[0]  
        self.bad_fit_dets = self.ts.header['xf_coords'][bad_fit_inds]
        ### remove the failed fit detectors from the sample of possible detectors

        # print(self.fits)
        # print(covs)
        self.ts.restrict_detectors(self.bad_fit_dets,det_coord_mode='xf');self.ts.Maps.restrict_detectors(self.bad_fit_dets, det_coord_mode='xf')

        ### update the list of good xf coordinates to include the results from the fitting as well ! 
        bad_x2 = [xf[0] for xf in self.bad_fit_dets]; bad_f2 = [xf[1] for xf in self.bad_fit_dets]
        self.good_x = np.intersect1d(self.good_x, bad_x2); self.good_f = np.intersect1d(self.good_x, bad_f2)
        self.good_xf = [(x,f) for x,f in zip(self.good_x, self.good_f)]


        if not os.path.isdir(self.save_path + 'good_dets/'):
            os.makedirs(self.save_path + 'good_dets')
        if not os.path.isdir(self.save_path + 'bad_dets/'):
            os.makedirs(self.save_path + 'bad_dets')


        map_copy = self.ts.Maps.maps.copy()
        xx,yy = np.meshgrid(self.ts.Maps.dx_center,self.ts.Maps.dy_center)
        map_shape = self.ts.Maps.maps.shape
        map_shape2 = [map_shape[1], map_shape[2]]
        model_func = self.ts.Maps.beam_fit_results['function']  #.reshape(detector_map.shape)
        all_bfits = self.ts.Maps.beam_fit_results['fits']
        xf_coords = self.ts.header['xf_coords']

        ### make some data holders for the reduced chi squared and the SNR ratio of the plots
        beam_amps = np.zeros((16,60))
        beam_amps[:,:] = np.nan
        map_of_kept_rem_dets = np.zeros((16,60))
        map_of_kept_rem_dets[:,:] = 16
        ### we will be updating the list of good xf coordinates again 
        good_list = []
        self.first_pass_fits = []
        self.first_pass_list = []

        #### produce quality control plots!
        ### calculate a meshgrid of ra/dec values 
        ### note that dx center and dy center are the ra/dec edges - their center 

        resid_maps = np.zeros((map_shape))

        resid_amps = np.zeros((16,60))
        peak_amps = np.zeros((16,60))
        map_counter = 0 
        for xf in xf_coords:
            good_flag = False
            x = xf[0]; f = xf[1]
            ### create models and compute chi squared / SNR 
            idx = self.ts.Maps.get_xf(x,f)
            detector_map = map_copy[idx]
            bfits = all_bfits[idx]
            model_map = model_func((xx,yy,np.ones(xx.size)), *bfits).reshape(map_shape2)

            #### First Flag, does beam size make sense?
            frequency = f_ind_to_val(f)
            theoretical_size = 1.2 * 2.998e8/frequency/1e9 / 12 * 180/np.pi*3600

            if self.beamtype == 'Gauss': 
                beam_size_flag = bfits[3]*3600 > theoretical_size
            elif self.beamtype == 'rotate_gauss':
                beam_size_flag = np.nanmin([bfits[3],bfits[4]]) * 3600 > theoretical_size
            # beam_size_flag = True
            # print('Beam Size Flag is hard Coded to be True ------ 371')

            nan_num = np.sum(np.isnan(detector_map))
            im_size = detector_map.size
            
            #### Second Flag, Is the planet within the preset boundaries, i.e. is it hanging off the edge of the map
            out_of_bounds_flag = False
            if self.beamtype == 'Gauss': 
                if nan_num / im_size < NaN_ratio:
                    if yy.max() > (bfits[2] + rad_scale*bfits[3]) or yy.min() < (bfits[2] - rad_scale*bfits[3]):
                        out_of_bounds_flag = True
                    if xx.max() > (bfits[1] + rad_scale*bfits[3]) or xx.min() < (bfits[1] - rad_scale*bfits[3]):
                        out_of_bounds_flag = True
            out_of_bounds_flag = True
            # print('Setting out of bounds flag to be True')


            #### Third Flag, SNR of the planet 
            rr = ( (xx-bfits[1]) / (rad_scale*bfits[3]))**2 + ((yy - bfits[2])/ (rad_scale*bfits[3]))**2 ### radius of plane
            circ_mask = rr > 1
            detector_map = map_copy[idx]
            det_copy = detector_map.copy()
            det_copy[circ_mask] = np.nan
            amp_for_snr = np.nanmax(det_copy)

            circ_mask = rr < 1
            detector_map = map_copy[idx]
            det_copy = detector_map.copy()
            det_copy[circ_mask] = np.nan
            amp_over_std = amp_for_snr / np.nanstd(det_copy)
            amp_flag = (amp_for_snr > amp_cut) and (amp_over_std > snr_cut)

            # amp_flag = True
            # print('AMP FLAG SET TO TRUE')

            # print(amp_flag)
            #### Fourth Flag, RMS Check, is there even a planet in this data? 
            dmap_rms = np.sqrt( np.nansum(detector_map**2) / detector_map.size)
            resid_rms = np.sqrt(np.nansum((detector_map - model_map)**2) / detector_map.size)
            rms_bool = resid_rms / dmap_rms < rms_cut ### deciding on whether or not to cut rms 

            #### Now Combine Flags Together 
            good_flag = beam_size_flag and amp_flag and rms_bool and out_of_bounds_flag
            
            # amp_flag = True
            # print(beam_size_flag, 'beam size flag', amp_flag, 'amp_flag', rms_bool, 'rms_bool', out_of_bounds_flag, 'oob')

            #### BIT MAPPING
            map_of_kept_rem_dets[x,f] = 0
            if beam_size_flag:
                map_of_kept_rem_dets[x,f] += 1
            if amp_flag:
                map_of_kept_rem_dets[x,f] += 2
            if rms_bool:
                map_of_kept_rem_dets[x,f] += 4
            if out_of_bounds_flag:
                map_of_kept_rem_dets[x,f] += 8
            if sigma:
                if good_flag:
                    good_list.append((x,f))                
                    beam_amps[x,f] = bfits[0]
            else: 
                ### save a copy of a list of detectors that passed the first set of quality checks
                good_list.append((x,f)) 
                if good_flag:
                    self.first_pass_fits.append([bfits]) ### but do pass on the fits for the offset calcs only for good flags
                    self.first_pass_list.append((x,f))

            # if self.debugging_plots:
            #     ### set up some plotting parameters for debugging plots.
            #     fig,axes = plt.subplots(3,1,figsize=(9,9),sharex=True,sharey=True)
            #     fig.patch.set_facecolor('w')
            #     w = .8
            #     h = .8
            #     fig.subplots_adjust(hspace=0,wspace=0,left=.1,right=.1+w,top=.1+h,bottom=.1)
            #     ax_cb1 = fig.add_axes([.95,.1+h/3+.01*h/3,.03,1.99*h/3])
            #     ax_cb2 = fig.add_axes([.95,.1,.03,.99*h/3])
            #     # axes[1].set(ylabel='Y Offset from Center [arcsec]',aspect='equal')
            #     # axes[2].set(xlabel='X Offset from Center [arcsec]',aspect='equal')
            #     plt.setp(axes[0].get_xticklabels()+axes[1].get_xticklabels(),visible=False)

            #     axes[0].set(aspect='equal')
            #     axes[1].set(aspect='equal')
            #     axes[2].set(aspect='equal')
            #     vmin = 0 
            #     (vmin,vmax,cmap) = (np.nanmin(detector_map) ,np.nanmax(detector_map),'hot')
            #     cm1 = axes[0].pcolor(self.ts.Maps.dx_edge * 3600, self.ts.Maps.dy_edge* 3600, detector_map,cmap=cmap, vmin=0, vmax=vmax)
            #     axes[1].pcolor(self.ts.Maps.dx_edge * 3600, self.ts.Maps.dy_edge * 3600, model_map, cmap=cmap, vmin=0, vmax=vmax)
            #     (vmin,vmax,cmap) = (-np.nanmax(np.abs(detector_map-model_map)),np.nanmax(np.abs(detector_map-model_map)),'coolwarm')
            #     cm2 = axes[2].pcolor(self.ts.Maps.dx_edge * 3600, self.ts.Maps.dy_edge * 3600, detector_map-model_map, cmap=cmap)
            #     # axes[0].set_xlim(-500,500)
            #     # axes[0].set_ylim(-250,250)
            #     axes[2].set_xlabel('Distance From Center [as]')
            #     axes[0].set_ylabel('Distance From Center [as]')
            #     axes[1].set_ylabel('Distance From Center [as]')
            #     axes[2].set_ylabel('Distance From Center [as]')
            #     axes[0].text(.025,.8,'Data',fontweight='bold',color='w',transform=axes[0].transAxes)
            #     axes[1].text(.025,.8,'Model',fontweight='bold',color='w',transform=axes[1].transAxes)
            #     axes[2].text(.025,.8,'Difference',fontweight='bold',color='k',transform=axes[2].transAxes)
            #     fig.colorbar(cm1, cax=ax_cb1, label='Counts [Arb]')
            #     fig.colorbar(cm2, cax=ax_cb2, label='Residuals [Arb]')
            #     if sigma:
            #         if good_flag:
            #             fig.savefig(self.save_path + 'good_dets/qual_assurance_' + ident + '_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')
            #         else:
            #             fig.savefig(self.save_path + 'bad_dets/qual_assurance_' + ident + '_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')
            #     else:
            #         if good_flag:
            #             fig.suptitle('PASSED')
            #         else:
            #             fig.suptitle('DECLINED')
            #         fig.savefig(self.save_path + 'pre_filtering/qual_assurance_' + ident + '_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')

            #     plt.close(fig)

            #     resid_maps[map_counter,:,:] = detector_map - model_map 

                map_counter += 1 

                # if not os.path.isdir(self.save_path + 'side_profiles/'):
                #     os.mkdir(self.save_path+'side_profiles')
                # x_profile = np.sum(detector_map, axis=0)
                # y_profile = np.sum(detector_map, axis=1)
                # x_profile_model = np.sum(model_map, axis=0)
                # y_profile_model = np.sum(model_map, axis=1)
                # if self.debugging_plots:
                #     fig, axs = plt.subplots(1,2, sharey=True)
                #     axs[0].plot(self.ts.Maps.dx_edge[:-1]*3600, x_profile, label='data')
                #     axs[0].plot(self.ts.Maps.dx_edge[:-1]*3600, x_profile_model, label='best fit')
                #     axs[1].plot(self.ts.Maps.dy_edge[:-1]*3600, y_profile, label='data')
                #     axs[1].plot(self.ts.Maps.dy_edge[:-1]*3600, y_profile_model, label='best fit')
                #     axs[0].legend(framealpha=0); axs[1].legend(framealpha=0)
                #     axs[0].set_xlabel('$\\Delta$RA [arcsecond]'); axs[0].set_ylabel('Counts [ADU]')
                #     axs[1].set_xlabel('$\\Delta$DEC [arcsecond]'); axs[1].set_ylabel('Counts [ADU]')
                #     fig.savefig(self.save_path + 'side_profiles/' + ident + '_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')
                #     plt.close(fig)
        if sigma: 
            path2 = 'amp_over_std_second_pass_p%s' % self.mc
            path3 = 'kept_rem_second_pass_%s' % self.mc
        else:
            path2 = 'amp_over_std_p%s' % self.mc
            path3 = 'kept_rem_first_pass_p%s' % self.mc


        ### Make bit mask map 
        labels = ['BS', 'SNR', 'SNR+BS', 'RMS', 'RMS+BS', 'RMS+SNR', 'RMS+BS+SNR', 'OOB', 'OOB+BS', 'OOB+SNR', 'OOB+SNR+BS', 'OOB+RMS', 'OOB+RMS+BS', 'OOB+RMS+SNR', 'ALL', 'NO DATA']
        levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16]
        bounds = [li +0.5 for li in levels]
        color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#a55194", "#393b79", "#637939", "#8c6d31", "#5C0000", "black"] #"#ff1493"]
        cmap = ListedColormap(color_list, name="custom16")
        norm = BoundaryNorm(bounds, cmap.N)
        plot_maps(self.save_path + path3, data=map_of_kept_rem_dets, markers=None, title='',
            clabel='', cmin=None, cmax=None, scale=1.0, missing_color='grey',
            cmap=cmap, norm=norm, mux_space=False, marker_labels=None,
            cticks=levels, cticklabels=labels, fontsize=14, titlesize=16, legendsize=11,
            logscale=False)  
        np.save(self.save_path + 'npy_files/' + path3, map_of_kept_rem_dets, allow_pickle=True)

        ### plot beam amplitudes
        plot_maps(self.save_path + path2, data=beam_amps, markers=None, title='',
            clabel='Beam Amp [arb]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
            cmap='inferno', norm=None, mux_space=False, marker_labels=None,
            cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
            logscale=False)  

        self.good_xf = good_list 

        ### save beam parameters 
        if sigma: 
            np.save(self.save_path + 'npy_files/best_fits_second_pass_p%s' % self.mc, self.fits, allow_pickle=True)
            np.save(self.save_path + 'npy_files/best_fits_second_pass_e_p%s' % self.mc, covs, allow_pickle=True)
        else:
            np.save(self.save_path + 'npy_files/best_fits_first_pass_p%s' % self.mc, self.fits, allow_pickle=True)
            np.save(self.save_path + 'npy_files/best_fits_first_pass_e_p%s' % self.mc, covs, allow_pickle=True)


        if sigma:
            ### if second pass get rid of failed detectors and re-fit
            ### this is somethign that can be optimized the refitting is done to make the length of 
            ### fit parameters and uncertainties match the length of the detectors... which is a very lazy 
            ### and inefficient way of doing it, something should be added to restrict_detectors that just cuts those indices...
            self.ts.restrict_detectors(good_list, det_coord_mode='xf'); self.ts.Maps.restrict_detectors(good_list, det_coord_mode='xf')
            self.fit_function,self.fits,covs = self.ts.Maps.repurposed_beam_fit(nofit_handling='allow', beamtype=self.beamtype)
        else:
            ### save two arrays of x and f coordinates that passed first checks that can be used later for debugging 
            first_pass_x = np.array([xf[0] for xf in self.first_pass_list])
            first_pass_f = np.array([xf[1] for xf in self.first_pass_list])
            np.save(self.save_path + 'npy_files/first_pass_x_coords', first_pass_x, allow_pickle=True)
            np.save(self.save_path + 'npy_files/first_pass_f_coords', first_pass_f, allow_pickle=True)

        return self.fits,covs, resid_amps, peak_amps, resid_maps

    def define_dynamic_mask(self, scale=90/3600):
        """ Computes a dynamic mask based off of the best fit parameters to the beams, this is then used to make better filtered maps 
        Parameters
        ----------
        scale: in units of degrees, the radius of a mask for masking detector timestreams 
        -------
        fits : (array) (n_det, n_param) an array of the best fit parameters
        covs : (array) (n_det, n_param,n_param) covariance matrices for the best fit parameters 
        """
        ndets = len(self.ts.header['xf_coords'])
        xx,yy = np.meshgrid(self.ts.Maps.x_center,self.ts.Maps.y_center)
        ra_bins = np.zeros((ndets,xx.shape[1]))
        dec_bins = np.zeros((ndets,xx.shape[0]))
        ra_dec_mask = np.zeros((ndets,xx.shape[0], xx.shape[1])).astype('bool') 
        idx = 0 
        for xf in self.ts.header['xf_coords']:
            x = xf[0]; f = xf[1]
            idx = self.ts.Maps.get_xf(x,f)
            rr = np.sqrt((xx - self.x_cent)**2 + (yy - self.y_cent)**2)
            circ_mask = rr < (scale)
            ra_dec_mask[idx,circ_mask] = True
            idx += 1 
        self.ra_dec_mask = ra_dec_mask 
        return self.ra_dec_mask

    def find_offsets(self, boresight=7):
        """ Uses Maps.beam_offsets to correct the feed offsets to align with the boresight
        Parameters
        ----------
        boresight : (int) 
        the feed index (starting at 0) that we align the rest of the feeds with, default is 7 (0 indexed).
        fits : ()
        Returns
        -------
        joint_offsets : (UDT)
        a user defined type which contains the offset parameters
        """

        def ret_x(x, coord_list):
            matches = [v[0]==x for v in coord_list]
            indices = np.nonzero(matches)[0]
            return indices

        self.first_pass_fits = np.squeeze(np.array(self.first_pass_fits))
        ### now we correct for the feed offsets 
        dx = np.zeros(16)
        dy = np.zeros(16)
        med_inds1 = ret_x(7, self.first_pass_list)
        med_inds2 = ret_x(8, self.first_pass_list)
        x0_fits1 = self.first_pass_fits[med_inds1]
        x0_fits2 = self.first_pass_fits[med_inds2]

        # print(med_inds1, med_inds2)

        x0_med_x1 = np.nanmedian(x0_fits1[:,1])
        x0_med_x2 = np.nanmedian(x0_fits2[:,1])
        x0_med_y1 = np.nanmedian(x0_fits1[:,2])
        x0_med_y2 = np.nanmedian(x0_fits2[:,2])

        # print(x0_med_x1, x0_med_x2, x0_med_y1, x0_med_y1)


        x0_med_x = (x0_med_x1 + x0_med_x2) / 2 
        x0_med_y = (x0_med_y1 + x0_med_y2) / 2 
        
        # print(x0_med_x, x0_med_y)

        # x0_med_x = np.nanmedian(x0_fits[:,1])
        # x0_med_y = np.nanmedian(x0_fits[:,2])
        for i in range(16): #16 because there are 16 feeds
            i_inds = ret_x(i, self.first_pass_list)
            past_fits = self.first_pass_fits[i_inds]
            dx_i, dy_i = self.ts.Maps.beam_offsets(boresight, False, x0_med_x=x0_med_x, x0_med_y=x0_med_y,x_fits=past_fits, past_fits=True).get(i)
            dx[i] += dx_i
            dy[i] += dy_i

        print(self.ts.header)

        theta = self.ts.header['scan_pars']['map_angle_offset']

        theta_arr = np.array(theta)
        np.save(self.save_path + 'npy_files/offset_dx', dx, allow_pickle=True)
        np.save(self.save_path + 'npy_files/offset_dy', dy, allow_pickle=True)
        np.save(self.save_path + 'npy_files/theta', theta_arr, allow_pickle=True)


        theta_rad = np.deg2rad(theta)
        fig, axs = plt.subplots(1, figsize=(6,6))
        # fig.suptitle('Map Angle is %s') % theta
        axs.set_title('Map Angle is %s' % theta)
        axs.scatter(dx,dy, label='Base Offsets (Rotated)')
        dxr = dx * np.cos(theta_rad) + np.sin(theta_rad) * dy 
        dyr = -dx * np.sin(theta_rad) + np.cos(theta_rad) * dy
        axs.scatter(dxr, dyr, label='Offsets (De-Rotated +1)')
        axs.legend()
        fig.savefig(self.save_path + 'K-mirror Rotated', bbox_inches='tight')
        plt.close(fig)





        diff_x = np.nanmedian(np.diff(dx)); diff_y = np.nanmedian(np.diff(dy))
        beam_mult = np.arange(16) - boresight 
        if np.sum(np.isnan(dx)) > 0:
            bad_ind = np.isnan(dx)
            dx[bad_ind] = (beam_mult * diff_x)[bad_ind]
            dy[bad_ind] = (beam_mult * diff_y)[bad_ind]


        # exit()



        joint_offsets = Offsets(dx,dy,frame=self.ts.header['epoch'])
        self.ts.set_feed_offsets(joint_offsets,overwrite=True)
        self.ts.make_map(pixel=self.pixel, use_offsets=True)
        return joint_offsets

    def quasar_set_bf_centers(self):
        ### This is depreciated also didnt do 2D observations of Quasars in engineering run...
        self.bf_centers = np.zeros((len(self.ts.header['xf_coords']),2))
        for idx in range(len(self.ts.header['xf_coords'])):
            xf = self.ts.header['xf_coords'][idx]
            x = xf[0]; f = xf[1]
            feed_offset_x, feed_offset_y = self.ts.Maps.get_feed_offsets(x,f,'xf')
            ra, dec = self.ts.offset_pos(x) ### extract ra, dec timestreams that have been corrected for feed offsets.
            peak_ra = self.fits[idx][1] + self.cr - feed_offset_x / self.ts.Maps.dx_scale_factor; peak_dec = self.fits[idx][2] + self.cd - feed_offset_y
            self.bf_centers[idx,0] = peak_ra; self.bf_centers[idx,1] = peak_dec

    def find_planet_each_det(self, max_ra_diff=1e8, max_dec_diff = 1e8):
        """Finds the location of a planet in the scan for each individual detector which is used to compute delta RA/DEC and cuts detectors with a positional
        difference above max_ra_diff or max_dec_diff.

        Returns
        -------
        planet_centers : array 
        ' a (ndetector, 2) of the RA/DEC pointing of the planet
        max_ra_diff : float
        the max difference between the measured pointing from the beam fit and the epheremis RA, default is 1e8 (e.g. no restriction)
        the max difference between the measured pointing from the beam fit and the epheremis DEC, default is 1e8 (e.g. no restriction)
        """
        # make holding arrays to hold data that has been calculated


        print('find_planet_each_det')

        ### FOR DYNAMIC MASK LATER 
        ### get the positions of each detector image after correcting for offset 
        ### Note beam fits for these centers are first pass beam fits and thus the maps were not feedhorn offset corrected
        self.x_cent = np.zeros((16))
        self.y_cent = np.zeros((16))
        self.max_fwhm = np.zeros((16)) 
        x_list = np.array([li[0] for li in self.first_pass_list])
        for xi in range(16):
            inds = np.where(x_list== xi)[0] ### hopefully we see planets in at least one feedhorn...
            if len(inds) > 0:
                rel_fits = self.first_pass_fits[inds]
                feed_offset_x, feed_offset_y = self.ts.Maps.get_feed_offsets(xi,'xf')
                self.x_cent[xi] = np.nanmedian(rel_fits[:,1]) + self.cr - feed_offset_x / self.ts.Maps.dx_scale_factor 
                self.y_cent[xi]= np.nanmedian(rel_fits[:,2]) + self.cd - feed_offset_y
            else: 
                self.x_cent[xi] = np.nan 
                self.y_cent[xi] = np.nan

        self.x_cent = np.nanmedian(self.x_cent)
        self.y_cent = np.nanmedian(self.y_cent)

        self.planet_centers = np.zeros((len(self.first_pass_list),2))
        self.planet_az_el = np.zeros((len(self.first_pass_list),2))
        planet_ra_grid = np.zeros((16,60))
        planet_dec_grid = np.zeros((16,60))
        planet_az_grid = np.zeros((16,60)) 
        planet_el_grid = np.zeros((16,60))
        planet_ra_grid[:,:] = np.nan 
        planet_dec_grid[:,:] = np.nan 
        planet_az_grid[:,:] = np.nan 
        planet_el_grid[:,:] = np.nan

        good_xf_list = [] 
        
        if not os.path.isdir(self.save_path + 'pointing_tests/'):
            os.mkdir(self.save_path+'pointing_tests')

        self.bf_centers = np.zeros((len(self.first_pass_list),2))
        for idx in range(len(self.first_pass_list)):
            xf = self.first_pass_list[idx]
            x = xf[0]; f = xf[1]
            feed_offset_x, feed_offset_y = self.ts.Maps.get_feed_offsets(x,f,'xf')
            ra, dec = self.ts.offset_pos(x) ### extract ra, dec timestreams that have been corrected for feed offsets.
            peak_ra = self.first_pass_fits[idx][1] + self.cr - feed_offset_x / self.ts.Maps.dx_scale_factor; peak_dec = self.first_pass_fits[idx][2] + self.cd - feed_offset_y
            self.bf_centers[idx,0] = peak_ra; self.bf_centers[idx,1] = peak_dec

            #### Now estimate the timestamp at which the center of the planet was observed, try matching with decreasing precision until 
            #### something works... there might be a better way to do this 
            try:
                t_ind = np.where(np.logical_and(np.round(peak_ra,4) == np.round(ra,4), np.round(peak_dec,4) == np.round(dec,4)))[0] ### something a little more rigorous than this is probably in order, does not account for cosmic rays / other issues
                timestamp = str(self.ts.t[t_ind][0])
            except IndexError:
                try:         
                    t_ind = np.where(np.logical_and(np.round(peak_ra,3) == np.round(ra,3), np.round(peak_dec,3) == np.round(dec,3)))[0] ### something a little more rigorous than this is probably in order, does not account for cosmic rays / other issues
                    timestamp = str(self.ts.t[t_ind][0])
                except IndexError:
                    try:
                        t_ind = np.where(np.logical_and(np.round(peak_ra,2) == np.round(ra,2), np.round(peak_dec,2) == np.round(dec,2)))[0] ### something a little more rigorous than this is probably in order, does not account for cosmic rays / other issues
                        timestamp = str(self.ts.t[t_ind][0])
                    except IndexError:
                        try:
                            t_ind = np.where(np.logical_and(np.round(peak_ra,1) == np.round(ra,1), np.round(peak_dec,1) == np.round(dec,1)))[0] ### something a little more rigorous than this is probably in order, does not account for cosmic rays / other issues
                            timestamp = str(self.ts.t[t_ind][0])
                        except IndexError:
                            print('warning could not estimate peak timestamp from fits, using max value as planet location, x=%s f = %s' % (x,f))
                            t_ind = np.where(self.ts.data[idx] == np.nanmax(self.ts.data[idx]))[0][0]
                            timestamp = t_ind

            t = Time(timestamp, format='unix')
            loc = EarthLocation.from_geodetic(lat=(31 + 57/60 + 10.8 / 3600)*u.deg, lon=(-111 - 36/60 - 54.0 / 3600)*u.deg, height=1897.3*u.m)
            if self.ts.header['object'] in self.planet_list:
                planet_eph = get_body(self.ts.header['object'], t, loc)
                # planet_eph = planet_eph.transform_to('icrs')   
                earth_eph = get_body('earth',t, loc)              
                altaz_system = AltAz(obstime=t,location=loc)
                dist = earth_eph.separation_3d(planet_eph).to(u.m).value
                planet_ra = float(planet_eph.ra.to_string(decimal=True))
                planet_dec = float(planet_eph.dec.to_string(decimal=True))
                planet_az_el = planet_eph.transform_to(apcoord.AltAz(obstime=t,location=loc))
                planet_az = float(planet_az_el.az.to_string(decimal=True))
                planet_el = float(planet_az_el.alt.to_string(decimal=True))  
            else:
                ### hard coded for some qusars I was testing previously, this should be improved later
                if self.ts.header['object'].upper() == '3C273':
                    planet_ra = 187.277415; planet_dec = 2.0525
                elif self.ts.header['object'].upper() == '3C279':
                    planet_ra = 194.046; planet_dec = -5.7894
                elif self.ts.header['object'].upper() == '3C454.3':
                    planet_ra =  343.490; planet_dec =  16.1482
                elif self.ts.header['object'].upper() == '3C84':
                    planet_ra = 49.9515; planet_dec = 41.5117                
                plan_coord = SkyCoord(planet_ra * u.deg, planet_dec * u.deg)
                planet_az_el = plan_coord.transform_to(apcoord.AltAz(obstime=t,location=loc))
                planet_az = float(planet_az_el.az.to_string(decimal=True))
                planet_el = float(planet_az_el.alt.to_string(decimal=True))  
            self.planet_centers[idx,:] = np.array([planet_ra, planet_dec])
            peak_coords = SkyCoord(peak_ra * u.deg, peak_dec * u.deg)
            az_el_peak = peak_coords.transform_to(apcoord.AltAz(obstime=t,location=loc))
            peak_az = float(az_el_peak.az.to_string(decimal=True))
            peak_el = float(az_el_peak.alt.to_string(decimal=True))   
            if self.debugging_plots:
                fig, axs = plt.subplots(1)
                cm = plt.cm.get_cmap('rainbow')
                axs.axis('equal')
                vmin = np.nanmin(self.ts.data[idx]);vmax=np.nanmax(self.ts.data[idx])
                sc = axs.scatter(ra,dec,c=self.ts.data[idx], vmin=vmin, vmax=vmax, cmap=cm)
                divider = make_axes_locatable(axs)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(sc, label='Jy',cax=cax)
                axs.scatter(planet_ra, planet_dec, marker='v', s=25, color='black', label='epheremis')
                axs.scatter(self.bf_centers[idx,0], self.bf_centers[idx,1], marker='P', s=25, color='black',label='best fit')
                fig.savefig(self.save_path + 'pointing_tests/point_ts_x%s_f%s_p%s' % (x,f, self.mc), bbox_inches='tight')
                plt.close(fig)
            planet_ra_grid[x,f] = peak_ra - planet_ra; planet_dec_grid[x,f] = peak_dec - planet_dec 
            planet_az_grid[x,f] = peak_az - planet_az; planet_el_grid[x,f] = peak_el - planet_el 
            ### filter obviously wrong offsets
            if np.abs(planet_ra_grid[x,f])*3600 < max_ra_diff and np.abs(planet_dec_grid[x,f])*3600 < max_dec_diff: 
                good_xf_list.append((x,f))
            else:
                planet_ra_grid[x,f] = np.nan; planet_dec_grid[x,f] = np.nan; planet_el_grid[x,f] = np.nan; planet_az_grid[x,f] = np.nan
        if self.debugging_plots:
            plot_maps(self.save_path +'pointing_tests/ra', data=planet_ra_grid * 3600, markers=None, title='',
                clabel='$\\Delta$RA [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path +'pointing_tests/el', data=planet_el_grid * 3600, markers=None, title='',
                clabel='$\\Delta$el [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path + 'pointing_tests/az', data=planet_az_grid * 3600, markers=None, title='',
                clabel='$\\Delta$Az [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path + 'pointing_tests/dec', data=planet_dec_grid * 3600, markers=None, title='',
                clabel='$\\Delta$DEC [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path +'pointing_tests/ra_med_sub', data=(planet_ra_grid-np.nanmedian(planet_ra_grid)) * 3600, markers=None, title='',
                clabel='$\\Delta$RA [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path +'pointing_tests/el_med_sub', data=(planet_el_grid-np.nanmedian(planet_el_grid)) * 3600, markers=None, title='',
                clabel='$\\Delta$el [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path + 'pointing_tests/az_med_sub', data=(planet_az_grid-np.nanmedian(planet_az_grid)) * 3600, markers=None, title='',
                clabel='$\\Delta$Az [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path + 'pointing_tests/dec_med_sub', data=(planet_dec_grid-np.nanmedian(planet_dec_grid)) * 3600, markers=None, title='',
                clabel='$\\Delta$DEC [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)

        np.save(self.save_path + 'npy_files/planet_eph_el', planet_el, allow_pickle=True)
        np.save(self.save_path + 'npy_files/planet_eph_az', planet_az, allow_pickle=True)
        np.save(self.save_path + 'npy_files/el_offsets',planet_el_grid, allow_pickle=True)
        np.save(self.save_path + 'npy_files/az_offsets',planet_az_grid, allow_pickle=True)
        np.save(self.save_path + 'npy_files/ra_offsets',planet_ra_grid, allow_pickle=True)
        np.save(self.save_path + 'npy_files/dec_offsets',planet_dec_grid, allow_pickle=True)
        return self.planet_centers




    def setup_masked_filter(self, cent_ra, cent_dec, mask_func, mask_params,dims = [0.5,0.5],common_mode_params={'remove_cm':True,'strength':1, 'mux_block':4,'freq_block':[0,8,16,24,36,48,60]}):
        """set up an array of standard deviation values in the same shape as the raw data (ndetector, ndatapoint) to be
        used when producing variance weighted maps 

        Parameters
        ----------
        cent_ra (float) : decimal degrees, center of the new image in RA 
        cent_dec (float) : decimal degrees, center of the new image in DEC 
        mask_func (func) : this can be a new type of mask that someone defines or it can be the many circle mask 
        mask_params (list) : list of params for the mask_func, in the case of the many circle mask the mask params should be 
        a list of lists that contain ra/dec coordinates and radii for each circular mask 
        dims (list of floats) : the dimensions of the map in degrees
        Returns
        -------
        None
        """
        ra_dec_mask = mask_func(mask_params)
            
    
        ### set up folders for debugging plots 
        if self.debugging_plots:
            if not os.path.isdir(self.save_path + 'mask_tests/'):
                os.mkdir(self.save_path+'mask_tests')
            if not os.path.isdir(self.save_path + 'masking_tests/'):
                os.mkdir(self.save_path+'masking_tests')
            ### note that timestreams do not get saved as that is too much, but functionality can be turned on for the enthusiastic practitioner.
            if not os.path.isdir(self.save_path + 'timestreams/'):
                os.mkdir(self.save_path+'timestreams/')

        cr = self.ts.Maps.x_center; cd = self.ts.Maps.y_center
        for idx in range(len(self.ts.header['xf_coords'])):

            ### use mask function to create the mask 
            xf = self.ts.header['xf_coords'][idx]
            x = xf[0]; f = xf[1]
            ra_bins = cr; dec_bins = cd
            ra, dec = self.ts.offset_pos(self.ts.header['xf_coords'][idx][0])
            ra_mask_bin_inds = np.digitize(ra,ra_bins)
            dec_mask_bin_inds = np.digitize(dec,dec_bins)
            mask = ra_dec_mask[idx]
            big_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2))
            big_mask[1:-1,1:-1] = mask
            mask_flag = big_mask[tuple(dec_mask_bin_inds),tuple(ra_mask_bin_inds)]
            # if self.debugging_plots:
            #     fig, axs = plt.subplots(2,1)
            #     axs[0].scatter(ra, dec, c=self.ts.data[idx])
            #     self.ts.data[idx][mask_flag==1] = np.nan
            #     axs[1].scatter(ra, dec, c=self.ts.data[idx])
            #     axs[0].set_title('Data')
            #     axs[1].set_title('Mask')
            #     # axs[0].set_aspect('equal')
            #     # axs[1].set_aspect('equal')
            #     fig.savefig(self.save_path + 'mask_tests/mask_check_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')
            #     plt.close(fig)

        ### Revert to the raw data and re-apply all these corrections 
        ### then we do the new masked version of the planet 
        ### in the case of other observations, this is the first time data gets processed so the reset doesn't actually
        ### have an effect 
        self.ts.reset()
        self.ts.correct_tau(self.atm_function)
        self.ts.restrict_detectors(self.good_xf,det_coord_mode='xf') ### throw out bad fits again
        self.ts.remove_obs_flag() # Get rid of points where the telescope wasn't in an "observing" mode
        self.ts.flag_scans() # Identify individual scans across the map
        self.ts.remove_scan_flag() # If data didn't appear to belong to a scan, drop it
        self.ts.remove_short_scans(thresh=self.thresh) # Remove anything that appears to short to be a real scan
        self.ts.remove_end_scans(n_start=self.n_edge_start,n_finish=self.n_edge_finish) # Remove the first few scans and last few scans (ie top and bottom of the map)
        self.ts.remove_scan_edge(n_start=self.n_scan_start, n_finish=self.n_scan_finish) # Remove a few data points from the edges of the scan, where the telescope may still be accelerating
        
        self.ts.filter_scan(use_offsets=True,mode='mask', n=self.filter_deg,nscans=0,mask_ra_bins=ra_bins, mask_dec_bins=dec_bins, mask=ra_dec_mask, make_stds=True, save_dir=self.save_path + 'timestreams/')

        print('data',self.ts.data.shape)
        print('data copy',self.ts.data_copy.shape)
        ### before correcting for feed offset, do commmon mode removal
        remove_common_mode = common_mode_params['remove_cm']
        if remove_common_mode:
            print('----------------------------------')
            print('--- removing common mode noise ---')
            print('----------------------------------')
            
            #self.ts.no_cm_template_data = self.ts.data_copy.copy()
            self.ts.remove_common_mode_noise(save_path = self.save_path,pixel=self.pixel,common_mode_params=common_mode_params)
            

        #self.ts.filter_scan(use_offsets=True,mode='mask', n=self.filter_deg,nscans=0,mask_ra_bins=ra_bins, mask_dec_bins=dec_bins, mask=ra_dec_mask, make_stds=True, save_dir=self.save_path + 'timestreams/')

        # if self.debugging_plots:
        #     for idx in range(len(self.ts.header['xf_coords'])):
        #         xf = self.ts.header['xf_coords'][idx]
        #         x = xf[0]; f = xf[1]
        #         ra, dec = self.ts.offset_pos(x) ### Save the estimated STDs
        #         idx = self.ts.Maps.get_xf(x,f)
        #         fig,axs = plt.subplots(1,figsize=(12,12))
        #         plt.rcParams.update({'font.size':24})
        #         axs.set_ylabel('standard deviation [counts]')
        #         axs.set_xlabel('Declination [deg]')
        #         axs.scatter(dec, self.ts.scan_stds[idx]) #### by default this is dec... can be updated to match scan direction in future
        #         fig.savefig(self.save_path + 'masking_tests/standard_deviations_x%s_f%s_p%s' % (x,f,self.mc))
        #         plt.close(fig)  

        ### Finally make the variance weighted maps 
        self.ts.var_weighted_make_map(pixel=self.pixel, use_offsets=True, center=[cent_ra, cent_dec], dims=dims)
        # print('------------------------ TURNED OFF VAR WEIGHTED MAP FOR NOW BECAUSE GETTINGG NANS AND CANT DEUBG -------------------------------------')

        

        # self.ts.make_map(pixel=self.pixel, use_offsets=True, center=[cent_ra,cent_dec], dims=dims)

    def plot_scan_pattern(self):
        '''
        This just plots the scan patterns
        Parameters 
        ----------
        None
        Returns
        ----------
        None
        '''
        fig,axs = plt.subplots(1,3, figsize=(20,12))
        axs[0].plot(self.ts.t-self.ts.t[0], self.ts.ra)
        axs[0].set_xlabel('time since start of obs'); axs[0].set_ylabel('right ascension')
        axs[1].plot(self.ts.t-self.ts.t[0], self.ts.dec)
        axs[1].set_xlabel('time since start of obs'); axs[1].set_ylabel('declination')
        axs[2].plot(self.ts.ra, self.ts.dec)    
        axs[2].set_xlabel('right ascension'); axs[2].set_ylabel('declination')
        plt.savefig(self.save_path + 'pointing_vs_time_p%s' % self.mc)
        plt.close(fig)

    def plot_det_data(self, ident):
        '''
        This plots all of the images produced in the Maps.maps object as well as error maps and a hit map

        Parameters:
        -----------
        ident : (str)
        identifier str for saving the images
        '''
        if not os.path.isdir(self.save_path + 'three_panels/'):
            os.mkdir(self.save_path+'three_panels/')
        x_vals = self.ts.Maps.dx_edge + self.cr; y_vals = self.ts.Maps.dy_edge + self.cd
        for idx in range(len(self.ts.header['xf_coords'])):
            xf = self.ts.header['xf_coords'][idx]
            x = xf[0]; f = xf[1]
            idx_map = self.ts.Maps.maps[idx]
            idx_emap = self.ts.Maps.e_maps[idx]
            idx_hmap = self.ts.Maps.h_maps[idx]
            fig, axs = plt.subplots(1,3, figsize=(16,8),sharey=True)
            plt.rcParams.update({'font.size':18})
            axs[0].set_title('Flux Map'); axs[1].set_title('Error Map'); axs[2].set_title('Hit Map')
            (vmin,vmax,cmap) = (np.nanmin(idx_map),np.nanmax(idx_map),'plasma')
            im = axs[0].pcolormesh(x_vals, y_vals,idx_map,vmin=vmin, vmax=vmax,cmap='plasma')
            axs[0].axis('equal')
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, label='Flux [Jy/B]', cax=cax)
            finite_inds = np.isfinite(idx_emap)
            im = axs[1].pcolormesh(x_vals, y_vals,idx_emap,cmap='plasma')
            axs[1].axis('equal')
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, label='Error [Jy/B]', cax=cax)
            (vmin,vmax,cmap) = (np.nanmin(idx_hmap),np.nanmax(idx_hmap),'plasma')
            im = axs[2].pcolormesh(x_vals, y_vals,idx_hmap,cmap='plasma', vmin=vmin, vmax=vmax)
            axs[2].axis('equal')
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, label='hits per bin', cax=cax)
            fig.savefig(self.save_path + 'three_panels/%s_three_panel_x%s_f%s_p%s' % (ident,x,f,self.mc), bbox_inches='tight')
            plt.close(fig)

        ### this plots stacked images which can be used to assess offset issues
        if not os.path.isdir(self.save_path + 'stack_checks/'):
            os.mkdir(self.save_path+'stack_checks/')
        x_list = np.array([xf[0] for xf in self.ts.header['xf_coords']])
        f_list = np.array([xf[1] for xf in self.ts.header['xf_coords']])
        if self.ts.header['object'] in self.planet_list:
            for f in range(60):
                fig = plt.figure(figsize=(8,8))
                f_inds = np.where(f_list == f)
                sub_list_f = f_list[f_inds]; sub_list_x = x_list[f_inds]
                for i in range(len(sub_list_f)):
                    fi = sub_list_f[i]
                    xi = sub_list_x[i]
                    x_inds = np.where(x_list == xi)
                    idx = np.intersect1d(x_inds, f_inds)[0]
                    mapi = self.ts.Maps.maps[idx].copy()
                    mapi /= np.nanmax(mapi)
                    if i == 0:
                        f_map = mapi
                    else:
                        f_map += mapi 
                if np.sum(f_inds) > 0:
                    plt.imshow(f_map, origin='lower', cmap='terrain')
                    plt.colorbar(label='Beam Peak Normed to 1')
                    plt.savefig(self.save_path + 'stack_checks/f%s_stack_check_p_%s' % (f,self.mc))
                    plt.close(fig)
                else:
                    plt.close(fig)

    def circular_mask(self, center, radius):
        '''
        Exactly what it says on the tin, can be used for setup_masked_filter 
        
        Center (list of floats): center ra and dec for mask 
        radius (float) : radius of mask in degrees
        '''
        ### exactly what it says on the tin, can be used in 
        ndets = len(self.ts.header['xf_coords'])
        xx,yy = np.meshgrid(self.ts.Maps.x_edge,self.ts.Maps.y_edge)
        ra_bins = np.zeros((ndets,xx.shape[1]))
        dec_bins = np.zeros((ndets,xx.shape[0]))
        ra_dec_mask = np.zeros((ndets,xx.shape[0], xx.shape[1])) 
        idx = 0 
        rr = np.sqrt((xx-center[0])**2 + (yy-center[1])**2)
        mask = rr < radius
        ra_dec_mask[:,mask] = True
        self.ra_dec_mask = ra_dec_mask 
        return ra_dec_mask
    
    def all_valid_mask(self):
        '''
        This is equivalent to saying there is no mask, can be helpful in debugging sometimes 

        '''
        ndets = len(self.ts.header['xf_coords'])
        xx, yy = np.meshgrid(self.ts.Maps.x_edge, self.ts.Maps.y_edge)
        ra_dec_mask = np.zeros((ndets,xx.shape[0], xx.shape[1]))
        self.ra_dec_mask = ra_dec_mask
        return ra_dec_mask

    def many_circle_mask(self, params):
        # print(centers, radii)
        """
        Generate a mask for multiple circular regions defined by centers and radii.
    
        Parameters:
        centers : list of tuples
            List of (x, y) coordinates for circle centers.
        radii : list of floats
            Corresponding radii for each center.
    
        Returns:
        ra_dec_mask : ndarray
            Boolean mask with the same dimensions as the grid, where True indicates masked regions.
        """
        centers = params[0]
        radii   = params[1]
        
        ndets = len(self.ts.header['xf_coords'])
        xx, yy = np.meshgrid(self.ts.Maps.x_edge, self.ts.Maps.y_edge)
        ra_dec_mask = np.zeros((ndets, xx.shape[0], xx.shape[1]), dtype=bool)
    
        for center, radius in zip(centers, radii):
            rr = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
            mask = rr < radius
            ra_dec_mask[:, mask] = True  
    
        self.ra_dec_mask = ra_dec_mask
        return ra_dec_mask



class One_D_Maps_Handler():
    def __init__(self, path, obj_override='None', date_override=False, tau_override=False, scan_override = 'None',debugging_plots=False, save_path='/data/vaughan/inverse_variance_weighted_maps/maps_testing/',thresh=100,n_edge_start=4,n_edge_finish=4,n_scan_start=200,n_scan_finish=200,filter_deg=5,pixel=(0.0055,0.0055),tau_path='/data/time/tau_forecasts/', verbose=False):
        """ Initializer for handling weighted maps
        
        Parameters
        ----------
        path : str 
        The path to the netcdf files for the observation we want to analyze 
        obj_override : str
        over-rides the object name of our observation, this is necessary for engineering data as meta 
        data failed to be properly uploaded to the netcdffiles. 
        date_override : str 
        depreciated
        tau_override : str 
        A value of tau that will be used to apply atmospheric opacity corrections, in the future this function 
        needs to be updated to either read it directly from the netcdf file or from the logbook
        debugging plots : bool
        True saves debugging plots to the save_path directory (so make sure save_path exists!), default is False
        save_path : str
        path to where you want debugging plots to be saved if debugging_plots = True, default is None 

        Returns
        -------
        None
        """

        self.planet_sizes = {'jupiter' : 69911e3,
                             'mars' : 3390e3,
                             'uranus': 25559e3,
                             'venus': 6025e3} #units of m, source JPL / NASA
        # good_det_list = np.loadtxt('1644101949_bad_dets.csv', delimiter=',')
        # good_xf = []
        # for f in range(60):
        #     for x in range(16):
        #         if good_det_list[f,x] == 1:
        #             good_xf.append((x,f))

        self.ts = Timestream(path, mc=0, store_copy=True,impose_frame='J2000_hard', xf=[(14,53)])
        

        # exit()
        # print(self.ts.dec, 'starting dec')
        # print(self.ts.dec_copy, 'dec copy')
        # exit()

        if obj_override != 'None':
            self.ts.header['object'] = obj_override.casefold()
        if date_override:
            
            # timestamp = int(os.path.basename(path))
            timestamp = 1643886151
            dtime = datetime.datetime.fromtimestamp(timestamp)
            date_str = dtime.strftime("%Y%m%d")
            self.ts.header['date'] = date_str
        # if tau_override:
        #     # timestamp = int(os.path.basename(path))
        #     timestamp = 1643886151
        #     dtime = datetime.datetime.fromtimestamp(timestamp)
        #     date_str = dtime.strftime("%Y%m%d"); time_str = dtime.strftime("%H:%M:%S")
        #     tau_override = get_tau(date_str, time_str,tau_path)

        #     print(tau_override, date_str, 'this is TAU')
        #     # print(date,date[0:10], date[11:19])
        #     self.ts.header['tau'] = tau_override
        #     self.ts.set_tau(tau_override) ### in the futurethis can be read in from sql/csv file...
        #     self.ts.tau_copy = self.ts.tau
        #     self.tau_override = tau_override
        timestamp = int(os.path.basename(path))
        # timestamp = 1643886151
        dtime = datetime.datetime.fromtimestamp(timestamp)
        date_str = dtime.strftime("%Y%m%d"); time_str = dtime.strftime("%H:%M:%S")
        tau_override = get_tau(date_str, time_str,tau_path)

        # print(date,date[0:10], date[11:19])
        self.ts.header['tau'] = tau_override
        self.ts.set_tau(tau_override) ### in the futurethis can be read in from sql/csv file...
        self.ts.tau_copy = self.ts.tau
        self.tau_override = tau_override
        self.atm_function = setup_simple_atm_model()
        self.ts.correct_tau(self.atm_function)




            # self.atm_function = setup_simple_atm_model()
        if scan_override != 'None':
            self.ts.header['scan_pars']['direction'] = scan_override
        self.save_path = save_path 
        self.debugging_plots = debugging_plots
        self.thresh = thresh; self.n_edge_start = n_edge_start; self.n_edge_finish=n_edge_finish 
        self.n_scan_start = n_scan_start; self.n_scan_finish = n_scan_finish; self.filter_deg = filter_deg
        self.pixel = pixel 
        self.planet_list = ['Jupiter','uranus','mars','jupiter','Mars', 'Uranus']
        self.quasar_list = ['3C454.3','3C273','3C279','3C84']
             
    def apply_first_det_cuts(self, verbose = False):
        """ This function doesn't do anything special, it might not even need to be a function but what it does is it removes
        all images that are filled with zeros or filled with nans. This first step saves a lot of computation time by not having to deal
        with detector data that is unpopulated. It also saves the good detectors for later use. 
        
        Parameters
        ----------
        None
        Returns
        -------
        None

        """
        if self.ts.data.shape[0] == 16:
            self.ts.data = self.ts.data.reshape(16 * 60, self.ts.data.shape[2])
        ### first set of data cuts
        #print(self.ts.data.shape)
        nancheck = np.all(np.isnan(self.ts.data),axis=1)
        zerocheck = np.all(self.ts.data==0,axis=1)
        idx = np.nonzero((~nancheck) & (~zerocheck))[0].astype('int')
        #print(self.ts.header['xf_coords'], 'original xf coords')
        #print(self.ts.header['xf_coords'][0][0], self.ts.header['xf_coords'][0][1])
        # exit()
        #print(idx, 'idx')
        #print(len(self.ts.header['xf_coords']), 'len of coords')
        # print(self.ts.header['xf_coords'])
        #print('testing printing...')
        # for i in range(960):
        #     print(self.ts.header['xf_coords'][i])
        # print(self.ts.header['xf_coords'][0])
        #self.ts.header['xf_coords'] = np.array(self.ts.header['xf_coords'])
        # print(self.ts.header['xf_coords'].shape, 'array shape')
        # self.ts.header['cr_coords'] = np.array(self.ts.header['cr_coords'])
        #print(idx.shape)
        det_idx = self.ts.header['xf_coords'][idx]
        # print(self.ts.header['xf_coords'][idx])
        #print(det_idx, 'det_idx')
        #print(det_idx.size, self.ts.data.shape)
        # exit()
        self.ts.restrict_detectors(det_idx, det_coord_mode='xf')
        ### store a list of good_xf coordinates to be passed onto to other datasets if necessary, e.g. in the case where we are 
        ### using both a calibrating source as well as a science source. 
        self.good_x = [xf[0] for xf in det_idx]; self.good_f = [xf[1] for xf in det_idx]
        self.good_xf = [(x,f) for x,f in zip(self.good_x, self.good_f)]

    def apply_unmasked_filtering(self, filtering=True):
        """ Creates un-weighted error maps with a generic peak filtering algorithm

        Parameters, note that there are no parameters that get passed to this function, they are provided as inputs to the variance_maps_handler class. 
        ----------
        thresh : int
        the minimum number of data points in a scan before it is considered "short" and removed from the data set.
        This is to remove spurious data that may contaminate our results 
        n_edge_start : int 
        The number of scans to remove scans from the bottom of the map 
        n_edge_finish : int 
        The number of scans to remove scans from the top of the map
        n_scan_start: int 
        The number of scans to remove from the left side of the map 
        n_scan_finish: int 
        The number of scans to remove from the right side of the map 
        filter_deg : int 
        The polynomial degree used for atmospheric filtering 
        pixel : tuple 
        the x and y pixelsizes in arcseconds default to (0.006, 0.006)
        Returns
        -------
        None
        """
        
        ### do filtering         
        self.ts.remove_obs_flag() # Get rid of points where the telescope wasn't in an "observing" mode
        self.ts.flag_scans() # Identify individual scans across the map

        #print(self.ts.scan_flags, 'scan_flags after flag_scans()')
        self.ts.remove_scan_flag() # If data didn't appear to belong to a scan, drop it

        #print(self.ts.scan_flags, 'remove scan flag')
        self.ts.remove_short_scans(thresh=self.thresh) # Remove anything that appears to short to be a real scan

        #print(self.ts.scan_flags, 'remove short flags')
        # self.ts.remove_end_scans(n_start=self.n_edge_start,n_finish=self.n_edge_finish) # Remove the first few scans and last few scans (ie top and bottom of the map)
        #print(self.ts.scan_flags, 'remove end scans')

        self.ts.remove_scan_edge(n_start=self.n_scan_start, n_finish=self.n_scan_finish) # Remove a few data points from the edges of the scan, where the telescope may still be accelerating
        if filtering:
            self.ts.filter_scan(n=self.filter_deg)


        print(self.quasar_list, self.ts.header['object'])
        if self.ts.header['object'] in self.planet_list:
            use_offsets = False
        elif self.ts.header['object'].upper() in self.quasar_list:
            print('?')
            use_offsets=False
        else:
            use_offsets=True           
        print('made naive map with offsets %s' % use_offsets) 
        self.ts.make_1d(pixel=self.pixel[0], use_offsets=use_offsets)
        # for xf in self.ts.header['xf_coords']:
            # self.ts.LineMaps.plot(xf,det_coord_mode='xf',savepath=self.save_path + '%sx_%sf' % (xf[0],xf[1]),show=False,)
        #print(self.ts.header['xf_coords'], 'xf coords after unmasked filtering and naive map creation')
        ### get some pointing information for later, this is only used for debugging plots  

    def get_and_plot_individual_scans(self):

        if not os.path.isdir(self.save_path + 'ind_scans/'):
            os.mkdir(self.save_path + 'ind_scans/')

        fig, axs = plt.subplots(1, figsize=(6,6))

        med_ra = np.nanmedian(self.ts.ra)
        med_dec = np.nanmedian(self.ts.dec)

        print(len(np.unique(self.ts.scan_flags)))

        for i in np.unique(self.ts.scan_flags):
    
            scan_inds = np.where(self.ts.scan_flags == i)[0]
            # axs.scatter( (self.ts.ra[scan_inds]-med_ra) *3600, (self.ts.dec[scan_inds]-med_dec)*3600, label='scan id %s' % i)
            axs.scatter( self.ts.ra[scan_inds], self.ts.dec[scan_inds],  c=self.ts.data[0,scan_inds],label='scan id %s' % i)

            axs.set_xlim(187.27,187.285)
            axs.set_xlabel('RA [as]')
            axs.set_ylabel('DEC [as]')
            # axs.legend(framealpha=0)
        fig.savefig(self.save_path + 'ind_scans/aa_scan_pattern_p%s' % self.mc, bbox_inches='tight')

        root = int(np.ceil(np.sqrt(len(np.unique(self.ts.scan_flags)))))

        # xf_list = [(3,15),(3,19),(6,23),(9,34)]
        for xf in self.ts.header['xf_coords']:
        # for xf in xf_list:
            x=xf[0]; f=xf[1]
            fig2, axs2 = plt.subplots(1,figsize=(6,6))
            
            index = self.ts._get_coord(x,f,det_coord_mode='xf')
            fig, axs = plt.subplots(root,root,sharex=True,sharey=True)
            # axs.plot(self.ts.t, self.ts.data[index], color='red')
            # axs.plot(self.ts.ra, self.ts.data[index], color='red')
            for i in np.unique(self.ts.scan_flags):
                print(i)
                xi = i // root
                yi = i % root
                # fit_inds = np.where(np.isin(self.ts.scan_flags,[i]+[i+j+1 for j in range(nscans)]+[i-j-1 for j in range(nscans)]))[0]
                scan_inds = np.where(self.ts.scan_flags == i)[0]
                axs[xi,yi].plot(self.ts.ra[scan_inds],self.ts.data[index,scan_inds], color='black')
                # print(index, 'index')
                # axs[xi,yi].set_title('Scan id %s' % i)
                # axs.set_xlabel('Scan Time')
                # axs[xi,yi].set_xlabel('RA')
                # axs[xi,yi].set_ylabel('Flux [arb Counts]')
                # axs[xi,yi].set_xlim(187.27,187.285)
                axs[xi,yi].set_xticklabels([]); axs[xi,yi].set_yticklabels([])
                axs[xi,yi].set_xticks([]); axs[xi,yi].set_yticks([])

                axs2.plot(self.ts.ra[scan_inds],self.ts.data[index,scan_inds], color='black', alpha=0.2)
            fig2.savefig(self.save_path + 'ind_scans/all_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')
            fig.savefig(self.save_path + 'ind_scans/x%s_f%s_ind_scan_p%s' % (x,f,self.mc), bbox_inches='tight')
            plt.close(fig)

 

    def apply_beam_fits(self, ident=None, sigma=False, red_chi_lower=0.1, red_chi_upper=1e4, amp_cut=10, snr_cut=5, verbose = False, rms_cut = 0.8):
        """ Wrapper that fits a gaussian function to each detector, this finds the center of the planet, which can be used to backout
        offset corrections. It first fits to the data then cuts detectors with bad fits, then refits the data. It then does some cuts based off of reduced chi squared and signal to noise ratios.
        Parameters
        ----------
        ident : (str) an identifier for the debugging plots that gets appended to the saved maps directory
        sigma : (bool) a boolean operator that indicates whether or not to use error weighting in the fits or not.
        red_chi_lower: (float) a lower limit on the reduced chi squared cuts
        red_chi_upper: (float) an upper limit on the reduced chi squared cuts
        amp_cut: (float) a minimum threshold that the best fit amplitude has to be above
        snr_cut: (float) a minimum threshold for the SNR
        Returns
        -------
        fits : (array) (n_det, n_param) an array of the best fit parameters
        covs : (array) (n_det, n_param,n_param) covariance matrices for the best fit parameters 
        """

        # print(self.ts.header['xf_coords'], 'xf_coords in apply beam fits')
        ### this should only ever be used if we are looking at point sources, e.g. planets or quasars.
        fit_function,fits,covs = self.ts.LineMaps.beam_fit(nofit_handling='allow')#,sigma=sigma, verbose=verbose)
        ### need to add a flag here to pick which fit function to use - Ben 
        # fit_function,fits,covs = self.ts.Maps.beam_fit_plus_gradient(nofit_handling='allow')#,sigma=sigma)
        bad_fit_inds = np.nonzero(~np.isnan(fits[:,0]))[0]  #throw out bad fits
        self.bad_fit_dets = self.ts.header['xf_coords'][bad_fit_inds]
        self.ts.restrict_detectors(self.bad_fit_dets,det_coord_mode='xf');self.ts.LineMaps.restrict_detectors(self.bad_fit_dets, det_coord_mode='xf')

        fit_function,fits,covs = self.ts.LineMaps.beam_fit(nofit_handling='allow')#,sigma=sigma, verbose=verbose)

        ### update the list of good xf coordinates to include the results from the fitting as well ! 
        bad_x2 = [xf[0] for xf in self.bad_fit_dets]; bad_f2 = [xf[1] for xf in self.bad_fit_dets]
        self.good_x = np.intersect1d(self.good_x, bad_x2); self.good_f = np.intersect1d(self.good_x, bad_f2)
        self.good_xf = [(x,f) for x,f in zip(self.good_x, self.good_f)]

        ### make some directories to save debugging plots.
        if not os.path.isdir(self.save_path + 'good_dets/'):
            os.makedirs(self.save_path + 'good_dets')
        if not os.path.isdir(self.save_path + 'bad_dets/'):
            os.makedirs(self.save_path + 'bad_dets')
        ### make some data holders for the reduced chi squared and the SNR ratio of the plots
        red_chi_map = np.zeros((16,60))
        red_chi_map[:,:] = np.nan
        amp_over_std_map = np.zeros((16,60))
        amp_over_std_map[:,:] = np.nan
        bad_chis = np.zeros((16,60))
        bad_chis[:,:] = np.nan
        bad_amps = np.zeros((16,60))
        bad_amps[:,:] = np.nan
        map_of_kept_rem_dets = np.zeros((16,60))
        map_of_kept_rem_dets[:,:] = np.nan
        ### we will be updating the list of good xf coordinates again 
        good_list = []

        #### produce quality control plots!
        map_copy = self.ts.LineMaps.linemaps.copy()
        ### calculate a meshgrid of ra/dec values 
        ### note that dx center and dy center are the ra/dec edges - their center 
        # xx,yy = np.meshgrid(self.ts.Maps.dx_center,self.ts.Maps.dy_center)
        xx = self.ts.LineMaps.x_center 

        resid_amps = np.zeros((16,60))
        peak_amps = np.zeros((16,60))
        for xf in self.ts.header['xf_coords']:
            x = xf[0]; f = xf[1]
            #print('on x %s f %s' % (x,f))
            if self.debugging_plots:
                ### set up some plotting parameters for debugging plots.


                fig,axes = plt.subplots(2,1,figsize=(20,20),sharey=True)
                w = .8
                h = .8
                axes[0].set(ylabel='Flux [Arb Counts]')
                axes[1].set(ylabel='Flux [Arb Counts]')
                axes[1].set(xlabel='X Offset from Center [arcsec]')



            ### create models and compute chi squared / SNR 
            idx = self.ts.LineMaps.get_xf(x,f)
            detector_map = map_copy[idx]
            model_map = self.ts.LineMaps.beam_fit_results['function']((xx,np.ones(xx.size)),*self.ts.LineMaps.beam_fit_results['fits'][idx])
            print(model_map.shape)
            bfits = self.ts.LineMaps.beam_fit_results['fits'][idx].copy()
            print(len(self.ts.LineMaps.beam_fit_results['fits']), len(self.ts.header['xf_coords']))
            print(x,f, bfits[0], bfits[1],bfits[2]*3600, 'fits')#, bfits[4]*3600)
            frequency = f_ind_to_val(f)
            theoretical_size = 1.2 * 2.998e8/frequency/1e9 / 12 * 180/np.pi*3600
            ### circular mask is used to compute the background signal which is the denominator of the SNR
            rr =  (xx-bfits[1]) / (3*bfits[2])##+2*bfits[3]))**2 + ((yy - bfits[2])/ (2*bfits[3]))**2
            circ_mask = rr < 1
            det_copy = detector_map.copy()
            det_copy[circ_mask] = np.nan

            resid_amps[x,f] = np.nanmax(detector_map - model_map)
            peak_amps[x,f] = np.nanmax(detector_map)
            detector_map = map_copy[idx]
            ### circular mask is used to compute the background signal which is the denominator of the SNR
            amp_over_std_map[x,f]= bfits[0] / np.nanstd(det_copy)


            if sigma:
                red_chi = np.nansum( (detector_map - model_map)**2 / self.ts.Maps.e_maps[idx]**2) / detector_map.size
            else:
                red_chi = np.nansum( (detector_map - model_map)**2 / np.nanstd(det_copy)**2) / detector_map.size
                # red_chi = bfits[0] / np.nanstd(detector_map - model_map)

            ### rms_check 
            dmap_rms = np.sqrt( np.nansum(detector_map**2) / detector_map.size)
            resid_rms = np.sqrt(np.nansum((detector_map - model_map)**2) / detector_map.size)
            rms_bool = resid_rms / dmap_rms < rms_cut
            print(rms_bool, resid_rms, dmap_rms, rms_cut, 'bool resid, dmap, cut')
            #just check 1d for 1d...
            # red_chi_flag = bfits[2]*3600 > theoretical_size #and (bfits[4]*3600 > theoretical_size) #red_chi < 1e5 and red_chi > 1
            red_chi_flag = True
            if ~red_chi_flag:
                bad_chis[x,f] = 1
                map_of_kept_rem_dets[x,f] = 1
            amp_flag = bfits[0] > amp_cut and amp_over_std_map[x,f] > snr_cut
            print(amp_over_std_map[x,f], 'amp over std', amp_flag, 'amp_flag')
            good_flag = red_chi_flag and amp_flag and rms_bool
            print(good_flag)
            if ~amp_flag:
                bad_amps[x,f] = 1
                map_of_kept_rem_dets[x,f] = 2
            if sigma:
                good_flag = True
            if good_flag:
                good_list.append((x,f))
                map_of_kept_rem_dets[x,f] = 3

            red_chi_map[x,f] = red_chi
            if self.debugging_plots:
                axes[0].set(title='x%sf%s, $\\chi^2$=%.3F' % (x,f,red_chi))
                axes[0].plot(xx,detector_map, label='data', color='black')
                axes[0].plot(xx,model_map, label='best fit', color='red', linestyle='dashed')
                # axes[0].set_xlim(187.4,187.2)
                # axes[1].set_xlim(187.4,187.2)
                axes[0].set_xlim(bfits[1]-0.0277778,bfits[1]+0.0277778)
                # axes[1].set_xlim(bfits[1]-0.0277778,bfits[1]+0.0277778)

                axes[0].legend()
                axes[1].plot(xx, detector_map-model_map)
                # ps = np.abs(np.fft.fft(detector_map))**2
                # time_step = 1 / 30
                # freqs = np.fft.fftfreq(detector_map.size, 1/np.median(np.diff(xx)))
                # idx = np.argsort(freqs)
                # axes[2].plot(freqs[idx], ps[idx], label='Data FFT')
                # ps = np.abs(np.fft.fft(detector_map - model_map))**2
                # time_step = 1 / 30
                # freqs = np.fft.fftfreq(detector_map.size, 1/np.median(np.diff(xx)))
                # idx = np.argsort(freqs)
                # axes[2].plot(freqs[idx], ps[idx], label='Residual FFT')#plt.plot(xf, )
                # axes[2].legend()

                if good_flag:
                    fig.savefig(self.save_path + 'good_dets/qual_assurance_' + ident + '_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')
                else:
                    fig.savefig(self.save_path + 'bad_dets/qual_assurance_' + ident + '_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')
                plt.close(fig)


        if sigma: 
            path = self.save_path + 'red_chi_eweighted'
            path2 = self.save_path + 'amp_over_std_eweighted'
        else:
            path = self.save_path + 'red_chi'
            path2 = self.save_path + 'amp_over_std'
        
        # plot_maps(path, data=red_chi_map, markers=None, title='',
        #     clabel='Reduced $\chi^2$', cmin=None, cmax=None, scale=1.0, missing_color='grey',
        #     cmap='rainbow', mux_space=False, marker_labels=None,
        #     cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
        #     logscale=True)
        # plot_maps(path2, data=amp_over_std_map, markers=None, title='',
        #     clabel='SNR', cmin=None, cmax=None, scale=1.0, missing_color='grey',
        #     cmap='rainbow', mux_space=False, marker_labels=None,
        #     cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
        #     logscale=True)
        # plot_maps(self.save_path + 'bad_chi', data=bad_chis, markers=None, title='',
        #     clabel='Removed From Chi Squared', cmin=None, cmax=None, scale=1.0, missing_color='grey',
        #     cmap='rainbow', mux_space=False, marker_labels=None,
        #     cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
        #     logscale=False)   
        # plot_maps(self.save_path + 'bad_amps2', data=bad_amps, markers=None, title='',
        #     clabel='Removed From Amps', cmin=None, cmax=None, scale=1.0, missing_color='grey',
        #     cmap='rainbow', mux_space=False, marker_labels=None,
        #     cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
        #     logscale=False)       
        # plot_maps(self.save_path + 'kept_and_removed', data=map_of_kept_rem_dets, markers=None, title='',
        #     clabel='Detectors Kept or Removed', cmin=None, cmax=None, scale=1.0, missing_color='grey',
        #     cmap='rainbow', mux_space=False, marker_labels=None,
        #     cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
        #     logscale=False)  
       

        fig = plt.figure(figsize=(8,8))
        plt.hist(red_chi_map.flatten(), histtype='step', bins=np.logspace(np.log10(np.nanmin(red_chi_map)),np.log10(np.nanmin(red_chi_map))))
        plt.xscale('log')
        plt.xlabel('Reduced $\\chi^2$')
        fig.savefig(path + '_hist_p%s' % self.mc)
        plt.close(fig)

        ### refit after extra cuts
        if ~sigma:
            # print('this code block ran!!!')
            # self.ts.restrict_detectors(good_list, det_coord_mode='xf'); self.ts.LineMaps.restrict_detectors(good_list, det_coord_mode='xf')
            self.good_xf = good_list
            print('final set of beam fits')
            self.fit_function,self.fits,covs = self.ts.LineMaps.beam_fit(nofit_handling='allow')#,sigma=sigma)
        return self.fits,covs, resid_amps, peak_amps


    def find_planet_each_det(self, max_ra_diff=1e8, max_dec_diff = 1e8):
        """Finds the location of a planet in the scan for each individual detector which is used to compute delta RA/DEC and cuts detectors with a positional
        difference above max_ra_diff or max_dec_diff.

        Returns
        -------
        planet_centers : array 
        ' a (ndetector, 2) of the RA/DEC pointing of the planet
        max_ra_diff : float
        the max difference between the measured pointing from the beam fit and the epheremis RA, default is 1e8 (e.g. no restriction)
        the max difference between the measured pointing from the beam fit and the epheremis DEC, default is 1e8 (e.g. no restriction)
        """
        # make holding arrays to hold data that has been calculated
        self.planet_centers = np.zeros((len(self.ts.header['xf_coords']),2))
        self.planet_az_el = np.zeros((len(self.ts.header['xf_coords']),2))
        planet_ra_grid = np.zeros((16,60))
        planet_dec_grid = np.zeros((16,60))
        planet_az_grid = np.zeros((16,60)) 
        planet_el_grid = np.zeros((16,60))
        planet_ra_grid[:,:] = np.nan 
        planet_dec_grid[:,:] = np.nan 
        planet_az_grid[:,:] = np.nan 
        planet_el_grid[:,:] = np.nan

        good_xf_list = [] 
        
        if not os.path.isdir(self.save_path + 'pointing_tests/'):
            os.mkdir(self.save_path+'pointing_tests')

        self.bf_centers = np.zeros((len(self.ts.header['xf_coords']),2))
        for idx in range(len(self.ts.header['xf_coords'])):
            xf = self.ts.header['xf_coords'][idx]
            x = xf[0]; f = xf[1]
            # feed_offset_x, feed_offset_y = self.ts.Maps.get_feed_offsets(x,f,'xf')
            # ra, dec = self.ts.offset_pos(x) ### extract ra, dec timestreams that have been corrected for feed offsets.
            # peak_ra = self.fits[idx][1] + self.cr - feed_offset_x / self.ts.Maps.dx_scale_factor; peak_dec = self.fits[idx][2] + self.cd - feed_offset_y
            peak_ra = self.fits[idx][1] 
            peak_dec = np.nanmedian(self.ts.dec)
            ### it doesn't have to be perfect, just reasonably close in time...
            self.bf_centers[idx,0] = peak_ra; self.bf_centers[idx,1] = peak_dec
            print('before try...')
            try:
                t_ind = np.where(np.round(peak_ra,4) == np.round(self.ts.ra,4))#,4)))[0] ### something a little more rigorous than this is probably in order, does not account for cosmic rays / other issues
                timestamp = str(self.ts.t[t_ind][0])
            except IndexError:
                try:         
                    t_ind = np.where(np.round(peak_ra,3) == np.round(self.ts.ra,3))### something a little more rigorous than this is probably in order, does not account for cosmic rays / other issues
                    timestamp = str(self.ts.t[t_ind][0])
                except IndexError:
                    try:
                        t_ind = np.where(np.round(peak_ra,2) == np.round(self.ts.ra,2)) ### something a little more rigorous than this is probably in order, does not account for cosmic rays / other issues
                        timestamp = str(self.ts.t[t_ind][0])
                    except IndexError:
                        try:
                            t_ind = np.where(np.round(peak_ra,1) == np.round(self.ts.ra,1)) ### something a little more rigorous than this is probably in order, does not account for cosmic rays / other issues
                            timestamp = str(self.ts.t[t_ind][0])
                        except IndexError:
                            print('warning could not estimate peak timestamp from fits, using max value as planet location, x=%s f = %s' % (x,f))
                            t_ind = np.where(self.ts.data[idx] == np.nanmax(self.ts.data[idx]))[0][0]
                            timestamp = t_ind

            print('This where we at now')
            t = Time(timestamp, format='unix')
            loc = EarthLocation.of_site('Kitt Peak')
            if self.ts.header['object'] in self.planet_list:
                planet_eph = get_body(self.ts.header['object'], t, loc)
                earth_eph = get_body('earth',t, loc)              
                altaz_system = AltAz(obstime=t,location=loc)
                dist = earth_eph.separation_3d(planet_eph).to(u.m).value
                planet_rad = 69911e3 ### hard coded for a specific value should have some helper function that has planet information in it that can be drawn into for htis
                self.angular_rad = np.rad2deg(planet_rad / dist)
                # planet_eph = planet_eph.transform_to(apcoord.PrecessedGeocentric(equinox=t))
                planet_ra = float(planet_eph.ra.to_string(decimal=True))
                planet_dec = float(planet_eph.dec.to_string(decimal=True))
                planet_az_el = planet_eph.transform_to(apcoord.AltAz(obstime=t,location=loc))
                planet_az = float(planet_az_el.az.to_string(decimal=True))
                planet_el = float(planet_az_el.alt.to_string(decimal=True))  
            else:
                if self.ts.header['object'].upper() == '3C273':
                    planet_ra = 187.277915542; planet_dec = 2.05238833333
                elif self.ts.header['object'].upper() == '3C279':
                    planet_ra = 194.046527375; planet_dec = -5.78931255556
                elif self.ts.header['object'].upper() == '3C454.3':
                    planet_ra =  343.490623917; planet_dec =  16.1482113611
                elif self.ts.header['object'].upper() == '3C84':
                    planet_ra = 49.950667125; planet_dec = 41.5116960278          
                plan_coord = SkyCoord(planet_ra * u.deg, planet_dec * u.deg)

                planet_az_el = plan_coord.transform_to(apcoord.AltAz(obstime=t,location=loc))
                planet_az = float(planet_az_el.az.to_string(decimal=True))
                planet_el = float(planet_az_el.alt.to_string(decimal=True))  
            self.planet_centers[idx,:] = np.array([planet_ra, planet_dec])
            peak_coords = SkyCoord(peak_ra * u.deg, peak_dec * u.deg)
            az_el_peak = peak_coords.transform_to(apcoord.AltAz(obstime=t,location=loc))
            peak_az = float(az_el_peak.az.to_string(decimal=True))
            peak_el = float(az_el_peak.alt.to_string(decimal=True))   
            # peak_coords = SkyCoord(ra = peak_ra * u.deg, dec = peak_dec * u.deg, frame=apcoord.PrecessedGeocentric(equinox=t))
            # peak_azel = peak_coords.transform_to(altaz_system)
            # peak_az = float(peak_azel.az.to_string(decimal=True))
            # peak_el = float(peak_azel.alt.to_string(decimal=True))
            if self.debugging_plots:
                fig, axs = plt.subplots(1)
                cm = plt.cm.get_cmap('rainbow')
                # axs.axis('equal')
                vmin = np.nanmin(self.ts.data[idx]);vmax=np.nanmax(self.ts.data[idx])
                sc = axs.scatter(self.ts.ra, self.ts.data[idx])# vmin=vmin, vmax=vmax, cmap=cm)
                # divider = make_axes_locatable(axs)
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig.colorbar(sc, label='Jy',cax=cax)
                # axs.scatter(planet_ra, planet_dec, marker='v', s=25, color='black')
                axs.axvline(planet_ra, linestyle='dashed', color='black')
                axs.axvline(self.bf_centers[idx,0], color='black', linestyle='dotted')
                axs.legend()
                fig.savefig(self.save_path + 'pointing_tests/point_ts_x%s_f%s_p%s' % (x,f,self.mc), bbox_inches='tight')
                plt.close(fig)
            planet_ra_grid[x,f] = peak_ra - planet_ra; planet_dec_grid[x,f] = peak_dec - planet_dec 
            planet_az_grid[x,f] = peak_az - planet_az; planet_el_grid[x,f] = peak_el - planet_el 
            if np.abs(planet_ra_grid[x,f])*3600 < max_ra_diff and np.abs(planet_dec_grid[x,f])*3600 < max_dec_diff: # and np.abs(planet_az_grid[x,f])*3600 < 100 and np.abs(planet_el_grid[x,f])*3600 < 100:
                good_xf_list.append((x,f))
            else:
                planet_ra_grid[x,f] = np.nan; planet_dec_grid[x,f] = np.nan; planet_el_grid[x,f] = np.nan; planet_az_grid[x,f] = np.nan
        if self.debugging_plots:
            plot_maps(self.save_path +'pointing_tests/ra', data=planet_ra_grid * 3600, markers=None, title='',
                clabel='$\\Delta$RA [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path +'pointing_tests/el', data=planet_el_grid * 3600, markers=None, title='',
                clabel='$\\Delta$el [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path + 'pointing_tests/az', data=planet_az_grid * 3600, markers=None, title='',
                clabel='$\\Delta$Az [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path + 'pointing_tests/dec', data=planet_dec_grid * 3600, markers=None, title='',
                clabel='$\\Delta$DEC [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path +'pointing_tests/ra_med_sub', data=(planet_ra_grid-np.nanmedian(planet_ra_grid)) * 3600, markers=None, title='',
                clabel='$\\Delta$RA [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path +'pointing_tests/el_med_sub', data=(planet_el_grid-np.nanmedian(planet_el_grid)) * 3600, markers=None, title='',
                clabel='$\\Delta$el [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path + 'pointing_tests/az_med_sub', data=(planet_az_grid-np.nanmedian(planet_az_grid)) * 3600, markers=None, title='',
                clabel='$\\Delta$Az [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
            plot_maps(self.save_path + 'pointing_tests/dec_med_sub', data=(planet_dec_grid-np.nanmedian(planet_dec_grid)) * 3600, markers=None, title='',
                clabel='$\\Delta$DEC [arcsec]', cmin=None, cmax=None, scale=1.0, missing_color='grey',
                cmap='rainbow', mux_space=False, marker_labels=None,
                cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
                logscale=False)
        if len(good_xf_list) > 0:
            self.ts.restrict_detectors(good_xf_list, det_coord_mode='xf'); self.ts.LineMaps.restrict_detectors(good_xf_list, det_coord_mode='xf')
        self.good_xf = good_xf_list


        np.save(self.save_path + 'pointing_tests/planet_el', planet_el, allow_pickle=True)
        np.save(self.save_path + 'pointing_tests/planet_az', planet_az, allow_pickle=True)
        np.save(self.save_path + 'pointing_tests/el_vals',planet_el_grid, allow_pickle=True)
        np.save(self.save_path + 'pointing_tests/az_vals',planet_az_grid, allow_pickle=True)
        np.save(self.save_path + 'pointing_tests/ra_vals',planet_ra_grid, allow_pickle=True)
        np.save(self.save_path + 'pointing_tests/dec_vals',planet_dec_grid, allow_pickle=True)
        return self.planet_centers
