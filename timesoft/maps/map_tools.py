from timesoft.helpers import coordinates
from timesoft.helpers._class_bases import _Datastruct_base
from timesoft.helpers.planet_info import get_planet
from timesoft.calibration import Offsets
from skimage.draw import line

from inspect import signature
import os
import warnings
import copy

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import convolve, fftconvolve
import matplotlib.pyplot as plt

from astropy.time import Time

# SOME DATASET VARIABLES:
nchan = 60
ndet = 16

class Map(_Datastruct_base):
    #################################
    #### LOADING AND SAVING MAPS ####
    #################################
    def __init__(self,path,xf=None,cr=None):

        file_in = np.load(path, allow_pickle=True)

        if 'is_map' not in file_in.files:
            raise ValueError("The specified file does not appear to be a TIME map")
        self._check_header(file_in['header'][()])

        self.header = file_in['header'][()] # This weird syntax turns back into a dictionary

        self.x_edge = np.array(file_in['x_edge'])
        self.y_edge = np.array(file_in['y_edge'])

        self.hit_maps = np.array(file_in['h_maps'])
        self.maps = np.array(file_in['maps'])
        self.e_maps = np.array(file_in['e_maps'])
        self.counts = np.array(file_in['counts'])
        self.extras = np.array(file_in['extras'])[()]

        self.e_maps = np.array(file_in['e_maps'])

        self._input_consistency_check()

        if xf is not None or cr is not None:
            if xf is not None and cr is not None:
                raise ValueError('cannot specify both xf and cr indexing')

            elif xf is not None:  
                c1c2 = xf
                det_coord_mode = 'xf'
            elif cr is not None:
                c1c2 = cr
                det_coord_mode = 'cr'

            self.restrict_detectors(c1c2,det_coord_mode)

        self._construct_axes()
            

    def _construct_axes(self,):
        """Construct different useful representations of the axes"""

        self.x_center = np.array([(self.x_edge[i]+self.x_edge[i+1])/2 for i in range(len(self.x_edge)-1)])
        self.y_center = np.array([(self.y_edge[i]+self.y_edge[i+1])/2 for i in range(len(self.y_edge)-1)])

        self.dx_scale_factor = np.cos(self.header['map_pars']['y_center']*np.pi/180)
        self.dx_edge = (self.x_edge - self.header['map_pars']['x_center']) * self.dx_scale_factor
        self.dx_center = (self.x_center - self.header['map_pars']['x_center']) * self.dx_scale_factor
        self.dy_edge = self.y_edge - self.header['map_pars']['y_center']
        self.dy_center = self.y_center - self.header['map_pars']['y_center']


    def _input_consistency_check(self):
        """Check some inputs to __init__ make sense"""

        if not self.header['flags']['maps_initialized']:
            raise ValueError("Header flags indicate no maps initialized")

        # Check everything loaded looks good
        if self.maps.shape[2] != len(self.x_edge)-1:
            raise ValueError("Map dimensions do not match x-axis")
        if self.maps.shape[1] != len(self.y_edge)-1:
            raise ValueError("Map dimensions do not match y-axis")

        if self.counts.shape[1:] != self.maps.shape[1:]:
            raise ValueError("Map dimensions do not match counts dimensions")
        if self.counts.shape[0] != 16:
            raise ValueError("Number of counts arrays does not equal number of detectors (16)")
        extras_check = [self.extras[key].shape[1:] != self.maps.shape[1:] for key in self.extras.keys()]
        if np.any(extras_check):
            raise ValueError("Map dimesions do not match extras dimensions")
        extras_check = [self.extras[key].shape[0] != 16 for key in self.extras.keys()]
        if np.any(extras_check):
            raise ValueError("Number of extras sub-arrays does not equal number of detectors (16)")


    def write(self, filepath, compress=False, overwrite=True):
        """Save maps as a ``.npz`` file

        This will save the maps as a ``.npz`` file. 
        The saved files can be loaded with utilities 
        from the ``timesoft.maps`` module for further
        analysis.

        Parameters
        ----------
        filepath : str
            The path and filename where the data should
            be saved. ``.npz`` will be appended to the
            end of the file name if it is not already 
            included.
        compress : bool, default=False
            If True the file will be compressed (saved 
            with ``np.savez_compressed`` instead of 
            ``np.savez``).
        overwrite : bool, default=True
            Determines how the method behaves if a file
            already exists at the specified `filepath`.
            If True, the prior file is overwritten. If 
            False, an error is raised. This is to help
            avoid overwriting files by accident.

        Returns
        -------
        None        
        """

        if compress:
            savefunc = np.savez_compressed
        else:
            savefunc = np.savez

        if os.path.exists(filepath) and not overwrite:
            raise ValueError("The file you are trying to create already exists")

        print('writing')

        savefunc(filepath, is_map=True, 
                 header=self.header,x_edge=self.x_edge,y_edge=self.y_edge,maps=self.maps,e_maps=self.e_maps, h_maps=self.h_maps,counts=self.counts,extras=self.extras)


    def restrict_detectors(self,new_c1c2,det_coord_mode='xf'):
        """Limit dataset to a subset of the current detectors
        
        This method removes all but a specified set of detectors. 
        Removed detectors are not recoverable, except by re-loading 
        the original data file.

        Parameters
        ----------
        new_c1c2 : list
            A list of tuples containing the indices of the 
            detectors to keep, either in xf or muxcr 
            coordinates, as specified by `det_coord_mode`.
        det_coord_mode : {'xf','cr','idx'}, default='xf'
            Coordinate system used to identify detectors.
            'xf' (default), 'cr' or 'idx' are accepted.
            'idx' is the actual index in the detector list.

        Returns
        -------
        None        
        """

        inds = self._get_det_inds(new_c1c2,det_coord_mode)

        self.maps = self.maps[(inds)]
        self.header['xf_coords'] = self.header['xf_coords'][(inds)]
        self.header['cr_coords'] = self.header['cr_coords'][(inds)]
        self.header['n_detectors'] = len(inds)

        if self.header['flags']['beam_fits_initialized']:
            self.beam_fit_results['fits'] = self.beam_fit_results['fits'][(inds)]
            self.beam_fit_results['covs'] = self.beam_fit_results['covs'][(inds)]

        if self.header['flags']['has_gains']:
            self.header['gains'] = self.header['gains'][(inds)]
        if self.header['flags']['has_time_constants']:
            self.header['time_constants'] = self.header['time_constants'][(inds)]
        if self.header['flags']['has_beams']:
            self.header['beams'] = self.header['beams'][(inds)]


    ######################
    #### EDITING MAPS ####
    ######################
    def crop(self,xrange=(-np.inf,np.inf),yrange=(-np.inf,np.inf),xy_mode='coord'):

        if xy_mode == 'coord':
            x = np.copy(self.x_center)
            y = np.copy(self.y_center)
        elif xy_mode == 'offset':
            x = np.copy(self.dx_center)
            y = np.copy(self.dy_center)
        else:
            raise ValueError("xy_mode not recognized")

        self.maps = self.maps[:,((y>=yrange[0]) & (y<=yrange[1]))]
        self.maps = self.maps[:,:,((x>=xrange[0]) & (x<=xrange[1]))]
        self.counts = self.counts[:,((y>=yrange[0]) & (y<=yrange[1]))]
        self.counts = self.counts[:,:,((x>=xrange[0]) & (x<=xrange[1]))]
        self.x_center = self.x_center[((x>=xrange[0]) & (x<=xrange[1]))]
        self.y_center = self.y_center[((y>=yrange[0]) & (y<=yrange[1]))]
        self.dx_center = self.dx_center[((x>=xrange[0]) & (x<=xrange[1]))]
        self.dy_center = self.dy_center[((y>=yrange[0]) & (y<=yrange[1]))]
        self.x_edge = self.x_center - self.header['map_pars']['x_pixel']/2
        self.x_edge = np.concatenate((self.x_edge,[np.max(self.x_edge)+self.header['map_pars']['x_pixel']]))
        self.y_edge = self.y_center - self.header['map_pars']['y_pixel']/2
        self.y_edge = np.concatenate((self.y_edge,[np.max(self.y_edge)+self.header['map_pars']['y_pixel']]))
        self.dx_edge = (self.x_edge - self.header['map_pars']['x_center']) * self.dx_scale_factor
        self.dy_edge = self.y_edge - self.header['map_pars']['y_center']

        self.header['map_pars']['x_dim'] = np.ptp(self.x_edge)
        self.header['map_pars']['y_dim'] = np.ptp(self.y_edge)

    def smooth(self,pixels,mode='gauss'):

        x_pix = np.arange(self.maps.shape[2])
        x_pix = x_pix-np.median(x_pix)
        y_pix = np.arange(self.maps.shape[1])
        y_pix = y_pix-np.median(y_pix)
        if mode == 'gauss':
            x_kernel = np.exp(-4*np.log(2)*(x_pix/pixels)**2)
            y_kernel = np.exp(-4*np.log(2)*(y_pix/pixels)**2)
            kernel = x_kernel.reshape(1,1,-1) * y_kernel.reshape(1,-1,1)
        if mode == 'boxcar':
            kernel = np.zeros((1,len(y_pix),len(x_pix)))
            xx,yy = np.meshgrid(x_pix,y_pix)
            r2 = (xx**2+yy**2).reshape(1,len(y_pix),len(x_pix))
            kernel[r2<pixels**2/4] = 1
        kernel /= np.sum(kernel)

        self.maps = fftconvolve(self.maps,kernel,mode='same',axes=(1,2))

    def correct_gains(self,gains=None,gains_e=None,check_dets=True,drop_missing_dets=True,overwrite=False,reverse=False):
        """Apply counts to intensity conversion

        Convert from counts to units of intensity by applying
        the measured gains to the ``Map.data`` array.
        If the gains have already been specified using 
        ``Map.set_gains``, then no arguments are needed.
        If no gains have been set, then a ``timesoft.calibration.DetectorConstants``
        class containing the gains can be passed as an argument.
        This will then set the gains and apply them in one
        method call. 

        Parameters
        ----------
        gains : ``timesoft.calibration.DetectorConstants`` instance, optional
            An object containing gain measurements for a subset of the
            detectors. If gains for the timesream have already been 
            set then this argument is not necessary. If it is specified
            then it will be used to initialize the gains for the ``Map``
            instance and the remaining parameters are passed to 
            ``Map.set_gains``.
        reverse : bool, default=False
            If True, the gains correction will be reversed.
        """

        if gains is None and not self.header['flags']['has_gains']:
            raise ValueError("No Gain values specified")
        elif gains is not None:
            self.set_gains(gains=gains,check_dets=check_dets,drop_missing_dets=drop_missing_dets,overwrite=overwrite)
        
        if not reverse:
            if self.header['flags']['corrected_gains']:
                raise ValueError("Gain correction already applied")
            self.maps *= self.header['gains'].reshape(-1,1,1)
            self.header['flags']['corrected_gains'] = True
        if reverse:
            if not self.header['flags']['corrected_gains']:
                raise ValueError("No gain correction to remove")
            self.maps /= self.header['gains'].reshape(-1,1,1)
            self.header['flags']['corrected_gains'] = False



    ##########################
    #### VISUALIZING MAPS ####
    ##########################
    def plot(self,c1,c2,det_coord_mode='xf',xy_mode='coord',xy_unit='degrees',cbar=True,vminscale=1,vmaxscale=1,savepath=None,show=True,figsize=8.0,cmap='viridis',contour_levels=None,contour_colors='w'):
        """View map of single detector
        
        This method plots the map of a single detector.

        Parameters
        ----------
        c1, c2 : int
            The coordinates in either xf (default) or muxcr space
            of the detector to process. The coordinates system to 
            use is controlled by `det_coord_mode`.
        det_coord_mode : {'xf','cr'}, default='xf'
            Coordinate system used to identify detectors.
            'xf' (default) or 'cr' are accepted.
        xy_mode : {'offset','coord'}, default='coord'
            If 'coord' then the axes are the sky coordinates of 
            the data. If 'offset' then the axes are the offsets
            relative to the map center.
        xy_unit : {'degrees','arcminutes','arcseconds'}
            Unit to use for the axes
        cbar : bool, default=True
            If True, the displayed plot will include a colorbar
        vminscale, vmaxscale : float, default=1
            The colorbar minimum and maximum values will be rescaled
            by these amounts.
        show : bool, default=True
            If True, plt.show() will be called (set to False if you
            don't want the plot shown immediately)
        savepath : str, optional
            If specified, the resulting plot will be saved in the 
            path provided.
        figsize : float, optional
            The size of the larger axis of the figure (the smaller)
            one is determined by the aspect ratio of the image

        Returns
        -------
        None        
        """

        index = self._get_coord(c1,c2,det_coord_mode)

        x_shrinkage = self.dx_scale_factor
        edge_ratio = np.ptp(self.x_edge)*x_shrinkage / np.ptp(self.y_edge)
        if edge_ratio < 1:
            figsize = (figsize*edge_ratio,figsize)
        else:
            figsize = (figsize,figsize/edge_ratio)
            
        fig = plt.figure(figsize=figsize)

        if xy_mode == 'coord':
            x = self.x_edge
            y = self.y_edge
            xc = self.x_center
            yc = self.y_center
            aspect = 1/x_shrinkage
        elif xy_mode == 'offset':
            x = self.dx_edge
            y = self.dy_edge
            xc = self.dx_center
            yc = self.dy_center
            aspect = 'equal'
        else:
            raise ValueError("xy_coord not recognized")

        if xy_unit == 'degrees':
            ax_scale = 1
        elif xy_unit == 'arcminutes':
            ax_scale = 60
        elif xy_unit == 'arcseconds':
            ax_scale = 3600

        ax = fig.add_subplot(111)
        ax.set(aspect=aspect)
        if self.header['map_pars']['coords'] == 'az-el':
            ax.set(xlabel='Az (degrees)', ylabel='El (degrees)')
        elif self.header['map_pars']['coords'] == 'ra-dec':
            ax.set(xlabel='RA (degrees)', ylabel='Dec (degrees)')
            ax.invert_xaxis()
        elif self.header['map_pars']['coords'] == 'l-b':
            ax.set(xlabel='l (degrees)', ylabel='b (degrees)')
            ax.invert_xaxis()
        ax.set(title='Detector {}=({},{})'.format(det_coord_mode,c1,c2))

        vmin = vminscale*np.nanmin(self.maps[index])
        vmax = vmaxscale*np.nanmax(self.maps[index])
        c = ax.pcolormesh(x*ax_scale,y*ax_scale,self.maps[index],vmin=vmin,vmax=vmax,cmap=cmap)
        
        if contour_levels is not None:
            ax.contour(xc*ax_scale,yc*ax_scale,self.maps[index],levels=contour_levels,colors=contour_colors,linewidths=.75)

        if cbar:
            cb=fig.colorbar(c)
        
        if savepath is not None:
            plt.savefig(savepath)
        
        if show:
            plt.show()
        else:
            if cbar:
                return fig,ax,cb
            return fig,ax
        
    
    def detector_grid(self,savepath=None,show=True):
        """Creates a plot showing all detectors in a grid

        Plots a 16 x 60 grid showing the maps of every
        detector. Note that generating this many sub-plots
        is quite resource intensive, and displaying the 
        resulting image may crash the matplotlib viewer.
        The recommended use is to save the grid as PDF 
        file and inspect it with a PDF viewer. This 
        can be accomplished by setting the `show` parameter
        to False and the providing a location to save the
        file in the `savepath` parameter.

        Parameters
        ----------
        savepath : str, optional
            If specified, the resulting plot will be saved in the 
            path provided.
        show : bool, default=True            
            If True, plt.show() will be called end of the method 
            execution. If False plt.close() will be called at the
            end of the method execution. The recommended usage is
            to specify a `savepath` and then set `show` to False.
            Attempting to show the plot in the matplotlib viewer
            will likely crash and certainly be very slow.
        """

        fig = plt.figure(figsize=(4*ndet,4*nchan))
        plot = np.ones((ndet,nchan))

        for x in range(ndet):
            print('Plot Detectors: plotting feed '+str(x)+'...')
            for f in range(nchan):
                ax = fig.add_subplot(nchan,ndet,1+x+ndet*f)
                if (x,f) in self.header['xf_coords']:
                    ax.pcolormesh(self.x_edge,self.y_edge,self.maps[self.get_xf(x,f)])
                if f == 59:
                    ax.set(xlabel='Feedhorn '+str(x))
                if x == 0:
                    ax.set(ylabel='Channel '+str(f))
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        plt.close()


    ########################
    #### ANALYZING MAPS ####
    ########################
    def compute_rms(self,mask_x_bins=None, mask_y_bins=None, mask=None, xy_mode='coord'):

        if mask is None:
            rms = np.nanstd(self.maps,axis=(1,2))
        
        else:
            if xy_mode == 'coord':
                x,y = np.meshgrid(self.x_center,self.y_center)
            elif xy_mode == 'offset':
                x,y = np.meshgrid(self.dx_center,self.dy_center)
            else:
                raise ValueError('xy_mode not recognized')
            
            xbin = np.digitize(x,mask_x_bins).flatten()
            ybin = np.digitize(y,mask_y_bins).flatten()
            big_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2))
            big_mask[1:-1,1:-1] = mask

            masked = np.array(big_mask[tuple(ybin),tuple(xbin)]).reshape(1,*self.maps[0].shape) * np.ones((self.header['n_detectors'],1,1))
            masked = masked==0
            masked[np.isnan(self.maps)] = False

            rms = np.std(self.maps,axis=(1,2),where=masked)
        
        return rms

    def compute_sum(self,mask_x_bins=None, mask_y_bins=None, mask=None, xy_mode='coord'):

        if mask is None:
            sum = np.nansum(self.maps,axis=(1,2))
        
        else:
            if xy_mode == 'coord':
                x,y = np.meshgrid(self.x_center,self.y_center)
            elif xy_mode == 'offset':
                x,y = np.meshgrid(self.dx_center,self.dy_center)
            else:
                raise ValueError('xy_mode not recognized')
            
            xbin = np.digitize(x,mask_x_bins).flatten()
            ybin = np.digitize(y,mask_y_bins).flatten()
            big_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2))
            big_mask[1:-1,1:-1] = mask

            masked = np.array(big_mask[tuple(ybin),tuple(xbin)]).reshape(1,*self.maps[0].shape) * np.ones((self.header['n_detectors'],1,1))
            masked = masked==1
            masked[np.isnan(self.maps)] = False

            sum = np.sum(self.maps,axis=(1,2),where=masked)
        
        return sum


    #######################
    #### MODEL FITTING ####
    #######################
    def fit_model_det(self,c1,c2,fit_function,xy_mode='offset',use_feed_offsets=False,det_coord_mode='xf',p0=None,nofit_handling='raise',bounds=(-np.inf,np.inf), sigma=False, verbose = False):
        """Fit a model to the map of a specified detector

        This function will fit a function `fit_function` to
        the map of a specified detector using the curve_fit
        procedure from scipy.optimize.

        Parameters
        ----------
        c1, c2 : int
            The coordinates in either xf (default) or muxcr space
            of the detector map to fit. The coordinates system to 
            use is controlled by `det_coord_mode`.
        xy_mode : {'offset','coord'}, default='offset'
            Specifies whether the x and y values used for the fit
            should be given in terms of offsets from the map center
            ('offset', default) or absolute coordinates ('coord').
        use_feed_offsets : bool, default=False
            Specifies whether feedhorn offsets should be applied
            to the x and y values when doing the fits.
        det_coord_mode : {'xf','cr'}, default='xf'
            Coordinate system used to identify detectors.
            'xf' (default) or 'cr' are accepted.
        p0 : tupe or None, default=None
            An initial guess of the best fit parameters, passed 
            directly to curve_fit. See curve_fit documentation
            for details.
        bounds : list of tuples or None, default=None
            Bounds for the parameters being fit. Passed directly
            to curve_fit. See curve_fit documentation for 
            details.
        nofit_handling : {'raise', 'allow'}, default='raise'
            If 'raise' an error will be raised when the fit fails. 
            If 'allow', the returned values will be arrays of NaNs.
        sigma : bool, default = False
            if True uncertainty from variance weighted maps is used in the fits

        Returns
        -------
        fit : array
            The best fit parameters
        cov : array
            The covariance array
        """
        # nofit_handling: raise - raises error
        # allow - returns array of nans

        if nofit_handling not in ['raise','allow']:
            raise ValueError('nofit_handling not recognized')

        det_ind = self._get_coord(c1,c2,det_coord_mode)
        x_coord = self.header['xf_coords'][det_ind][0]
        
        if use_feed_offsets:
            feed_offset_x, feed_offset_y = self.get_feed_offsets(c1,c2,det_coord_mode)
        else:
            feed_offset_x = 0
            feed_offset_y = 0
        if xy_mode == 'coord':
            x = self.x_center - feed_offset_x / self.dx_scale_factor
            y = self.y_center - feed_offset_y
        elif xy_mode == 'offset':
            x = self.dx_center - feed_offset_x
            y = self.dy_center - feed_offset_y
        else:
            raise ValueError("xy_coord not recognized")

        xx,yy = np.meshgrid(x,y)

        # xx = xx.flatten()
        # yy = yy.flatten()
        if p0[0] < 0:
            p0[0] *= -1
        target_map = self.maps[det_ind]
        fit_inds = np.isfinite(self.maps[det_ind])
        if sigma:
            try:
                fit,cov = curve_fit(fit_function,(xx,yy,fit_inds),target_map[fit_inds],sigma=self.e_maps[det_ind][fit_inds],p0=p0,bounds=bounds, absolute_sigma=True)
            except Exception as err:
                if nofit_handling=='raise':
                    raise err
                else:
                    if verbose == True:
                    	print("No fit found for {}=({},{})".format(det_coord_mode,c1,c2))
                    npar = len(p0)
                    fit = np.zeros(npar)
                    fit[:] = np.nan
                    cov = np.zeros((npar,npar))
                    cov[:] = np.nan
            return fit,cov
        else:
            try:
                fit,cov = curve_fit(fit_function,(xx,yy,fit_inds),target_map[fit_inds],p0=p0,bounds=bounds, absolute_sigma=True)
            except Exception as err:
                if nofit_handling=='raise':
                    raise err
                else:
                    if verbose == True:
                    	print("No fit found for {}=({},{})".format(det_coord_mode,c1,c2))
                    npar = len(p0)
                    fit = np.zeros(npar)
                    fit[:] = np.nan
                    cov = np.zeros((npar,npar))
                    cov[:] = np.nan            
            return fit,cov

    def fit_model(self,fit_function,xy_mode='offset',use_feed_offsets=False,p0=None,nofit_handling='raise',bounds=(-np.inf,np.inf),sigma=False, verbose = False):
        """Fit a model to to each map

        This function will fit a function `fit_function` to
        the map for all detectors using the curve_fit procedure 
        from scipy.optimize.

        Parameters
        ----------
        xy_mode : {'offset','coord'}, default='offset'
            Specifies whether the x and y values used for the fit
            should be given in terms of offsets from the map center
            ('offset', default) or absolute coordinates ('coord').
        use_feed_offsets : bool, default=False
            Specifies whether feedhorn offsets should be applied
            to the x and y values when doing the fits.
        det_coord_mode : {'xf','cr'}, default='xf'
            Coordinate system used to identify detectors.
            'xf' (default) or 'cr' are accepted.
        p0 : array or None, default=None
            An initial guess of the best fit parameters
            If p0 is the same for all detectors, use a 1d array
            matching the number of parameters. If p0 is different
            for each detector use an array of shape
            (n_detectors, n_parameters)
        bounds : list of tuples or None, default=None
            Bounds for the parameters being fit. Passed directly
            to curve_fit. See curve_fit documentation for 
            details.
        nofit_handling : {'raise', 'allow'}, default='raise'
            If 'raise' an error will be raised when the fit fails. 
            If 'allow', the returned values will be arrays of NaNs.
        sigma: boolean
            If true use error maps to weight the fit

        Returns
        -------
        fit : array
            Array of shape (n_detectors, n_parameters) with
            best fit parameters.
        cov : array
            Array of shape (n_detectors, n_parameters, n_parameters) 
            with covariance arrays.
        """

        if p0 is not None:
            if p0.ndim == 2 and len(p0) != self.header['n_detectors']:
                raise ValueError("Number of provided p0 values ({}) does not match number of detectors ({})".format(len(p0),self.header['n_detectors']))

        fits = []
        covs = []
        for i in range(self.header['n_detectors']):

            if p0 is None:
                p0_i = None
            elif p0.ndim == 1:
                p0_i = p0
            else:
                p0_i = p0[i]

            xf = self.header['xf_coords'][i]

            fit_i,cov_i = self.fit_model_det(xf[0],xf[1],fit_function,det_coord_mode='xf',xy_mode=xy_mode,use_feed_offsets=use_feed_offsets,p0=p0_i,nofit_handling=nofit_handling,bounds=bounds,sigma=sigma, verbose = verbose)
            fits.append(fit_i)
            covs.append(cov_i)

        return np.array(fits),np.array(covs)

    def _rotated_ellipse_gaussian(self, x, y, x0, y0, fwhm_x, fwhm_y, theta):
        # rotated elliptical Gaussian
        X = x - x0
        Y = y - y0
        theta = np.deg2rad(theta)
        c, s = np.cos(theta), np.sin(theta)
        sigma_x = fwhm_x / (2*np.sqrt(2*np.log(2)))
        sigma_y = fwhm_y / (2*np.sqrt(2*np.log(2)))
        xr = c*X + s*Y
        yr = -s*X + c*Y
        return  np.exp(-0.5 * ((xr/sigma_x)**2 + (yr/sigma_y)**2)) 

    def _gaussian(self, x, y, x0, y0, fwhm):
        X = x - x0
        Y = y - y0
        sigma = fwhm / (2*np.sqrt(2*np.log(2)))
        return  np.exp(-0.5 * (((x-x0)/sigma)**2 + ((y-y0)/sigma)**2)) 
        
    def _one_sided_exponential_kernel(self, xgrid, alpha):
        expo = np.exp( -(xgrid)/alpha) #/ alpha
        # expo /= expo.sum()
        expo = np.roll(expo, expo.size//2)
        return expo

    def _double_sided_exponential_kernel(self, xgrid, alpha):
        expo = np.exp( -(xgrid)/alpha) #/ alpha
        # expo /= expo.sum()
        expo = np.roll(expo, expo.size//2)

        if xgrid.shape[0] % 2 == 1:
            expo[:expo.size//2] = expo[expo.size//2:][::-1][:-1] ###### just drop element to fit into array... should be decayed close to zero anyway.
        else:
            expo[:expo.size//2] = expo[expo.size//2:][::-1]



        return expo

    def _make_2d_kernel_from_1d(self, xgrid, kernel):
        ny, nx = xgrid.shape
        arr = np.zeros((ny, nx), dtype=float)

        L = len(kernel)
        cx, cy = nx // 2, ny // 2

        # endpoints of the kernel line in centered coordinates
        r = L // 2
        x0, y0 = -r, 0
        x1, y1 =  r, 0

        # rotation
        theta = 0
        c, s = np.cos(theta), np.sin(theta)
        x0r, y0r = c * x0 - s * y0, s * x0 + c * y0
        x1r, y1r = c * x1 - s * y1, s * x1 + c * y1
    

        x0i, y0i = int(round(x0r + cx)), int(round(y0r + cy))
        x1i, y1i = int(round(x1r + cx)), int(round(y1r + cy))

        # Bresenham line pixels
        rr, cc = line(y0i, x0i, y1i, x1i)

        # clip to array
        mask = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
        rr, cc = rr[mask], cc[mask]

        # resample kernel along this line
        k = np.linspace(0, L-1, len(rr)).astype(int)
        arr[rr, cc] = kernel[k]

        return arr

    def beam_fit(self,use_feed_offsets=False,p0=None,bounds=None,nofit_handling='raise',normalize_planet=False,force_pointsource=False,croppercentile=100,apply_angle_offset=True):
        """Fit a Gaussian beam to to each map, deconvolving
        the source if necessary.

        This function fit a 2D gaussian with parameters amplitude, 
        x_center, y_center, x_width, y_width. It is only designed to
        work on planets. It will attempt to deconvolve the planet size
        from the beam fit when the planet is large relative to the 
        beam.

        It returns the fit function, fits and covariances, but 
        it also saves these in the ``Map.beam_fit_results`` dictionary.

        This is a wrapper around the ``fit_model`` method.

        Parameters
        ----------
        use_feed_offsets : bool, default=False
            Specifies whether feedhorn offsets should be applied
            to the x and y values when doing the fits.
        p0 : None or array, default=None
            An initial guess of the best fit parameters. If
            None is given, an initial guess is generated for
            each map using the brightest pixel to for the center
            and amplitude, and a reasonable value of FWHM for 
            a 12M telescope at 230 GHz.
            If p0 is the same for all detectors, use a 1d array
            matching the number of parameters. If p0 is different
            for each detector use an array of shape
            (n_detectors, n_parameters)
        bounds: None or array, default=None
            bounds for the parameters
        normalize_planet : bool, default=False
            If True, sum of the planet model will be equal to unity.
            Useful for using the amplitude parameter to derive gains.
        force_pointsource : bool, default=False
            Force the planet size to be zero rather than using an actual
            model of the planet disk
        apply_angle_offset : bool, default=True
            If true, rotate the coordinates so that x is along the scan direction
            and y is perpendicular. Otherwise x is ra and y is dec

        Returns
        -------
        fit_function : function
            The function used in the fits, including any convolutions
            with the planet model.
        fit : array
            Array of shape (n_detectors, n_parameters) with
            best fit parameters.
        cov : array
            Array of shape (n_detectors, n_parameters, n_parameters) 
            with covariance arrays.
        """

        # Get planet coordinates and size for time of observation
        # If force_pointsource just use zero for the size
        if force_pointsource:
            planet_radius = 0
        else:
            if self.header['object'].lower() not in ['venus','mars','jupiter','saturn']:
                raise ValueError("Target does not appear to be a planet")

            time = Time(self.header['med_time'],format='unix')

            # _, arcsec, arcsec; give freq=230 but this is not used for anything (would be if we cared about brightness temp)
            _, planet_size_major, planet_size_minor = get_planet(self.header['object'].lower(), time, 230)
            
            # In degrees
            planet_radius = (planet_size_major+planet_size_minor)/2/2 / 3600 # Divide by 4 - average + diam to radius
        
        if apply_angle_offset:
            theta = -1*self.header['map_pars']['map_angle_offset'] * np.pi/180
        else:
            theta = 0

        # If the planet is large, account for its effect on the beam
        if planet_radius > np.sqrt(self.header['map_pars']['dx_pixel']**2 + self.header['map_pars']['y_pixel']**2):
            rad = np.sqrt((self.dx_center.reshape(1,-1))**2+self.dy_center.reshape(-1,1)**2)
            planet_model = np.zeros(self.maps.shape[1:])
            planet_model[rad<=planet_radius] = 1
            print(np.sum(planet_model), planet_radius)
            if normalize_planet:
                planet_model /= np.sum(planet_model)

            def fit_function(X,amp,cx,cy,wx,wy):
                (x,y,inds) = X
                xp = x*np.cos(theta) - y*np.sin(theta)
                yp = x*np.sin(theta) + y*np.cos(theta)

                sx = wx / (2*np.sqrt(2*np.log(2)))
                sy = wy / (2*np.sqrt(2*np.log(2)))
                beam = amp * np.exp(-0.5 * ((xp-cx)**2/sx**2 + (yp-cy)**2/sy**2))
                model = convolve(beam, planet_model, mode='same')
                return model.flatten()
        
        else:
            def fit_function(X,amp,cx,cy,wx,wy):
                (x,y,inds) = X
                xp = x*np.cos(theta) - y*np.sin(theta)
                yp = x*np.sin(theta) + y*np.cos(theta)

                sx = wx / (2*np.sqrt(2*np.log(2)))
                sy = wy / (2*np.sqrt(2*np.log(2)))
                beam = amp * np.exp(-0.5 * ((xp-cx)**2/sx**2 + (yp-cy)**2/sy**2))
                return beam.flatten()

        if p0 is None:
            p0 = []
            fwhm_guess = 1.2 * (1.2/1000) / 12 * 180/np.pi # approx beam size, degrees
            for i in range(self.header['n_detectors']):
                
                copy_map = np.copy(self.maps[i])
                copy_map[copy_map > np.nanpercentile(copy_map,croppercentile)] = 0
                peak_index = np.unravel_index(np.nanargmax(copy_map),self.maps[i].shape)
                x0 = self.dx_center[peak_index[1]]
                y0 = self.dy_center[peak_index[0]]
                if use_feed_offsets:
                    feed_offset_x, feed_offset_y = self.get_feed_offsets(self.header['xf_coords'][i][0],self.header['xf_coords'][i][1],det_coord_mode='xf')
                    x0 = x0-feed_offset_x
                    y0 = y0-feed_offset_y
                
                if apply_angle_offset:
                    x0p = x0*np.cos(theta) - y0*np.sin(theta)
                    y0p = x0*np.sin(theta) + y0*np.cos(theta)
                    x0, y0 = x0p, y0p

                a0 = np.nanmax(copy_map)

                p0.append([a0,x0,y0,fwhm_guess,fwhm_guess])
                
            p0 = np.array(p0)

        if bounds is None:
            bounds = ([0,-np.inf,-np.inf,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf])

        fits,covs = self.fit_model(fit_function,xy_mode='offset',use_feed_offsets=use_feed_offsets,p0=p0,nofit_handling=nofit_handling,bounds=bounds)

        self.beam_fit_results = {'fits':fits, 'covs':covs, 'function':fit_function}
        self.header['flags']['beam_fits_initialized'] = True

        return fit_function,fits,covs

    def repurposed_beam_fit(self,use_feed_offsets=False,p0='guess',nofit_handling='raise', xy_mode='offset', verbose = False, beamtype='Gauss', sigma=False):
        """Fit a Gaussian beam to to each map, deconvolving
        the source if necessary.

        This function fit a 2D gaussian with parameters amplitude, 
        x_center, y_center, x_width, y_width. It is only designed to
        work on planets. It will attempt to deconvolve the planet size
        from the beam fit when the planet is large relative to the 
        beam.

        It returns the fit function, fits and covariances, but 
        it also saves these in the ``Map.beam_fit_results`` dictionary.

        This is a wrapper around the ``fit_model`` method.

        Parameters
        ----------
        use_feed_offsets : bool, default=False
            Specifies whether feedhorn offsets should be applied
            to the x and y values when doing the fits.
        p0 : 'guess' or array or None, default='guess'
            An initial guess of the best fit parameters. If
            'guess' is given, an initial guess is generated for
            each map using the brightest pixel to for the center
            and amplitude, and a reasonable value of FWHM for 
            a 12M telescope at 230 GHz.
            If p0 is the same for all detectors, use a 1d array
            matching the number of parameters. If p0 is different
            for each detector use an array of shape
            (n_detectors, n_parameters)
        xy_mode : str, 'coords' or 'offset', default = 'offset'
            The coordinate system to preform the fits in. Coords does
            the fit along the RA/DEC grid (useful for finding absolute
            position), whereas offset will do the fits along a delta 
            RA/DEC grid (useful for finding beam offsets)

        Returns
        -------
        fit_function : function
            The function used in the fits, including any convolutions
            with the planet model.
        fit : array
            Array of shape (n_detectors, n_parameters) with
            best fit parameters.
        cov : array
            Array of shape (n_detectors, n_parameters, n_parameters) 
            with covariance arrays.
        """

        if beamtype == 'Gauss':
            def fit_func(x,y, *params):
                A, x0, y0, fwhm, bg = params 
                beam = self._gaussian(x,y,x0,y0,fwhm)
                beam = beam * A
                # beam = norm_func(beam) * A + bg
                return beam, bg
        elif beamtype == 'rotate_gauss':
            def fit_func(x,y, *params):
                A, x0, y0, fwhm_x, fwhm_y, theta, bg = params
                beam = self._rotated_ellipse_gaussian(x, y, x0, y0, fwhm_x, fwhm_y, theta)
                beam = beam * A
                return beam, bg
        elif beamtype == 'rotate_gauss+2tc':
            def fit_func(x,y, *params):
                x_cent = x.shape[0] // 2 
                one_d = x[x_cent,:]
                A, x0, y0, fwhm_x, fwhm_y, theta, bg, alpha = params
                if fwhm_x <= 0 or fwhm_y <= 0 or alpha < 0:
                    return np.full_like(x, 1e12)
                k1d = self._double_sided_exponential_kernel(one_d, alpha)
                k2d = self._make_2d_kernel_from_1d(x, k1d)
                g2d = self._rotated_ellipse_gaussian(x, y, x0, y0, fwhm_x, fwhm_y, theta)
                # beam = beam * A + bg
                beam = fftconvolve(g2d, k2d, mode='same')
                beam = A * beam + bg
                return beam
        elif beamtype == 'Gauss+2tc':
            def fit_func(x,y, *params):
                x_cent = x.shape[0] // 2 
                one_d = x[x_cent,:]
                A, x0, y0, fwhm, bg, alpha = params
                if fwhm <= 0 or fwhm <= 0 or alpha < 0: 
                    return np.full_like(x, 1e12)
                k1d = self._double_sided_exponential_kernel(one_d, alpha)
                k2d = self._make_2d_kernel_from_1d(x, k1d)
                g2d = self._gaussian(x, y, x0, y0, fwhm)
                # beam = beam * A + bg
                beam = fftconvolve(g2d, k2d, mode='same')
                beam = A *beam + bg
                return beam
        elif beamtype == 'Gauss+tc':
            def fit_func(x,y, *params):
                x_cent = x.shape[0] // 2 
                one_d = x[x_cent,:]
                A, x0, y0, fwhm, bg, alpha = params
                if fwhm <= 0 or fwhm <= 0 or alpha < 0: 
                    return np.full_like(x, 1e12)
                k1d = self._one_sided_exponential_kernel(one_d, alpha)
                k2d = self._make_2d_kernel_from_1d(x, k1d)
                g2d = self._gaussian(x, y, x0, y0, fwhm)
                beam = fftconvolve(g2d, k2d, mode='same')
                beam = A* beam + bg
                return beam 

        elif beamtype == 'rotate_Gauss+tc':
            def fit_func(x,y, *params):
                x_cent = x.shape[0] // 2 
                one_d = x[x_cent,:]
                A, x0, y0, fwhm_x, fwhm_y, theta, bg, alpha = params
                if fwhm_x <= 0 or fwhm_y <= 0 or alpha < 0: 
                    return np.full_like(x, 1e12)
                k1d = self._double_sided_exponential_kernel(one_d, alpha)
                k2d = self._make_2d_kernel_from_1d(x, k1d)
                g2d = self._rotated_ellipse_gaussian(x, y, x0, y0, fwhm_x, fwhm_y, theta)
                beam = fftconvolve(g2d, k2d, mode='same')
                beam = A* beam + bg
                return beam


        # Get planet coordinates and size for time of observation
        available_planets = ['venus','mars','jupiter','saturn', 'Jupiter', 'Saturn', 'Mars', 'Venus','3C279','3C454.3','3C84','3C273']



        if self.header['object'].lower() in available_planets:
            time = Time(self.header['med_time'],format='unix')

            # In K, arcsec, arcsec:
            planet_tbb, planet_size_major, planet_size_minor = get_planet(self.header['object'].lower(), time, 230)
            
            # In degrees
            planet_radius = (planet_size_major+planet_size_minor)/2/2 / 3600 # Divide by 4 - average + diam to radius

            # If the planet is large, account for its effect on the beam

            print(planet_radius, 'planet radius')
            ### dx_pixel and y_pixel which one should it be 

            ### w/o this convolution 
            ### w/o the background 
            if planet_radius > np.sqrt(self.header['map_pars']['dx_pixel']**2 + self.header['map_pars']['y_pixel']**2):
                rad = np.sqrt((self.dx_center.reshape(1,-1))**2+self.dy_center.reshape(-1,1)**2)
                planet_model = np.zeros(self.maps.shape[1:])
                planet_model[rad<planet_radius] = 1

                def fit_function(X,*params): #amp,cx,cy,wx,wy):
                    (x,y,mask_inds) = X
                    mask_inds = np.where(mask_inds.flatten() > 0.1)[0]
                    beam, bg = fit_func(x,y,*params)
                    model = convolve(beam, planet_model, mode='same').flatten()
                    model = beam.flatten()[mask_inds] + bg
                    return model
            else: 
                def fit_function(X,*params): #amp,cx,cy,wx,wy):
                    (x,y,mask_inds) = X
                    x_cent = x.shape[0] // 2 
                    y_cent = x.shape[1] // 2
                    mask_inds = np.where(mask_inds.flatten() > 0.1)[0]
                    beam, bg = fit_func(x,y,*params)
                    return beam.flatten()[mask_inds] + bg
        
        else:
            def fit_function(X, *params):
                (x,y,mask_inds) = X
                x_cent = x.shape[0] // 2 
                y_cent = x.shape[1] // 2
                mask_inds = np.where(mask_inds.flatten() > 0.1)[0]
                beam, bg = fit_func(x,y,*params)
                return beam.flatten()[mask_inds] + bg

        if p0 == 'guess':
            p0 = []
            fwhm_guess = 1.2 * (1.2/1000) / 12 * 180/np.pi # approx beam size, degrees
            for i in range(self.header['n_detectors']):
                plt.clf()
                #plt.imshow(self.maps[i])
                #plt.savefig('/home/vaughan/TIME-analysis/testing_nan_map')
                try:
                    peak_index = np.unravel_index(np.nanargmax(self.maps[i]),self.maps[i].shape)
                    x0 = self.dx_center[peak_index[1]]
                    y0 = self.dy_center[peak_index[0]]
                except ValueError:
                    x0 = self.maps[i].shape[1] // 2
                    y0 = self.maps[i].shape[0] // 2 
                    warnings.warn('Peak Index not found estimating as center of map')
            
                    # continue ### 

                if use_feed_offsets:
                    feed_offset_x, feed_offset_y = self.get_feed_offsets(self.header['xf_coords'][i][0],self.header['xf_coords'][i][1],det_coord_mode='xf')
                    x0 = x0-feed_offset_x
                    y0 = y0-feed_offset_y
                
                a0 = np.nanmax(self.maps[i])
                bg_guess = 0 
                theta_guess = 0 
                theta2_guess = 20
                tc_guess = 0.3e-3
                if beamtype == 'Gauss':
                    p0.append([a0,x0,y0,fwhm_guess,bg_guess])
                    bounds = ([0,-np.inf,-np.inf,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf])
                elif beamtype == 'rotate_gauss':
                    p0.append([a0, x0, y0, fwhm_guess, fwhm_guess, bg_guess, theta_guess])
                    bounds = ([0,-np.inf,-np.inf,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf, np.inf,360])
                elif beamtype == 'rotate_gauss+2tc':
                                # A, x0, y0, fwhm_x, fwhm_y, theta, bg, alpha, theta2 = params

                    p0.append([a0, x0, y0, fwhm_guess, fwhm_guess, theta_guess, bg_guess, tc_guess])
                    bounds = (
                        [0,       -np.inf, -np.inf,   0,      0,      0,  -np.inf,     0],     # lower bounds
                        [1e5,   np.inf,  np.inf,   np.inf, np.inf, 360, np.inf,     1]  # upper bounds
                    )
                elif beamtype == 'Gauss+2tc':
                    p0.append([a0, x0, y0, fwhm_guess, bg_guess, tc_guess])
                    bounds = (
                        [0, -np.inf, -np.inf, 0, -np.inf, 0],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, 1]
                    )
                elif beamtype == 'Gauss+tc':
                    p0.append([a0, x0, y0, fwhm_guess, bg_guess, tc_guess])
                    bounds = (
                        [0, -np.inf, -np.inf, 0, -np.inf, 0],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, 1]
                    )
                elif beamtype == 'rotate_Gauss+tc':
                    p0.append([a0, x0, y0, fwhm_guess, fwhm_guess, theta_guess, bg_guess, tc_guess])
                    bounds = (
                        [0, -np.inf, -np.inf, 0, 0, 0, -np.inf, 0],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, 360, np.inf, 1]
                    )


                
            p0 = np.array(p0)


        fits,covs = self.fit_model(fit_function,xy_mode=xy_mode,use_feed_offsets=use_feed_offsets,p0=p0,nofit_handling=nofit_handling,bounds=bounds, verbose = verbose, sigma=sigma)

        self.beam_fit_results = {'fits':fits, 'covs':covs, 'function':fit_function}
        self.header['flags']['beam_fits_initialized'] = True

        return fit_function,fits,covs

    def beam_offsets(self,x0,set_header_offsets=False, x0_med_x=None, x0_med_y=None, x_fits=None, past_fits=False):
        """Compute offsets between all feedhorns and a 
        reference feedhorn
        """

        if not self.header['flags']['beam_fits_initialized'] and not past_fits:
            raise ValueError("No fits to the beam available")

        if x0 not in np.arange(16):
            raise ValueError("x0 bust be between 0 and 15")

        if not past_fits:
            x0_fits = self.beam_fit_results['fits'][(self.get_x(x0))]

        # x0_med_x = np.nanmedian(x0_fits[:,1])
        # x0_med_y = np.nanmedian(x0_fits[:,2])

        ra_offsets = np.zeros(16)
        dec_offsets = np.zeros(16)
        ra_offsets[:] = np.nan
        dec_offsets[:] = np.nan

        for x in range(16):
            try:
                if not past_fits:
                    x_fits = self.beam_fit_results['fits'][(self.get_x(x))]
            except ValueError:
                #print("No detectors found for x={}".format(x))
                continue

            ra_offsets[x] = np.nanmedian(x_fits[:,1])-x0_med_x
            dec_offsets[x] = np.nanmedian(x_fits[:,2])-x0_med_y
        
        offsets = Offsets(ra_offsets,dec_offsets,frame=self.header['epoch'],spectrometer=self.header['mc'])
        if set_header_offsets:
            self.set_feed_offsets(offsets)

        return offsets


    #########################
    #### COMBINING MAPS  ####
    #########################
    def sum_maps(self,mode='all',weights=None,rescales=None,nan_handle='zero'):
        if mode not in ['all']:
            raise ValueError("Mode not recognized")
        if nan_handle not in ['zero','nan']:
            raise ValueError("nan_handle not recognized")
        if nan_handle == 'zero':
            sum_func = np.nansum
        if nan_handle == 'nan':
            sum_func = np.sum

        if weights is None:
            weights = np.ones(self.header['n_detectors'])
        elif len(weights) != self.header['n_detectors']:
            raise ValueError("Length of weights array must match number of detectors")
        counts = np.array([self.counts[self.header['xf_coords'][i][0]] for i in range(self.header['n_detectors'])])
        counts[counts>0] = 1
        weights = counts * np.array(weights).reshape(-1,1,1)

        if rescales is None:
            rescales = np.ones(self.header['n_detectors'])
        elif len(rescales) != self.header['n_detectors']:
            raise ValueError("Length of rescales array must match number of detectors")
        rescales = np.array(rescales).reshape(-1,1,1)

        if mode=='all':
            sum_map = sum_func(weights * rescales * self.maps, axis=0) / np.sum(weights, axis=0)

        sum_map = MapConstructor(self.header, self.header['map_pars'], self.x_edge, self.y_edge, sum_map.reshape(1,*sum_map.shape), np.ones(self.counts.shape))
        sum_map.header['xf_coords'] = [(0,0)]
        sum_map.header['cr_coords'] = [(0,0)]

        sum_map.header['n_detectors'] = 1
        sum_map.header['flags']['beam_fits_initialized'] = False
        if sum_map.header['flags']['has_gains']:
            sum_map.header.pop('gains')
            sum_map.header['flags']['has_gains'] = False
        if sum_map.header['flags']['has_time_constants']:
            sum_map.header.pop('time_constants')
            sum_map.header['flags']['has_time_constants'] = False

        return sum_map
    
    def gains_to_focal_grid(self):
        gains = self.header['gains']
        gains_e = self.header['gains_e']
        focal_grid_gains = np.zeros((16,60))
        focal_grid_gains_e = np.zeros((16,60))
        focal_grid_gains[:,:] = np.nan; focal_grid_gains_e[:,:] = np.nan
        xf_coords = self.header['xf_coords']
        x_coords = np.array([xf[0] for xf in xf_coords])
        f_coords = np.array([xf[1] for xf in xf_coords])
        focal_grid_gains[x_coords,f_coords] = gains 
        focal_grid_gains_e[x_coords,f_coords] = gains_e
        self.header['focal_grid_gains'] = focal_grid_gains
        self.header['focal_grid_gains_e'] = focal_grid_gains_e
        
    def maps_to_focal_grid(self):
        ndetector, x_size, y_size = self.maps.shape
        focal_grid = np.zeros((16,60,x_size,y_size))
        focal_grid[:,:,:,:] = np.nan
        xf_coords = self.header['xf_coords']
        x_coords = np.array([xf[0] for xf in xf_coords])
        f_coords = np.array([xf[1] for xf in xf_coords])
        focal_grid[x_coords,f_coords,:,:] = self.maps
        self.focal_grid_maps = focal_grid
        self.header['focal_grid_maps'] = focal_grid

    def emaps_to_focal_grid(self):
        ndetector, x_size, y_size = self.e_maps.shape
        focal_grid = np.zeros((16,60,x_size,y_size))
        focal_grid[:,:,:,:] = np.nan
        xf_coords = self.header['xf_coords']
        x_coords = np.array([xf[0] for xf in xf_coords])
        f_coords = np.array([xf[1] for xf in xf_coords])
        focal_grid[x_coords,f_coords,:,:] = self.e_maps
        self.focal_grid_e_maps = focal_grid
        self.header['focal_grid_maps'] = focal_grid

    def make_f_co_add(self, sanity_check=True):
        ndet,nf,x_size,y_size = self.focal_grid_maps.shape
        weights = 1 / self.header['focal_grid_gains']
        ca = (np.nansum((self.focal_grid_maps.reshape(ndet*nf,x_size,y_size).T * weights.flatten()).T.reshape((ndet,nf,x_size,y_size)),axis=0).T / np.nansum(weights,axis=0)).T
        self.co_add = ca 
        self.header['co_add'] = ca


    def plot_frequency(self, frequency_data, frequencies,title,colorbar=True):
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(title, fontsize=16)

        nrows, ncols = 6, 10
        # Calculate symmetric color limit
        # valid_values = frequency_data[~np.isnan(frequency_data)]
        valid_values = self.data[~np.isnan(self.data)] ### check that self.data is a thing.
        std = np.nanstd(valid_values)
        if np.isnan(std) or std == 0:
            print("Warning: std is NaN or zero — setting default color scale")
            color_max = 34.712654844322714 
        else:
            color_max = 12 * std
            print(f"Using ±12σ for color scale: ±{color_max}")
            
        cmap = 'RdBu'

        im = None  # to keep reference for the colorbar

        for i in range(self.maps.shape[0]): ### I think this is correct...
            row, col = divmod(i, ncols)
            index = i + 1
            this_data = frequency_data[i]

            ax = fig.add_subplot(nrows, ncols, index)
            im = ax.imshow(this_data, origin='lower', cmap=cmap)

            ax.set_title(f'{frequencies[i]}', fontsize=10)
            

            ax.coords[0].set_ticks_visible(False)
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[1].set_ticks_visible(False)
            ax.coords[1].set_ticklabel_visible(False)
            ax.set_xlim(time_wcs.wcs.crpix[0] - 10, time_wcs.wcs.crpix[0] + 10)
            ax.set_ylim(time_wcs.wcs.crpix[1] - 10, time_wcs.wcs.crpix[1] + 10)
            ax.invert_xaxis()

        cbar_ax = fig.add_axes([0.2, 0.07, 0.6, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        # cbar.set_ticks(np.linspace(-, 5))

        plt.subplots_adjust(wspace=0.1, hspace=0.3, bottom=0.15)
        plt.savefig(title.replace(" ", "_")+'.png', dpi=300) 
        plt.show() 


class MapConstructor(Map):

    def __init__(self,header,map_pars,x,y,maps,counts,e_maps=np.zeros((1)),h_maps=np.zeros((1)),extras={}):

        self._check_header(header)

        self.header = copy.deepcopy(header)
        self.header['map_pars'] = copy.deepcopy(map_pars)
        self.header['flags']['maps_initialized'] = True

        self.x_edge = np.copy(x)
        self.y_edge = np.copy(y)
        
        self.maps = np.copy(maps)
        self.e_maps = np.copy(e_maps)
        self.h_maps = np.copy(h_maps)
        self.counts = np.copy(counts)
        self.extras = copy.deepcopy(extras)

        self._input_consistency_check()
        self._construct_axes()
