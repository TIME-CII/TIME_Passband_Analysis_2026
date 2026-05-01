from timesoft.helpers import coordinates
from timesoft.helpers._class_bases import _Datastruct_base
from timesoft.calibration import Offsets

import os
import warnings
import copy
from inspect import signature

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# SOME DATASET VARIABLES:
nchan = 60
ndet = 16

class LineMap(_Datastruct_base):

    def __init__(self,path,xf=None,cr=None):

        file_in = np.load(path, allow_pickle=True)

        if 'is_linemap' not in file_in.files:
            raise ValueError("The specified file does not appear to be a TIME linemap")
        self._check_header(file_in['header'][()])

        self.header = file_in['header'][()] # This weird syntax turns back into a dictionary

        self.ax_edge = np.array(file_in['ax_edge'])
        self.x_center = np.array([(self.ax_edge[i]+self.ax_edge[i+1])/2 for i in range(len(self.ax_edge)-1)])

        self.linemaps = np.array(file_in['linemaps'])
        self.counts = np.array(file_in['counts'])
        self.extras = np.array(file_in['extras'])[()]

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

    def _input_consistency_check(self):

        if not self.header['flags']['1d_initialized']:
            raise ValueError("Header flags indicate no linemaps initialized")

        # Check everything loaded looks good
        if self.linemaps.shape[1] != len(self.x_center):
            raise ValueError("Linemap dimensions do not match the axis")

        if self.counts.shape[1:] != self.linemaps.shape[1:]:
            raise ValueError("Linemap dimensions do not match counts dimensions")
        extras_check = [self.extras[key].shape[1:] != self.linemaps.shape[1:] for key in self.extras.keys()]
        if np.any(extras_check):
            raise ValueError("Linemap dimesions do not match extras dimensions")
        extras_check_2 = [self.extras[key].shape[0] != 16 for key in self.extras.keys()]
        if np.any(extras_check):
            raise ValueError("Extras does not match number of feeds (16)")

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

        self.linemaps = self.linemaps[(inds)]
        self.header['xf_coords'] = self.header['xf_coords'][(inds)]
        self.header['cr_coords'] = self.header['cr_coords'][(inds)]
        self.header['n_detectors'] = len(inds)

        if self.header['flags']['has_gains']:
            self.header['gains'] = self.header['gains'][(inds)]
        if self.header['flags']['has_time_constants']:
            self.header['time_constants'] = self.header['time_constants'][(inds)]
        if self.header['flags']['has_beams']:
            self.header['beams'] = self.header['beams'][(inds)]

    def beam_fit(self,use_feed_offsets=False,p0='guess',nofit_handling='raise', xy_mode='offset', verbose = False):
        """Fit a Gaussian beam to to each map, deconvolving
        the source if necessary.

        This function fit a 1D gaussian with parameters amplitude, 
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



        # Get planet coordinates and size for time of observation

        def fit_function(X,amp,c,w):
            (x,mask_inds) = X
            mask_inds = np.where(mask_inds.flatten() > 0.1)[0]
            s = w / (2*np.sqrt(2*np.log(2)))
            beam = amp * np.exp(-0.5 * ((x-c)**2/s**2 ))
            return beam.flatten()[mask_inds]

        if p0 == 'guess':
            p0 = []
            fwhm_guess = 1.2 * (1.2/1000) / 12 * 180/np.pi # approx beam size, degrees
            for i in range(self.header['n_detectors']):
                if verbose == True:
                    print(self.header['xf_coords'][i])
                    print(self.maps.shape)
                    print(np.nanmax(self.maps[i]))
                plt.clf()
                #plt.imshow(self.maps[i])
                #plt.savefig('/home/vaughan/TIME-analysis/testing_nan_map')
                
                peak_index = np.nanargmax(self.linemaps[i])
                x0 = self.x_center[peak_index]
                if use_feed_offsets:
                    feed_offset_x, feed_offset_y = self.get_feed_offsets(self.header['xf_coords'][i][0],self.header['xf_coords'][i][1],det_coord_mode='xf', map_style='1d')
                    x0 = x0-feed_offset_x
                
                a0 = np.nanmax(self.linemaps[i])

                p0.append([a0,x0,fwhm_guess])
                
            p0 = np.array(p0)

        bounds = ([0,-np.inf,0],[np.inf,np.inf,np.inf])

        fits,covs = self.fit_model(fit_function,xy_mode=xy_mode,use_feed_offsets=use_feed_offsets,p0=p0,nofit_handling=nofit_handling,bounds=bounds, verbose = verbose)

        self.beam_fit_results = {'fits':fits, 'covs':covs, 'function':fit_function}
        self.header['flags']['beam_fits_initialized'] = True

        return fit_function,fits,covs

    def fit_model_det(self,c1,c2,fit_function,xy_mode='offset',use_feed_offsets=False,det_coord_mode='xf',p0=None,nofit_handling='raise',bounds=(-np.inf,np.inf), sigma=False, verbose = False, gradient=False):
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
            feed_offset_x = self.get_feed_offsets(c1,c2,det_coord_mode, map_style='1d')
        else:
            feed_offset_x = 0
        if xy_mode == 'coord':
            x = self.x_center - feed_offset_x / self.dx_scale_factor
        elif xy_mode == 'offset':
            x = self.x_center - feed_offset_x
        else:
            raise ValueError("xy_coord not recognized")

        if p0[0] < 0:
            p0[0] *= -1
        target_map = self.linemaps[det_ind]
        fit_inds = np.isfinite(target_map)

        if sigma:
            try:
                fit,cov = curve_fit(fit_function,(x,fit_inds),target_map[fit_inds],sigma=self.e_maps[det_ind][fit_inds],p0=p0,bounds=bounds, absolute_sigma=True)
            except RuntimeError as err:
                if nofit_handling=='raise':
                    raise err
                else:
                    if verbose == True:
                    	print("No fit found for {}=({},{})".format(det_coord_mode,c1,c2))
                    sig = signature(fit_function)
                    npar = len(sig.parameters)-1
                    fit = np.zeros(npar)
                    fit[:] = np.nan
                    cov = np.zeros((npar,npar))
                    cov[:] = np.nan
            return fit,cov    
        else:
            try:
                # exit()
                fit,cov = curve_fit(fit_function,(x,fit_inds),target_map[fit_inds],p0=p0,bounds=bounds)
            except RuntimeError as err:
                if nofit_handling=='raise':
                    raise err
                else:
                    if verbose == True:
                    	print("No fit found for {}=({},{})".format(det_coord_mode,c1,c2))
                    sig = signature(fit_function)
                    npar = len(sig.parameters)-1
                    fit = np.zeros(npar)
                    fit[:] = np.nan
                    cov = np.zeros((npar,npar))
                    cov[:] = np.nan
            return fit,cov

    def fit_model(self,fit_function,xy_mode='offset',use_feed_offsets=False,p0=None,nofit_handling='raise',bounds=(-np.inf,np.inf),sigma=False, verbose = False, gradient=False):
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
            fit_i,cov_i = self.fit_model_det(xf[0],xf[1],fit_function,det_coord_mode='xf',xy_mode=xy_mode,use_feed_offsets=use_feed_offsets,p0=p0_i,nofit_handling=nofit_handling,bounds=bounds,sigma=sigma, verbose = verbose, gradient=gradient)

            fits.append(fit_i)
            covs.append(cov_i)

        return np.array(fits),np.array(covs)

    def beam_offsets(self,x0,set_header_offsets=False, x0_fits=None, past_fits=False):
        """Compute offsets between all feedhorns and a 
        reference feedhorn
        """

        if not self.header['flags']['beam_fits_initialized'] and not past_fits:
            raise ValueError("No fits to the beam available")

        if x0 not in np.arange(16):
            raise ValueError("x0 bust be between 0 and 15")

        if not past_fits:
            x0_fits = self.beam_fit_results['fits'][(self.get_x(x0))]

        x0_med_x = np.nanmedian(x0_fits[:,1])

        offsets = np.zeros(16)
        offsets[:] = np.nan

        for x in range(16):
            try:
                if not past_fits:
                    x_fits = self.beam_fit_results['fits'][(self.get_x(x))]
                else:
                    x_fits = x0_fits
            except ValueError:
                #print("No detectors found for x={}".format(x))
                continue

            offsets[x] = np.nanmedian(x_fits[:,1])-x0_med_x
        
        offsets = Offsets(offsets,frame=self.header['epoch'],spectrometer=self.header['mc'], map_style='1d')
        if set_header_offsets:
            self.set_feed_offsets(offsets)

        return offsets

    def write(self, filepath, compress=False, overwrite=True):
        """Save linemaps as a ``.npz`` file

        This will save the linemaps as a ``.npz`` file. 
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

        savefunc(filepath, is_linemap=True, 
                 header=self.header,ax_edge=self.ax_edge,linemaps=self.linemaps,counts=self.counts,extras=self.extras)

    def plot(self,*coords,det_coord_mode='xf',savepath=None,show=True):
        """Make plots of detectors 1d binned data
        
        Plot 1d binned data as a function of the binned axis.
        A few different call signatures are possible. 
        ``LineMap.plot_linemap(d1,d2,d3,...)`` will 
        plot the timestream for detectors d1, d2, d3 and 
        so on where di are coordinate pairs in 
        the xf scheme (muxcr scheme is also available by 
        setting the keyword argument `det_coord_mode` to ``'cr'``).
        
        ``LineMap.plot_linemap(key,x1,x2,...)`` will plot the additional
        binned properties in feeds x1, x2, and so on. key should
        be any key from the ``LineMap.extras`` attribute or 'counts', 
        and x1, x2, etc. should be the feedhorn numbers.
        
        Parameters
        ----------
        *coords : tuple or key followed by ints
            What data to plot.
            If the first `coords` value is 'counts' the detector counts
            will be plotted for all following feedhorn numbers. If the 
            first `coords` value is any other string it will be treated
            as a key in ``LineMap.extras`` and the values of this key will 
            be plotted. Otherwise
            the `coords` values will be interpreted as detector 
            coordinates in either the xf (default) or muxcr
            scheme.
        det_coord_mode : {'xf','cr'}, default='xf'
            Coordinate system used to identify detectors.
            'xf' (default) or 'cr' are accepted.
        show : bool, default=True
            If True, plt.show() will be called (set to False if you
            don't want the plot shown immediately)
        savepath : str, optional
            If specified, the resulting plot will be saved in the 
            path provided.

        Returns
        -------
        None

        Examples
        --------
        To plot all detectors:
        >>> LineMap.plot()

        To plot detector xf=(0,0):
        >>> LineMap.plot((0,0))

        To plot detector xf=(0,0) and xf=(15,59):
        >>> LineMap.plot((0,0), (15,59))

        To plot the counts in each for the 0th feedhorn:
        >>> LineMap.plot('counts', 0)

        To plot the extra property 'mean_elevation' in the 
        zeroth and first feedhorn
        >>> LineMap.plot('mean_elevation', 0, 1)
        """

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set(xlabel=self.header['1d_pars']['axis']+' (degrees)', ylabel='Counts [Arb]')
        if self.header['1d_pars']['axis'] in ['R.A.']:
            ax.invert_xaxis()

        # If no coords are given, plot all detectors
        if len(coords) == 0:
            coords = self.header['xf_coords']
        
        if coords[0] == 'counts':
            for i in coords[1:]:
                ax.plot(self.x_center,self.counts[i],label=i)
            ax.set(ylabel='Samples in Bin')
            ax.legend(fontsize='x-small',title='Feedhorn')
        elif coords[0] in self.extras.keys():
            for i in coords[1:]:
                ax.plot(self.x_center,self.extras[coords[0]][i],label=i)
            ax.legend(fontsize='x-small',title='Feedhorn')
        else:     
            c1 = [c[0] for c in coords]
            c2 = [c[1] for c in coords]
            for i in range(len(coords)):
                ax.plot(self.x_center,self.linemaps[self._get_coord(c1[i],c2[i],det_coord_mode)],label='({},{})'.format(c1[i],c2[i]))
            ax.legend(fontsize='x-small',title=det_coord_mode)
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()

        plt.close(fig)
class LineMapConstructor(LineMap):

    def __init__(self,header,m1d_pars,x,linemaps,counts,extras={}):

        self._check_header(header)

        self.header = copy.deepcopy(header)
        self.header['1d_pars'] = copy.deepcopy(m1d_pars)
        self.header['flags']['1d_initialized'] = True
        
        self.ax_edge = np.copy(x)
        self.x_center = np.array([(self.ax_edge[i]+self.ax_edge[i+1])/2 for i in range(len(self.ax_edge)-1)])
        
        self.linemaps = np.copy(linemaps)
        self.counts = np.copy(counts)
        self.extras = copy.deepcopy(extras)

        self._input_consistency_check()