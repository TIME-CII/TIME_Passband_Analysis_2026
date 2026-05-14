"""This module defines Timestream, the basic class
for handling TIME's level 1 data products
(raw and processed timestreams).
"""

from timesoft.maps.map_tools import MapConstructor
from timesoft.maps.linemap_tools import LineMapConstructor
from timesoft.timestream.gridding_utils import mk_grid
from timesoft.raw.loading_utils import get_data
from timesoft.helpers import coordinates
from timesoft.helpers._class_bases import _Datastruct_base
from timesoft.logger.time_logger import time_logger
from timesoft.calibration import DetectorConstants
# from timesoft.detector_cuts import first_cut
from timesoft.calibration import Offsets
from timesoft.helpers.nominal_frequencies import f_ind_to_val
from ast import literal_eval
from astropy.constants import h, k_B, c
from astropy.coordinates import get_body
from scipy.interpolate import interp1d
import pandas as pd 
import sys, os, glob
import itertools
import warnings
import copy
import datetime
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import deconvolve
from scipy . special import erf
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, TETE
import astropy.units as u

import matplotlib

pi = np.pi

warnings.resetwarnings()

# SOME DATASET VARIABLES:
nchan = 60
nfeed = 16

# SOME TELESCOPE VARIABLES
lon12m = -(111 + 36/60 +54/3600)
lat12m = (31+57/60+10.8/3600)
height12m = 1897.3

# class to hold data from telescope and produce fit
class Timestream(_Datastruct_base):
    """Class to hold timestream data and perform data reduction

    The Timestream class loads data from a raw data file and provides numerous
    methods for processing and analyzing it. It is initialized by providing a
    path to the folder containing the raw data to be analyzed. 
    
    A ``Timestream`` instance can be initialized two ways. First it can be
    created from a directory containing the raw netCDF files for a single TIME
    observation. In this case the the class can be initialized by specifying the
    path of the directory containing the raw data in the `path` parameter, and
    the MCE to load in the `mc` parameter. A number of additional options are
    available to control the exact data loaded and some other behaviors.

    Alternatively, a ``Timestream`` instance can be initialized from a
    pre-existing timestream data file (saved as a .npz document), by specifying
    the path and filename to the file in the `path` parameter.

    The initialization of a class will guess which of these two modes to use
    based on input of the ``path`` parameter. Paths ending in '.npz' will be
    assumed to be level 1 timestream files, while anything else will be assumed
    to be level 0 raw data files.

    The resulting object will be a class instance containing the relevant data
    (positions in ra/dec and az/el, detector counts, flags, and some header
    information). These can either be accessed directly, or there are a numerous
    data reduction methods associated with the class instance that can be used
    for further data processing.

    Parameters
    ----------

    path : str
        The path to the raw data directory or timestream file to be loaded.
    files_in : list or None, optional
        For loading raw data only - if a timestream file is loaded it is
        ignored. If specified this should be a list containing the names of the
        files in the directory (as specified in `path`) to be loaded. If
        unspecified (or None), all files ending in '.nc' will be loaded.
    mc : {0, 1}, default=0 
        Determines which MCE to load. For loading raw data only - if a
        timestream file is loaded it is ignored and the MCE can be determined
        from the header.
    xf : list of tuples, optional
        Can be used to specify the detectors to load in terms of the xf
        coordinates. If specified, it should be a list of xf pairs e.g.
        ``xf=[(1,1), (15,2), (12,29)]``, and only data for those detectors will
        be returned. If not specified data for all detectors available in the
        data will be returned. If specified for a timestream file and some of
        the detectors listed are not present in the file, they will be ignored.
        Only one of xf or cr can be used (the other should be set as None or
        left out of the function call). 
    cr : list of tuples, optional
        Can be used to specify the detectors to load in terms of the muxcr
        coordinates. `cr` works analogously to `xf` except for the indexing
        system used. Only one of xf or cr can be used (the other should be set
        to None or left out of the function call).
    version : {'2022.dev.1', '2022.dev.0', '2019.final', '2019.dev.1',
    '2019.dev.0'}, optional
        For loading raw data only - if a timestream file is loaded it is
        ignored. Specifies which version of the raw data file structure is in
        use. The default is '2022.dev.1' which works for most data taken during
        the 2022 engineering run. Other versions may be needed for older data:
        '2022.dev.0' for a few very early observations from the 2022 run;
        '2019.final' for data from 2019-03-15 to end of 2019 engineering run;
        '2019.dev.1' for data from 2019-03-14 or earlier; and '2019.dev.0' for
        very first data from 2019 run
    store_copy : bool, default=True
        If True, then an untouched copy of the data will be stored in parallel
        to the editable version. This copy can then be restored using the
        ``Timestream.reset()`` method should you wish to undo your data edits,
        making it useful for interactively reducing the data. Default is False.
        If loading a timestream file which does not already contain a copy of
        the data, this parameter is ignored
    raisenotel : bool, default=False
        For loading raw data only - if a timestream file is loaded it is
        ignored. Determines the handling of files with no valid telescope
        telemetry data. If `raisenotel=True`, a ValueError will be raised. If
        `raisenotel=False` no error will be raised, but the returned telescope
        data (ra, dec, az, el, and tel_flags) will be arrays of filled with
        `np.nan`.
    mode : {'auto','raw','timestream'}, default='auto'
        This specifies whether the path is for raw data or timestream data. The
        default, is 'auto' which tries to determine what type of data is being
        loaded based on the ending of the `path` string.
    impose_frame : str, default = 'J2000_soft'
        if impose_frame = 'None', uses the default reference frame (reference
        frame assigned when observing) when computing ra and dec information if
        a reference frame is indicated, transform all ra and dec information to
        desired reference frame using ts.convert_coordinates(). if the indicated
        reference frame is followed by '_soft', do the transform on only the
        original copy if the indicated reference frame is followed by '_hard',
        do the transform also on the second copy in stored_copy (see below).
        frame options are: {'J2000', 'Galactic', 'apparent'} as consistant with
        ts.convert_coordinates(). 
    lon, lat, height : floats
        Latitude, longitude and altitude (in degrees, degrees, meters) of the
        telescope to use for frame transformations during initialization
        Defaults to coordinates for the ARO 12m
    telname : str
        The name of the telescope
    instname : str
        The name of the instrument


    Notes
    -----

    **Loading Raw Data Files**

    The ``Timestream`` class can be initialized by specifying a directory
    containing the netCDF files for a single TIME observation. If data was taken
    in spring 2022, the path is the only argument required. Otherwise the
    "version" parameter should be specified. Optionally the loaded data can be
    restricted to a subset of the files in the path using the files_in
    parameter, and to a subset of detectors using the xf or rc prameters. 

    For command line use, it may be useful to set "store_copy=True" when
    initializing the class - this will cause an unedited version of the
    timestream to be stored, making it possible to reset the data to its
    original form without the need to re-load. For scripts this can usually be
    left in its default (False), as there are fewer use cases for resetting the
    data.

    **Loading Processed Timestream Files**

    The ``Timestream`` class can also be initialized by specifying the path to
    an existing timestream file. In this case, many of the __init__ parameters
    used to control the loading of raw data are irrelevant, in particular, `mc`,
    `files_in`, and `version` are ignored, as this information was already set
    when the timestream was first created (and recoreded in the header of the
    timestream file). Detectors can still be specified with the `xf` and `cr`
    parameters, but if a detector is not present in the saved timestream it will
    be skipped. Similarly, an unedited copy of the data can be loaded with
    `store_copy`, but only if it was previously saved in the timestream file.

    **Contents of the** ``Timestream.header`` 

    mc : int n_detectors : int n_samples : int med_time : float observer : array
    object : string

    datetime : masked array
         {data, mask, fill_value, dtype}

    detector_pars : dict
         {datamode : array, 
          readout : array, live_detectors : masked_array, mask : list,
          fill_value : float, dtype}

    scan_pars : dict
         {'type', 'direction', 'crossing_time', 'scan_width', 'map_height',
         'map_row_spacing'}
    
    original_data_path : string original_files_in : string filesystem_version :
    string xf_coords : array cr_coords : array center_ra : float center_dec :
    float epoch : string filter_type : string flags : dict
          {has_tel_data : bool,
           has_tau_data : bool, has_feed_offsets : bool, has_gains : bool,
           has_beams : bool, has_time_constants : bool, corrected_tau : bool,
           corrected_gains : bool, corrected_time_constants : bool, store_copy :
           bool, scan_flags_initialized : bool, filtering_applied : bool,
           maps_initialized : bool, 1d_initialized : bool, beam_fits_initialized
           : bool}

    date : string tau : float


    **Keeping Track of Detector Indices**
     
    The detector data is stored in the ``Timestream.data`` attribute, which is a
    2d array of shape (len(xf), len(t)). Because the number of detectors loaded
    is flexible, the index in this array corresponding to a given detector can
    varry between different instances of the ``Timestream`` class. To handle
    this two helper methods are defined as part of the class:

    get_xf(x, f)
        Determine the index in the ``Timestream.data`` array corresponding to
        the detector with xf indices (x,f)
    get_cr(c, r)
        Determine the index in the ``Timestream.data`` array corresponding to
        the detector with muxcr indices (c,r)

    Additionally, any analysis methods which operate on a single detector are
    constructed to take detector indexing in xf or muxcr coordinates. Generally
    the call signatures for these functions will look something like
    ``Timestream.method(c1,c2[,other args],det_coord_mode='xf')``, where
    ``det_coord_mode`` specifies whether the coordinates are given in xf or
    muxcr coordinates, and ``(c1, c2)`` are the coordinates of the detector in
    the specified coordinate system.

    **Second Copy of the Data with** ``store_copy=True``

    When a timestream is initialized with ``store_copy=True`` an second copy of
    many attributes is created, which will be left untouched by the data
    processing methods. These can be used to compare the timestream before and
    after some processing tasks, but are mostly useful to enable the
    ``Timestream.reset()`` method, which will undo any changes made to the data
    and restore it to its original form.

    Whether a timestream contains this second copy can be checked by inspecting
    the ``Timestream.header['flags']['store_copy']`` attribute. If a stored copy
    exists, it will produce the follwing additional attributes:

    t_copy : ndarray
        Unedited copy of `t`
    ra_copy, dec_copy, az_copy, el_copy, tel_flags_copy : ndarray 
        Unedited copies of `ra`, `dec`, `az`, `el`, and `tel_flags`
    data_copy : ndarray
        Unedited copy of `data`

    **Performance and Debugging**

    The slowest processes appear to be loading the data from NCDF files, and
    filtering the scans. To reduce issues with the first, an unmodified copy of
    the loaded data can be stored permanently by setting "store_copy=True", and
    at any point you can restore this using ``Timestream.reset()``. This means
    you should only have to load once. The filtering is slow because it has to
    loop through each detector separately, if you have ideas how to improve this
    let me know.

    Despite this, you should be able to get from raw data to a full set of maps
    in only a few minutes.

    For testing settings, you can save time by filtering and mapping only one
    detector using the following methods:

    filter_scan_det(x, f, n)
        Speficy feed horn and channel with x, f and polynomial degree with n
    make_map_det(x, f, plot=True, pixel)
        makes the single detector map and produces a plot of it.
    
    These should work very quickly. You can also use muxcr indexing instead of
    x, f indexing. See the docs for these methods for more details.

    The filtering speed can also be improved by loading only the desired set of
    detectors when initializing.

    **Attribute and Method Listings**
    
    Attributes
    ----------

    mc : {1, 0}
        The MCE from which the data is taken
    header : dict
        A dictionary containing header informatin and flags used in the
        analysis.
    t: ndarray
        One dimensional array containing the timestamp of each sample. Times are
        given as unix times.
    data : ndarray
        An array of shape (len(xf), len(t)) which contains the readouts from
        each detector. The first index is ordered either by the order in which
        they were specified in the class initialization, or, if no detectors
        were specified, by increasing x then f. 
    ra, dec : ndarray
        Arrays containing the R.A. and declination associated with each sample
        in the timestream.
    az, el : ndarray
        Arrays containing the azimuth and elevation associated with each sample
        in the timestream.
    tel_flags : ndarray
        Array containing the 'direction' flags reported by the telescope. The
        values and associated meanings are: 0 - telescope is not on source
        (idle); 1 - telescope is tracking the field center; 2 - telescope is
        moving in the minus R.A. dirction; 3 - telescope is moving in the plus
        R.A. direction; 4 - telescope is turning around

    scan_flags : ndarray
        Initially an array of zeros. Once the timestream has been broken up into
        individual scans across the field using ``Timestream.flag_scans()``, it
        will contain the scan number to which each sample has been assigned.
    scan_direction_flags : ndarray
        Initially an array of zeros. Once the direction of scans has been
        determined it will contain that information (note this is now somewhat
        redundant with the flags attribute)
        

    Methods
    -------
    
    get_xf
        Determine the index in the ``Timestream.data`` array corresponding to
        the detector with xf indices (x,f)
    get_cr
        Determine the index in the ``Timestream.data`` array corresponding to
        the detector with muxcr indices (c,r)
    get_x
        Get the indices of all detectors of a specific feedhorn
    get_f
        Get the indices of all detectors of a specific frequency index
    _remove_samples
        Remove specified samples from the timestream data
    
    write
        Save generated maps as a ``.npz`` file
    write_map
        Save generated maps as a ``.npz`` file
    write_1d
        Save generated 1d binned data as a ``.npz`` file
    reset
        Reset the timestream to undo the effects of any analysis methods
        applied. 
    
    flag_scans
        Identify individual scans of the telescope across the map region
    renumber_scans
        Reset the indexing of the scans so that they run continuously from 1 to
        n
    remove_end_scans
        Remove a speciied number of scans from the start or end of an
        observation
    remove_scan_direction
        Remove all scans not along a specified scan direction
    remove_scan_edge
        Remove samples at the edges of scans
    remove_scan_flag
        Remove data segments that are not identified as part of a scan (e.g.
        turnarounds)
    remove_short_scans
        Remove scans with fewer than a specified number of samples
    remove_tel_flag
        Remove samples with specified values of ``Timestream.flag`` 

    convert_coordinates
        Transform data between coordinate frames
    set_offsets
        Add information about the position offsets between feeds
    offset_pos
        Get timstreams for the positions of a specific feedhorn

    filter_scan_det
        Polynomial filtering of detector counts
    filter_scan_det
        Polynomial filtering of detector counts  with source masking
    filter_scan
        Polynomial filtering of all detectors
    set_tau
        Add atmospheric opacity information
    correct_tau
        Correct atmospheric attenuation
    set_gains
        Add information about the conversion between counts and flux
    correct_gains
        Convert data from counts to flux
    set_time_constants
        Add information about the time constants of the detectors
    correct_time_constants
        Deconvolve time constants from timestreams (not implemented)

    make_map
        Make maps of all detectors
    make_map_det
        Quicklook map of a single detector
    plot_map
        View map of a single detector
    detector_grid
        Creates a plot showing all detectors in a grid
    make_1d
        Bin samples along a specified axis
    plot_1d
        Make plots of detectors 1d binned data
    skip_last
        Skip the last file when generating a list automatically (set to True when using a dataset in making)
    """
    ########################################
    #### LOADING AND SAVING TIMESTREAMS ####
    ########################################
    def __init__(self, path, files_in=None, mc=0, xf=None, cr=None, version='2026.dev.0', store_copy=False, raisenotel=False, mode='auto',impose_frame='J2000_soft',lon=lon12m,lat=lat12m,height=height12m,telname='ARO 12m',instname='TIME',skip_last=True):

        if mode not in ['auto','raw','timestream']:
            raise ValueError("mode parameter not recognized")
        if mode == 'auto':
            if path.endswith('.npz'):
                mode = 'timestream'
            else:
                mode = 'raw'
        if mode == 'raw' and not path.endswith('/'):
            path = path + '/'
        
        self.mc = mc
        # For loading raw data
        if mode == 'raw':
            # Check inputs
            if mc not in [0,1]:
                print('invalid mc')
                return

            # Check that a maximum of one of xf and cr have been provided
            if xf is not None and cr is not None:
                raise ValueError('cannot specify both xf and cr indexing')

            # Convert between xf and cr
            if cr is not None:
                xf = [coordinates.muxcr_to_xf(cri[0],cri[1],p=mc) for cri in cr]
            elif xf is not None:
                if mc == 0:
                    cr = [coordinates.xf_to_muxcr(xfi[0],xfi[1],p=mc) for xfi in xf]
                elif mc == 1:
                    cr = [coordinates.xf_to_muxcr_B(xfi[0],xfi[1]) for xfi in xf]
            else:
                xf = [(x,f) for x,f in itertools.product(np.arange(nfeed),np.arange(nchan))]
                if mc == 0:
                    cr = [coordinates.xf_to_muxcr(xfi[0],xfi[1],p=mc) for xfi in xf]
                elif mc == 1:
                    cr = [coordinates.xf_to_muxcr_B(xfi[0],xfi[1]) for xfi in xf]


            if version in ['2024.dev.1','2026.dev.0']:
                base_path = Path(path)
                run_id = base_path.name  # use name directly, don't strip 'mce_' prefix
                hk_path = base_path.parent / f"hk_netcdf_files/hk_{run_id}"
                mce_path = base_path.parent / f"netcdf_files/mce_{run_id}"
            else: 
                hk_path = None 
                mce_path = path 
            self.t, self.ra, self.dec, self.az, self.el, self.data, self.tel_flags, self.tau, header = get_data(mce_path, hk_path, mc, version, files_in, xf=xf, raisenotel=raisenotel, skip_last=skip_last)

            print(hk_path)
            print(mce_path)
            # # Set up a dictionary to contain header information
            # The current verion of the data structure is missing some 
            # useful metadata about the observations, hopefully these 
            # will be added in the future. For now the loading tools
            # leave these as nans or 'unknown'.
            if version == '2024.dev.1':
                datetime = run_id 
            else:
                datetime = header['datetime']

            # For observations where a planet has been renamed, change the name back to 
            # a real planet
            planets = ['Mercury','Venus','Mars','Jupiter','Saturn','Uranus','Neptune']
            replacements = ['Hermes','Aphrodite','Ares','Zeus','Cronus','Ouranos','Poseidon']
            if header['object'] in replacements:
                i = replacements.index(header['object'])
                header['object'] = planets[i]

            self.header = header
            self.header['n_samples'] = len(self.t)
            self.header['med_time'] = np.median(self.t)

            self.header['n_detectors'] = len(xf)
            self.header['xf_coords'] = np.empty(self.header['n_detectors'], dtype=object)
            self.header['xf_coords'][:] = xf
            self.header['cr_coords'] = np.empty(self.header['n_detectors'], dtype=object)
            self.header['cr_coords'][:] = cr

            # These are additional flags that will try to tell which scan the
            # telescope is on and what direction it is scanning
            self.scan_flags = np.zeros(self.tel_flags.shape, dtype='int')
            self.scan_direction_flags = np.zeros(self.tel_flags.shape, dtype='int')

            # Store a second copy of data that won't be modified
            if store_copy:
                self.t_copy = np.copy(self.t)
                self.ra_copy = np.copy(self.ra)
                self.dec_copy = np.copy(self.dec)
                self.az_copy = np.copy(self.az)
                self.el_copy = np.copy(self.el)
                self.tel_flags_copy = np.copy(self.tel_flags)
                self.tau_copy = np.copy(self.tau)
                self.data_copy = np.copy(self.data)

            # Flags for keeping track of analysis
            self.header['filter_type'] = 'none'
            self.header['flags'] = {'has_tel_data':header['has_tel_data'],
                                    'has_tau_data':header['has_tau_data'],
                                    'has_feed_offsets':False,
                                    'has_gains':False,
                                    'has_beams':False,
                                    'has_time_constants':False,
                                    'corrected_tau':False,
                                    'corrected_gains':False,
                                    'corrected_time_constants':False,
                                    'store_copy':store_copy,
                                    'scan_flags_initialized':False,
                                    'filtering_applied':False,
                                    'maps_initialized':False,
                                    '1d_initialized':False,
                                    'beam_fits_initialized':False}
        
            self.header['telescope_pars'] = {'longitude':lon,
                                             'latitude':lat,
                                             'height':height,
                                             'name':telname,
                                             'instrument':instname}
            #self.convert_coordinates(self.header['epoch'])
            
            # If available, make the commanded coordinates the "center"
            if self.header['command_epoch'] != 'unknown':
                if self.header['command_epoch'] != self.header['epoch']:
                    warnings.warn("Commanded position didn't get converted to match data")
                else:
                    self.header['center_ra'] = self.header['command_ra']
                    self.header['center_dec'] = self.header['command_dec']



        # For re-loading a pre-existing timestream
        if mode == 'timestream':

            file_in = np.load(path, allow_pickle=True)
            if 'is_timestream' not in file_in.files:
                raise ValueError("The specified file does not appear to be a TIME timestream")

            self._check_header(file_in['header'][()])
            self.header = file_in['header'][()] # This weird syntax turns back into a dictionary

            self.t = file_in['t']
            self.ra = file_in['ra']
            self.dec = file_in['dec']
            self.az = file_in['az']
            self.el = file_in['el']
            self.tau = file_in['tau']

            self.data = file_in['data']

            self.tel_flags = file_in['tel_flags']
            self.scan_flags = file_in['scan_flags']
            self.scan_direction_flags = file_in['scan_direction_flags']

            if store_copy and self.header['flags']['store_copy']:
                self.t_copy = file_in['t_copy']
                self.ra_copy = file_in['ra_copy']
                self.dec_copy = file_in['dec_copy']
                self.az_copy = file_in['az_copy']
                self.el_copy = file_in['el_copy']
                self.tau_copy = file_in['tau_copy']
                self.data_copy = file_in['data_copy']
                self.tel_flags_copy = file_in['tel_flags_copy']

            else:
                self.header['flags']['store_copy'] = False

            if xf is not None or cr is not None:
                if xf is not None and cr is not None:
                    raise ValueError('cannot specify both xf and cr indexing')

                elif xf is not None:  
                    c1c2 = xf
                    det_coord_mode = 'xf'
                elif cr is not None:
                    c1c2 = cr
                    det_coord_mode = 'cr'

                # print(c1c2)
                self.restrict_detectors(c1c2,det_coord_mode)
        
        """# Do coordinate transfrom to the desired frame
        if impose_frame != 'None' and impose_frame is not None:
            new_frame,frameforcing = impose_frame.split("_")
            print('self.header.epoch in timestream tools',self.header['epoch'])
            if frameforcing == 'soft':
                print(new_frame)
                self.convert_coordinates(newframe=new_frame,transform_copy=False)
            elif frameforcing == 'hard' and store_copy==True:
                self.convert_coordinates(newframe=new_frame,transform_copy=True)
            else:
                print('need to indicate if the coordinate reference frame in the  stored copy needs to be transformed.')
            self.header['epoch'] = new_frame"""
    
    def write(self, filepath, store_copy=True, compress=False, overwrite=True):
        """Save a timestream as a ``.npz`` file

        This will save the timestream, complete with
        any processing on the timestream itself (but 
        not any associated maps or binned 1d data).
        The saved file can be loaded again for further
        analysis.

        Parameters
        ----------
        filepath : str
            The path and filename where the data should
            be saved. ``.npz`` will be appended to the
            end of the file name if it is not already 
            included.
        store_copy : bool, default=True
            If the current timestream includes a stored
            copy this can be set to True to include it
            in the saved file or False to exclude it.
            If no stored copy is present, this parameter
            is ignored.
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

        # Reset the map flags and get rid of the map parameters, since we won't be saving maps
        new_header = copy.deepcopy(self.header)
        new_header['flags']['maps_initialized'] = False
        new_header['flags']['1d_initialized'] = False

        if not store_copy or not self.header['flags']['store_copy']:
            new_header['flags']['store_copy'] = False
            savefunc(filepath, is_timestream=True, 
                    header=new_header,
                    t=self.t, ra=self.ra, dec=self.dec, az=self.az, el=self.el, tau=self.tau,
                    data=self.data, 
                    tel_flags=self.tel_flags, scan_flags=self.scan_flags, scan_direction_flags=self.scan_direction_flags,
                    )
        else:
            savefunc(filepath, is_timestream=True, 
                    header=new_header,
                    t=self.t, ra=self.ra, dec=self.dec, az=self.az, el=self.el, tau=self.tau,
                    data=self.data, 
                    tel_flags=self.tel_flags, scan_flags=self.scan_flags, scan_direction_flags=self.scan_direction_flags,
                    t_copy=self.t_copy, ra_copy=self.ra_copy, dec_copy=self.dec_copy, az_copy=self.az_copy, el_copy=self.el_copy, tau_copy=self.tau_copy,
                    data_copy=self.data_copy, 
                    tel_flags_copy=self.tel_flags_copy
                    )



    def write_map(self, filepath, compress=False, overwrite=True):
        """Save generated maps as a ``.npz`` file

        This will save the maps generated by 
        ``Timestream.make_map()`` as a ``.npz`` file. 
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

        if not self.header['flags']['maps_initialized']:
            raise ValueError("No maps have been created")

        self.Maps.write(filepath=filepath, compress=compress, overwrite=overwrite)


    def write_1d(self, filepath, compress=False, overwrite=True):
        """Save generated 1d binned data as a ``.npz`` file

        This will save the 1d binned data generated by 
        ``Timestream.make_1d`` as a ``.npz`` file. 
        The saved files can be loaded with utilities 
        from the ``timesoft.maps`` module for further
        analysis.

        This is a wrapper around ``Timestream.LineMaps.write``.

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

        if not self.header['flags']['1d_initialized']:
            raise ValueError("No binned data have been created")

        self.LineMaps.write(filepath=filepath, compress=compress, overwrite=overwrite)


    def reset(self):
        """Reset the timestream to undo the effects of any analysis methods applied.

        This function resets the timestream to look exactly 
        how it did when loaded from the raw data files. This
        gets around reloading everything, which can be slow,
        particularly for large datasets.

        Arguments
        ---------
        None

        Returns
        -------
        None

        Notes
        -----
        Values of ``Timestream.t``, ``Timestream.ra``, ``Timestream.dec``,
        ``Timestream.az``, ``Timestream.el``, ``Timestself.tel_flags``, and
        ``Timestream.data`` are reset to their original values. Assigned
        scan numbers, scan directions, etc. are also reset.

        ``Timestream.header['flags']['store_copy']`` must be True in order 
        for this to work. This means the timestream must have been iniitialized 
        with a copy of the data. Otherwise an error will be raised.
        """
        if self.header['flags']['store_copy']:
            self.t = np.copy(self.t_copy)
            self.ra = np.copy(self.ra_copy)
            self.dec = np.copy(self.dec_copy)
            self.az = np.copy(self.az_copy)
            self.el = np.copy(self.el_copy)
            self.tau = np.copy(self.tau_copy)
            self.tel_flags = np.copy(self.tel_flags_copy)
            self.data = np.copy(self.data_copy)
            self.scan_flags = np.zeros(self.tel_flags.shape, dtype='int')
            self.scan_direction_flags = np.zeros(self.tel_flags.shape, dtype='int')
            self.header['flags']['scan_flags_initialized'] = False
            self.header['flags']['corrected_tau'] = False
            self.header['flags']['filtering_applied'] = False


        else:
            raise ValueError("No copy of data for reset.")


    ##########################
    #### HELPER FUNCTIONS ####
    ##########################
    # Tell python what to write when print(timesteam) is called
    def __str__(self):
        if self.header['scan_pars']['type'] in ['1D','1d']:
            scan_str = "\tobservation: 1D scan, \n\t\tlength {:.1} degrees in the {} direction\n".format(self.header['scan_pars']['scan_width']*60,self.header['scan_pars']['direction'])
        elif self.header['scan_pars']['type'] in ['2D','2d']:
            scan_str = "\tobservation: 2D scan, \n\t\trows of length {:.1} arcmin in the {} direction, \n\t\tseparated by {:.1} arcmin\n".format(self.header['scan_pars']['scan_width']*60,self.header['scan_pars']['direction'],self.header['scan_pars']['map_row_spacing']*60)
        elif self.header['scan_pars']['type'] in ['unknown']:
            scan_str = "\tobservation: unknown scan type\n"
        else:
            scan_str = "\tobservation: {}\n".format(self.header['scan_pars']['type'])

        str = "timesoft.Timestream instance\n\n" +\
              "\tsource: {}\n".format(self.header['object']) +\
              scan_str +\
              "\tdetectors in dataset: {}\n".format(self.header['n_detectors']) +\
              "\tsamples per detector: {} (cleaned data), {} (raw data)\n\n".format(len(self.t),self.header['n_samples']) +\
              "current flags:\n" +\
              "\tstored copy availabe: {}\n".format(self.header['flags']['store_copy']) +\
              "\tvalid telescope data: {}".format(self.header['flags']['has_tel_data']) +\
              "\t(coordinates represented in: {})\n".format(self.header['epoch']) +\
              "\tscans identified: {}\n".format(self.header['flags']['scan_flags_initialized']) +\
              "\tfiltering applied: {}".format(self.header['flags']['filtering_applied']) +\
              "\t(filtering type: {})\n".format(self.header['filter_type']) +\
              "\tmaps constructed: {}\n".format(self.header['flags']['maps_initialized']) +\
              "\tlinemaps constructed: {}".format(self.header['flags']['1d_initialized'])

        return str

    
    def _get_scan_direction(self):
        """Logic tree for determining scan direction (for internal use)"""

        if self.header['scan_pars']['direction'] in ['ra','RA','r.a.','R.A.']:
            scan_direction = 'ra'
        elif self.header['scan_pars']['direction'] in ['dec','Dec','DEC','declimation','Declination']:
            scan_direction = 'dec'
        elif self.header['scan_pars']['direction'] == 'unknown':
            warnings.warn("Warning: scan direction is unknown, assuming RA")
            scan_direction = 'ra'
        else:
            raise ValueError("Could not determine scan direction")

        return scan_direction


    def _remove_samples(self,indices):
        """Remove specified samples from a timestream

        This is a convenience function to remove the data
        corresponding to a specified set of samples from
        all attributes containing timestream data. It is a
        helper function for making data cuts in a way that
        keeps the various timestream arrays in synch with 
        one another.

        Parameters
        ----------
        indices : array_like
            The list of indices corresponding to the samples
            in the timestream you want to remove.

        Returns
        -------
        None

        Notes
        -----
        All data cuts should be accomplished with this method. 
        This ensures that if new timestream arrays are incorporated
        in future versions of this class, the code will only
        need to be changed in one place to update them.

        Examples
        --------
        To remove all data with an declination less than zero you would 
        call
        >>> Timestream._remove_samples(np.nonzero(Timestream.dec<0)[0])
        """

        self.t = self.t[indices]
        self.ra = self.ra[indices]
        self.dec = self.dec[indices]
        self.az = self.az[indices]
        self.el = self.el[indices]
        # print(self.tau, 'self.tau')
        self.tau = self.tau[indices]
        self.tel_flags = self.tel_flags[indices]
        self.scan_flags = self.scan_flags[indices]
        self.scan_direction_flags = self.scan_direction_flags[indices]

        self.data = self.data[:,indices]


    def restrict_detectors(self,new_c1c2,det_coord_mode='xf'):
        """Limit dataset to a subset of the current detectors
        
        This method removes all but a specified set of detectors
        from the ``Timestream`` instance. Removed detectors
        are not recoverable (even by resetting to the unedited
        copy of the data), except by re-loading the original 
        data file.

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

        self.data = self.data[(inds)]
        self.header['xf_coords'] = self.header['xf_coords'][(inds)]
        self.header['cr_coords'] = self.header['cr_coords'][(inds)]
        if self.header['flags']['store_copy']:
            self.data_copy = self.data_copy[(inds)]
        self.header['n_detectors'] = len(inds)

        if self.header['flags']['maps_initialized']:
            self.Maps.restrict_detectors(new_c1c2,det_coord_mode)
        if self.header['flags']['1d_initialized']:
            self.LineMaps.restrict_detectors(new_c1c2,det_coord_mode)

        if self.header['flags']['has_gains']:
            self.header['gains'] = self.header['gains'][(inds)]
            self.header['gains_e'] = self.header['gains_e'][(inds)]
        if self.header['flags']['has_time_constants']:
            self.header['time_constants'] = self.header['time_constants'][(inds)]
        if self.header['flags']['has_beams']:
            self.header['beams'] = self.header['beams'][(inds)]

    ##################################################################################
    #### FOR IDENTIFYING SCANS AND REMOVING DATA BASED ON SCAN AND TELESCOPE INFO ####
    ##################################################################################
    def flag_scans(self, dabs=None, dperp=None, scan_direction=None, remove_tel_flags=True, save_dir=None):
        """Identify individual scans of the telescope across the map region

        This function attempts to identify individual scans 
        over the mapped region and break up the timestream 
        accordingly. It sets the values of ``Timestream.scan_flags``
        and ``Timestream.scan_direction_flags`` attributes.

        ``Timestream.scan_flags`` is set by inspecting the direction
        flags from the telescope and identifying points in the
        timestream data where the flags change. These points are
        used to break the data up into scans, which are then numbered
        starting with the first scan in time as 1. The 
        ``Timestream.scan_flags`` attribute contains the scan number 
        of which each sample is a part. Index 0 is reserved for data
        which are not part of a scan.

        ``Timestream.scan_direction_flags`` is set by determining 
        whether a given scan is in an increasing or decreasing 
        direction along the scan coordinate. Scans with increasing
        values are given a value of 1 and scans with decreasing
        values are given a value of zero.

        Upon successfuly running this method will set 
        ``Timestream.header['flags']['scan_flags_initialized'] == True``.

        The inputs for this function are largely for dealing with
        legacy datasets. Telescope flags for the 2018 observing run 
        did accurately track scans, hence scans can also be identified
        when the steps between data points exceed some tolerance. 
        Typically this tolerance should be for movement perpendicular
        to the scan direction (`dperp` parameter), but some data also 
        has large loops in the turnarounds where the telescope moves
        quickly and can be filtered out by setting an absolute tolerance
        (`dabs` parameter).

        Parameters
        ----------
        dabs : float, optional
            The tolarance for total change in position between two
            consecutive samples before they are considered parts of 
            separate scans. Specified in degrees. By default no 
            tolerance is applied, as the telescope flags alone are 
            sufficient to identify scans in 2022+ data
        dperp : float, optional
            The tolerance for change in position perpendicular to the
            scan direction between two consecutive samples before they 
            are considered parts of separate scans. Specified in degrees.
            By default no tolerance is applied, as the telescope flags 
            alone are sufficient to identify scans in 2022+ data.
        scan_direction : {'ra', 'dec'}, optional
            Direction along which the scan is moving.
            For data up to the 2022 engineering run, scan direction 
            information is not stored in the data header. It can 
            optionally be specified here, otherwise the direction 
            will be assumed to be RA since the vast majority of scans
            during the engineering runs were conducted in this mode.
            For newer data this parameter can be left unspecified and
            the scan direction will be determined from the header.
        remove_tel_flags : bool, default=True
            If True, samples where the telescope telemetry indicates 
            that the telescope is waiting at the map center, moving
            to the starting point of the raster, or turning around
            (``Timestream.flag`` equal to 0, 1, or 4) are automatically
            excluded from scan patterns.

        Returns
        -------
        None
        """

        if not self.header['flags']['has_tel_data']:
            raise ValueError("No telescope data are available, scans cannot be identified")

        scan_flags = np.zeros(self.tel_flags.shape, dtype='int')
        scan_direction_flags = np.zeros(self.tel_flags.shape, dtype='int')

        # Determine which direction the telescope is moving

        if scan_direction is not None:
            if scan_direction in ['ra','RA','r.a.','R.A.']:
                scan_direction = 'ra'
            elif scan_direction in ['dec','Dec','DEC','declimation','Declination']:
                scan_direction = 'dec'
            else:
                raise ValueError("Scan direction not recognized")
            
            if self.header['direction'] != 'unknown':
                warnings.warn("Warning: overriding header scan direction")

        else:
            scan_direction = self._get_scan_direction()


        # Determine the change in dec, pointing angle, and flag from 
        # timestep to timestep and pick out where the declination or 
        # position exceed a tolerance or the flags are changing
        if dperp is not None:
            if scan_direction == 'ra':
                difs1 = np.abs(self.dec[:-1]-self.dec[1:]) # Large dec change
            elif scan_direction == 'dec':
                difs1 = np.abs(self.ra[:-1]-self.ra[1:]) # Large ra change
            inds1 = np.nonzero(difs1 > dperp)[0]
        else:
            inds1 = np.empty(0,dtype='int')
        if dabs is not None:
            difs2 = np.sqrt((np.cos((self.dec[:-1]+self.dec[1:])/2*np.pi/180)*(self.ra[:-1]-self.ra[1:]))**2 + (self.dec[:-1]-self.dec[1:])**2) # large absolute change
            inds2 = np.nonzero(difs2 > dabs)[0]
        else:
            inds2 = np.empty(0,dtype='int')
        


        difs3 = np.abs(self.tel_flags[:-1]-self.tel_flags[1:]) # Direction change

        if save_dir is not None:
            # plt.scatter(self.ra)
            # plt.plot(self.tel_flags)
            plt.plot(self.ra, self.tel_flags)
            plt.xlabel('Apparent RA [deg]')
            plt.ylabel('tel flags')
            plt.savefig('difs3')
            print(difs3)
        
        inds3 = np.where(difs3 > .1)[0]

        # Points that meet one of these criteria are considered breaks 
        # in the scan pattern and the data before and after will be 
        # treated as separate scans
        inds = np.sort(np.unique(np.concatenate((inds1,inds2,inds3))))

        # Count which scan each everything is on
        scan_flags[:inds[0]+1] = 1


        for i in range(1,len(inds)-1):
            scan_flags[inds[i]+1:inds[i+1]+1] = i
        scan_flags[inds[-1]:] = i

        # Remove scans that where the telescope flag
        # indicates turnarounds, not tracking, or moving to source
        if remove_tel_flags:
            scan_flags[np.isin(self.tel_flags,[0,1,4])] = 0

        # Find places where a datapoints's scan number isn't equal to either 
        # adjacent data point and declare them not scans
        scan_flags[1:-1][(scan_flags[1:-1] != scan_flags[:-2]) & (scan_flags[1:-1] != scan_flags[2:])] = 0

        # Renumber in case anything that  wasn't actually a scan go numbered
        scans = np.sort(np.unique(scan_flags))
        for i in range(len(scans)):
            scan_flags[scan_flags == scans[i]] = i
        if 0 not in scans:
            scan_flags += 1

        # Determine which way the scan is moving and set a flag for that
        if scan_direction == 'ra':
            difs4 = (self.ra[:-1]-self.ra[1:])
        elif scan_direction == 'dec':
            difs4 = (self.dec[:-1]-self.dec[1:])
        scan_direction_flags[1:][difs4<=0] = 1

        self.scan_flags = scan_flags
        self.scan_direction_flags = scan_direction_flags

        self.header['flags']['scan_flags_initialized'] = True


    def renumber_scans(self):
        """Reset the indexing of the scans so that they run continuously from 1 to n
        
        In the process of making data cuts, some originally identified
        scans may be removed from the timestream altogether, resulting 
        in gaps in the scan numbering. This function renumbers the scans
        so the numbering is continuous starting at 1. The 0 scan index 
        is still reserved for samples that do not belong to a scan.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        scans = np.unique(self.scan_flags)
        for i in range(len(scans)):
            self.scan_flags[self.scan_flags == scans[i]] = i
        if 0 not in scans:
            self.scan_flags += 1


    def remove_end_scans(self, n_start=0, n_finish=0):
        """Remove a speciied number of scans from the start or end of an observation
        
        This method removes a specified number of scans taken at 
        the start or end of a timestream.

        Parameters
        ----------
        n_start : int
            Number of scans to remove at the beginning of the timestream
        n_finish : int
            Number of scans to remove at the end of the timestream

        Returns
        -------
        None
        """

        if not self.header['flags']['scan_flags_initialized']:
            raise ValueError("Scan identification has not been performed")
        
        u = np.sort(np.unique(self.scan_flags))
        if n_finish == 0:
            u = u[:n_start]
        else:
            u = np.concatenate([u[:n_start],u[-n_finish:]])
        
        self._remove_samples(np.nonzero(~np.isin(self.scan_flags, u))[0])


    def remove_scan_direction(self,keep_direction='positive'):
        """Remove all scans not along a specified scan direction

        This function drops all of the scans moving in the
        specified direction. It is useful for comparing 
        scans taken in oposite directions. Note that the
        function calls for specifying the direction to keep
        not the one to drop.

        Parameters
        ----------
        keep_direction : {'positive','negative'}, default='positive'
            The scan direction to keep - 'positive' keeps scans that
            are increasing along the scanned coordinate direction,
            'negative' keeps scans that are decreasing.
        
        Returns
        -------
        None
        """

        if not self.header['flags']['scan_flags_initialized']:
            raise ValueError("Scan identification has not been performed")
        
        if keep_direction == 'positive': # Increasing RA flag = 0
            val = [1]
        elif keep_direction == 'negative': # Decreasing RA flag = 1
            val = [0]
        else:
            raise ValueError("Invalid direction")

        self._remove_samples(np.nonzero(np.isin(self.scan_direction_flags, val))[0])


    def remove_scan_edge(self, n_start=0, n_finish=0):
        """Remove samples at the edges of scans

        This function removes a specified number of samples
        from the beginning and or end of each scan. This can
        be useful if artifacts tend to show up while the 
        telescope is accelerating during a turnaround.

        Parameters
        ----------
        n_start : int, optional
            Number of samples to remove from the start of each scan
        n_finish : int, optional
            Number of samples to remove from the end of each scan
        """

        if not self.header['flags']['scan_flags_initialized']:
            raise ValueError("Scan identification has not been performed")
        

        scan_list, n = np.unique(self.scan_flags, return_counts=True)
        # print(self.scan_flags, 'in remove scan edge')
        inds = []
        for u in scan_list:
            ind_scan = np.nonzero(self.scan_flags==u)[0]
            if n_finish == 0:
                inds.append(ind_scan[n_start:])
            else:
                inds.append(ind_scan[n_start:-n_finish])

        inds = np.concatenate(inds)

        self._remove_samples(inds)


    def remove_scan_flag(self):
        """Remove data segments that are not identified as part of a scan

        The ``Timestream.flag_scans()`` method identifies some samples as
        not belonging to a scan. This method removes those scans from the
        timestream.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if not self.header['flags']['scan_flags_initialized']:
            raise ValueError("Scan identification has not been performed")
        
        self._remove_samples(np.nonzero(self.scan_flags != 0)[0])


    def remove_short_scans(self, thresh=100):
        """Remove scans with fewer than a specified number of samples
        
        This method removes scans that contain fewer than a specified
        number of scans. These typically are associated with turnarounds
        and/or moving the telescope to the starting point of the scan
        pattern.

        Parameters
        ----------
        thresh : int
            The minimum number of samples in a scan. Scans shorter 
            than this will be removed. Default is 100.

        Returns
        -------
        None
        """

        if not self.header['flags']['scan_flags_initialized']:
            raise ValueError("Scan identification has not been performed")
        
        u, n = np.unique(self.scan_flags, return_counts=True)
        u = u[np.where(n >= thresh)]
        self._remove_samples(np.nonzero(np.isin(self.scan_flags, u))[0])


    def remove_tel_flag(self,flags=[0,1]):
        """Remove samples with specified values of ``Timestream.flag``
        
        This function removes samples where the ``Timestream.flag`` 
        attribute (the telescope direction information) is in a provided
        list of values.

        The flag values are 0 if telescope is not on source, 1 if telescope
        is tracking field center, 2 if telescope is tracking in the
        negative RA direction, 3 if telescope is tracking in the positive
        RA direction, 4 if telescope is turning around.

        Parameters
        ----------
        flags : list
            A list of one or more flag values to be filtered from the
            timestream

        Returns
        -------
        None
        """
    
        for i in flags:
            self._remove_samples(np.nonzero(self.tel_flags != i)[0])

    
    ###############################################
    #### FOR TRANSFORMING DATA IN VARIOUS WAYS ####
    ###############################################
    def _get_c1(self,coords,galactic=False):
        """Returns either RA or Galactic l"""
        if galactic:
            return coords.galactic.l.degree
        else:
            return coords.ra.degree
    def _get_c2(self,coords,galactic=False):
        """Returns either Dec or Galactic b"""
        if galactic:
            return coords.galactic.b.degree
        else:
            return coords.dec.degree

    def convert_coordinates(self,newframe='J2000',transform_copy=False):
        """Transform data between coordinate frames

        This function converts between apparent coordinates 
        (which is what the telescope reports), 'J2000' 
        coordinates (in the ICRS frame), and galactic coordinates.
        The current frame of the data is taken from the
        header, while the new frame is specified in `newframe`

        If ``newframe='Galactic'`` is specified, the `l` and 
        `b` coordinates will be stored as the ``Timestream.ra``
        and ``Timestream.dec`` attributes respectively.

        The header values ``Timestream.header['center_ra']``,
        ``Timestream.header['center_dec']``, and
        ``Timestream.header['epoch']`` and are also transformed.
        
        Parameters
        ----------
        newframe : {'J2000', 'Galactic', 'apparent'}
            The frame into which the coordinates should be 
            converted.
        transform_copy: bool, optional 
            If True, do the coordinate transform on the stored copy too (see above). 

        Returns
        -------
        None

        Notes
        -----
        Conversion between apparent and other coordinate systems
        requires a time. The time used is taken as the median of
        the timestream (prior to any data processing). It may be
        worth investigating whether this is an adequate approximation
        for long datasets.
        """

        if not self.header['flags']['has_tel_data']:
            raise ValueError("No telescope data available - coordinates unknown")
        
        location = EarthLocation.from_geodetic(lon=self.header['telescope_pars']['longitude']*u.deg, lat=self.header['telescope_pars']['latitude']*u.deg, height=self.header['telescope_pars']['height']*u.m)
        time = Time(self.header['med_time'],format='unix')
        
        # Frames we understand
        valid_frames = [{'name':'apparent',
                        'names':['apparent'],
                        'framein':TETE(obstime=time, location=location),
                        'frameout':TETE(obstime=time, location=location),
                        'galactic':False},
                        {'name':'J2000',
                        'names':['j2000','j2000.0','2000','2000.0'],
                        'framein':'icrs',
                        'frameout':'icrs',
                        'galactic':False},
                        {'name':'galactic',
                        'names':['galactic'],
                        'framein':'galactic',
                        'frameout':'icrs',
                        'galactic':True}
                        ]

        if transform_copy and not self.header['flags']['store_copy']:
            warnings.warn("convert_coordinates: No copy stored - transform_copy being ignored")
            transform_copy = False

        frame0 = None
        frame1 = None
        framef = None
        for frameinfo in valid_frames:
            if self.header['epoch'].lower() in frameinfo['names']:
                frame0 = frameinfo
            if self.header['command_epoch'].lower() in frameinfo['names']:
                frame1 = frameinfo
            if newframe.lower() in frameinfo['names']:
                framef = frameinfo
        if frame0 is None:
            raise ValueError('Frame in header information not compatible with coordinate transforms')
        if framef is None:
            raise ValueError('newframe unrecognized')

        # Transform coords:
        coords = SkyCoord(self.ra*u.degree,self.dec*u.degree,frame=frame0['framein'])
        coords = coords.transform_to(framef['frameout'])

        # Transform copy:
        if transform_copy:
            copy_coords = SkyCoord(self.ra_copy*u.degree,self.dec_copy*u.degree,frame=frame0['framein'])
            copy_coords = copy_coords.transform_to(framef['frameout'])

        # Transform center
        c_coords = SkyCoord(self.header['center_ra']*u.degree,self.header['center_dec']*u.degree,frame=frame0['framein'])
        c_coords = c_coords.transform_to(framef['frameout'])                
        if frame1 is not None:
            command_coords = SkyCoord(self.header['command_ra']*u.degree,self.header['command_dec']*u.degree,frame=frame1['framein'])
            command_coords = command_coords.transform_to(framef['frameout'])                

        # Transform offsets
        if self.header['flags']['has_feed_offsets']:
            off_coords = SkyCoord((self.header['feed_offsets'][:,0]+self.header['center_ra'])*u.degree,(self.header['feed_offsets'][:,1]+self.header['center_dec'])*u.degree,frame=frame0['framein'])
            off_coords.transform_to(framef['frameout'])                

        self.ra = np.array(self._get_c1(coords,framef['galactic']))
        self.dec = np.array(self._get_c2(coords,framef['galactic']))
        if transform_copy:
            self.ra_copy = np.array(self._get_c1(copy_coords,framef['galactic']))
            self.dec_copy = np.array(self._get_c2(copy_coords,framef['galactic']))

        self.header['center_ra'] = self._get_c1(c_coords,framef['galactic'])
        self.header['center_dec'] = self._get_c2(c_coords,framef['galactic'])
        if frame1 is not None:
            self.header['command_ra'] = self._get_c1(command_coords,framef['galactic'])
            self.header['command_dec'] = self._get_c2(command_coords,framef['galactic'])
            self.header['command_epoch'] = framef['name']
        
        if self.header['flags']['has_feed_offsets']:
            self.header['feed_offsets'][:,0] = self._get_c1(off_coords,framef['galactic']) - self.header['center_ra']
            self.header['feed_offsets'][:,1] = self._get_c2(off_coords,framef['galactic']) - self.header['center_dec']

        self.header['epoch'] = framef['name']



    def filter_scan_det(self, c1, c2, n=1, exclude_max=True, max_sep=.006, nscans=0, det_coord_mode='xf', save_dir=None):
        """Polynomial filtering of detector counts

        For filtering all detectors in a timestream file see
        ``Timestream.filter_scan`` which wraps this method
        and provides some additional options.

        This method provides a rudimentary way to filter
        atmospheric fluctuations and/or detector drifts from
        a timestream. It iterates through each scan for a 
        specified detector in order to fit and remove
        a polynomial from each one (If scans have not been
        identified using ``Timestream.flag_scans`` all data
        is treated as a single scan).

        By default, the filtering will attempt to identify the
        peak in the timestream and mask a region around it which
        is excluded from the fits. This is meant to a bright
        source in the map from biasing the polynomial fits.
        Note that this is only likely to work well for compact 
        sources, and even then is only a rough tool. More 
        careful masking is possible with the ``Timestream.filter_scandet_mask()``
        method.

        Parameters
        ----------
        c1, c2 : int
            The coordinates in either xf (default) or muxcr space
            of the detector to process. The coordinates system to 
            use is controlled by `det_coord_mode`.
        n : int, default=1
            The order of polynomial to fit to each scan.
        exclude_max : bool, default=True
            If set to True, the code will first generate a preliminary
            filtered timestream, then attempt to identify the peak
            in this timestream. It will then re-do the fitting 
            excluding a region (controlled by `max_sep`) around in 
            order to prevent bright point sources from biasing the
            filtering results.
        max_sep : float, optional
            The radius around the flux peak to mask when 
            ``exclude_max=True`` is set. Should be specified in
            degrees. Defaults to 0.01 degrees (36 arcseconds).
        nscans : int, optional
            If set this will include `nscans` additional scans
            before and `nscans` additional scans after each scan
            when doing the fit - this allows fits to be performed
            using larger chunks of the timestream.
        det_coord_mode : {'xf','cr'}, default='xf'
            Coordinate system used to identify detectors.
            'xf' (default) or 'cr' are accepted.
        
        Returns
        -------
        None
        """
        if not self.header['flags']['scan_flags_initialized']:
            warnings.warn("Scans have not been identified yet, all data being treated as a single scan")
        
        index = self._get_coord(c1,c2,det_coord_mode)
        if exclude_max:
            temp = np.copy(self.data[index])
            for i in np.unique(self.scan_flags):
                fit_inds = np.where(np.isin(self.scan_flags,[i]+[i+j+1 for j in range(nscans)]+[i-j-1 for j in range(nscans)]))[0]
                clean_inds = np.where(self.scan_flags==i)[0]
                t_mid = np.median(self.t[clean_inds])
                fit = np.polyfit(self.t[fit_inds]-t_mid, self.data[index,fit_inds], deg=n)
                temp[clean_inds] -= np.polyval(fit,self.t[clean_inds]-t_mid)

            decpeak = self.dec[np.argmax(np.abs(temp))]
            rapeak = self.ra[np.argmax(np.abs(temp))]
            r2 = ((self.ra-rapeak)*np.cos(decpeak*np.pi/180))**2 + (self.dec-decpeak)**2

        temp = np.copy(self.data[index])

        # if save_dir is not None:
        #     fig, axs = plt.subplots(2,2, figsize=(12,12))

        for i in np.unique(self.scan_flags):
            clean_inds = np.where(self.scan_flags==i)[0]
            if exclude_max:
                fit_inds = np.where((np.isin(self.scan_flags,[i]+[i+j+1 for j in range(nscans)]+[i-j-1 for j in range(nscans)])) & (r2 > max_sep**2))[0]
            else:
                fit_inds = np.where(np.isin(self.scan_flags,[i]+[i+j+1 for j in range(nscans)]+[i-j-1 for j in range(nscans)]))[0]
            t_mid = np.median(self.t[clean_inds])

            fit = np.polyfit(self.t[fit_inds]-t_mid, temp[fit_inds], deg=n)

            # if save_dir is not None:
            #     fig.suptitle('x%sf%s' % (c1,c2))
            #     if self.header['scan_pars']['direction'] == 'DEC':
            #         axs[0,0].set_title('Fit Inds')
            #         axs[0,0].scatter(self.dec[fit_inds], self.data[index,fit_inds])#,# 'k-')

            #         axs[1,0].set_title('All Inds')
            #         axs[1,0].plot(self.dec[clean_inds], self.data[index,clean_inds], 'k-')

            #     elif self.header['scan_pars']['direction'] == 'RA':
            #         axs[0,0].set_title('Fit Inds')
            #         axs[0,0].scatter(self.ra[fit_inds], self.data[index,fit_inds])#,# 'k-')

            #         axs[1,0].set_title('All Inds')
            #         axs[1,0].plot(self.ra[clean_inds], self.data[index,clean_inds], 'k-')

            self.data[index,clean_inds] -= np.polyval(fit,self.t[clean_inds]-t_mid)


        #     if save_dir is not None:
        #         if self.header['scan_pars']['direction'] == 'DEC':
        #             axs[0,1].set_title('All Inds - Poly Fit')
        #             axs[0,1].plot(self.dec[clean_inds], self.data[index,clean_inds], 'k-')

        #             axs[1,1].set_title('Poly Fit')
        #             axs[1,1].plot(self.dec[clean_inds], np.polyval(fit,self.t[clean_inds]-t_mid), 'k-')
        #         elif self.header['scan_pars']['direction'] == 'RA':
        #             axs[0,1].set_title('All Inds - Poly Fit')
        #             axs[0,1].plot(self.ra[clean_inds], self.data[index,clean_inds], 'k-')

        #             axs[1,1].set_title('Poly Fit')
        #             axs[1,1].plot(self.ra[clean_inds], np.polyval(fit,self.t[clean_inds]-t_mid), 'k-')
        #             axs[0,0].set_xlabel('Right Ascension [deg]')
        #             axs[1,0].set_xlabel('Right Ascension [deg]')
        #             axs[0,1].set_xlabel('Right Ascension [deg]')
        #             axs[1,1].set_xlabel('Right Ascension [deg]')
        # print(save_dir + 'unmasked_x%s_f%s_p%s.png' % (c1,c2,self.mc))

        # fig.savefig(save_dir + 'unmasked_x%s_f%s_p%s.png' % (c1,c2,self.mc))
        # plt.close(fig)
        # plt.close('all')

    def filter_scan_det_mask(self, c1, c2, mask_ra_bins, mask_dec_bins, mask, n=1, det_coord_mode='xf',use_offsets=False, save_dir=None):

        """Polynomial filtering of detector counts with source masking

        For filtering all detectors in a timestream file see
        ``Timestream.filter_scan`` which wraps this method
        and provides some additional options.

        This method provides a rudimentary way to filter
        atmospheric fluctuations and/or detector drifts from
        a timestream. It iterates through each scan for a 
        specified detector in order to fit and remove
        a polynomial from each one (If scans have not been
        identified using ``Timestream.flag_scans`` all data
        is treated as a single scan).

        It allows for the specification of a masked region to 
        exclude from the polynomial fitting so that bright 
        sources to not bias the fits. The mask is specified 
        in terms of a 2D array - `mask_ra_bins` and 
        `mask_dec_bins` given the R.A. and declination of
        the edges of this array, while `mask` is a boolean
        array which specifies which cells of the grid are to 
        be masked (regions to be excluded should be ``True``).
        The size and resolution of the mask can be arbitrary.
        Regions outside the defined mask will be assumed to 
        need no masking. The coordinates system of the mask
        should match that of the data (which is in apparent
        coordinates by default but can be transformed to other
        coordinate systems using ``Timestream.convert_coordinates()``).

        Parameters
        ----------
        c1, c2 : int
            The coordinates in either xf (default) or muxcr space
            of the detector to process. The coordinates system to 
            use is controlled by `det_coord_mode`.
        n : int, default=1
            The order of polynomial to fit to each scan.
        max_sep : float, default=0.01
            Only used when `mode` is set to ``'max_mask'``
            or ``'circular_mask'``. The radius around the flux 
            peak to mask. Should be specified in degrees. 
            Defaults to 0.01 degrees (36 arcseconds).
        nscans : int, default=0
            
        ra_peak, dec_peak : float, optional
            Only used when `mode` is set to ``'circular_mask'``
            or ``'square_mask'``. This defines the central RA/dec
            for the mask in Degrees.
        angular_size : float, default=10
            The size of the mask in arcseconds will be 
            (angular_size) x (angular_size).
        mask_ra_bins, mask_dec_bins : array_like
            1D arrays containing the coordinates at the edges of 
            each cell of the mask array. These should be 
            specified in the same coordinate system as the 
            timestream RA and declination data are currently
            represented (apparent coordinates by default).
        mask : ndarray
            Boolean 2D array, where regions to be masked are set
            to True and regions not masked are False. The shape
            of the array should be ``(len(mask_ra_bins)-1,len(mask_dec_bins)-1)``
            as `mask_ra_bins` and `mask_dec_bins` specify the corners
            of each cell in the mask.
        det_coord_mode : {'xf','cr'}, default='xf'
            Coordinate system used to identify detectors.
            'xf' (default) or 'cr' are accepted.
        
        Returns
        -------
        None
        """
        
        if use_offsets and not self.header['flags']['has_feed_offsets']:
            raise ValueError('No offsets specified, cannot apply corrections')

        index = self._get_coord(c1,c2,det_coord_mode)

        if use_offsets:
            ra, dec = self.offset_pos(self.header['xf_coords'][index][0])
        else:
            ra = self.ra
            dec = self.dec

        ra_mask_bin_inds = np.digitize(ra,mask_ra_bins)
        dec_mask_bin_inds = np.digitize(dec,mask_dec_bins)
        big_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2))
        big_mask[1:-1,1:-1] = mask
        mask_flag = big_mask[tuple(dec_mask_bin_inds),tuple(ra_mask_bin_inds)]

        # if save_dir is not None:
        #     fig, axs = plt.subplots(2,2, figsize=(12,12))

        for i in np.unique(self.scan_flags):

            inds = np.nonzero(self.scan_flags==i)[0]
            t_mid = np.median(self.t[inds])
            fit_inds = np.nonzero((self.scan_flags==i) & (mask_flag==0))[0]

            if len(fit_inds) < 1:
                p = np.polyfit([3,3,3],[3,3,3], deg=n)
                fit = np.polyval(p, self.t[inds]-t_mid)
                fit[:] = np.nan 
                fit_std = fit.copy()
                warnings.warn('No fitting inds found for this scan, returning nan as best fit poly amplitudes')
            else:
                p, cov = np.polyfit(self.t[fit_inds]-t_mid, self.data[index,fit_inds], deg=n, cov=True)
                fit = np.polyval(p, self.t[inds]-t_mid)
                n = len(p) - 1
                X = np.vstack([(self.t[inds]-t_mid)**i for i in range(n, -1, -1)]).T  # [t^n, ..., 1]
                # print(X.shape, cov.shape)
                y_var = np.sum(X @ cov * X, axis=1)
                fit_std = np.sqrt(y_var)

            # print(ra[fit_inds], dec[fit_inds])

            # if save_dir is not None:
            #     axs[0,0].set_title('Fit Inds')
            #     axs[0,0].scatter(ra[fit_inds], self.data[index,fit_inds])#,# 'k-')

            #     axs[1,0].set_title('All Inds')
            #     axs[1,0].plot(ra[inds], self.data[index,inds], 'k-')

            self.data[index,inds] -= fit

            ### This doesn't belong here, but I can't figure out where it goes without breaking some of Ben's code, 
            ### so just making it check for self.scan_stds before running
            if hasattr(self,'scan_stds'):
                if self.header['flags']['has_gains']:
                    ind_scan_std = np.sqrt(np.nanstd(self.data[index,fit_inds])**2)
                    self.scan_stds[index,inds] = np.sqrt(ind_scan_std + self.data[index,inds]**2 / self.header['gains'][index]**2 * self.header['gains_e'][index]**2)
                else:
                    self.scan_stds[index,inds] = np.sqrt( np.nanstd(self.data[index,fit_inds])**2 )
                    self.header['gains_e'] = np.zeros((16,60)); self.header['gains'] = np.zeros((16,60))
                    gains = np.array([self.header['gains'][xf[0],xf[1]] for xf in self.header['xf_coords']])
                    gains_e = np.array([self.header['gains_e'][xf[0],xf[1]] for xf in self.header['xf_coords']])
                    self.header['gains'] = gains; self.header['gains_e'] = gains_e
            
        #     if save_dir is not None:
        #         axs[0,1].set_title('All Inds - Poly Fit')
        #         axs[0,1].plot(ra[inds], self.data[index,inds], 'k-')

        #         axs[1,1].set_title('Poly Fit')
        #         axs[1,1].plot(ra[inds], fit, 'k-')

        # fig.savefig(save_dir + 'x%s_f%s_p%s' % (c1,c2,self.mc))
        # plt.close(fig)

    def filter_scan(self, mode='max_mask',max_sep=0.01, n=1,nscans=0,mask_ra_bins=None, mask_dec_bins=None, mask=None, use_offsets=False, make_stds=False, save_dir=None, _warn_extra_filter=True):
        
        """Polynomial filtering of all detectors

        This method provides a rudimentary way to filter
        atmospheric fluctuations and/or detector drifts from
        a timestream. For a given detector, it iterates through 
        each scan in order to fit and remove a polynomial from 
        each one (If scans have not been identified using 
        ``Timestream.flag_scans`` all data is treated as a single 
        scan). It then loops through all detectors and repeats
        this filtering.

        Multiple modes are available for masking spatial regions
        covered by the data, to avoid bright sources affecting
        the fitting results. These are conrolled by the `mode`
        parameter. ``mode='no_mask'`` will not do any masking.
        ``mode='max_mask'`` will attempt to find the brightest 
        region in the data and produce a circular mask around
        this source. ``mode='circular_mask'`` will mask the region 
        within max_sep centered around a given peak RA and dec. 
        ``mode='square_mask'`` will mask a square region around 
        a given peak RA and dec and it will be specificed futher 
        using mask_size and angular_size. ``mode='mask'`` 
        allows the specification of exact regions in R.A. and 
        declination using a coordinate grid.

        Parameters
        ----------
        mode : {'no_mask','max_mask', 'circular_mask', 
                'square_mask', 'mask'}, default='max_mask'
            Determines what, if any, masking to apply to the
            timestream in order to exclude regions with bright
            sources from the polynomial fitting.
        n : int, default=1
            The order of polynomial to fit to each scan.
        max_sep : float, default=0.01
            Used when masking around the peak flux, gives the radial size in degrees
        nscans : int, default=0
        mask_ra_bins, mask_dec_bins : array_like
            Only used when `mode` is set to ``'mask'``.
            1D arrays containing the coordinates at the edges of 
            each cell of the mask array. These should be 
            specified in the same coordinate system as the 
            timestream RA and declination data are currently
            represented (apparent coordinates by default).
        mask : ndarray
            Only used when `mode` is set to ``'mask'``.
            Boolean array of either 2, 3, or 4 dimensions.
            If ``mask.ndim==2`` the array should be of shape 
            ``(nra,ndec)`` where ``nra=len(mask_ra_bins)-1``,
            ``ndec=len(mask_dec_bins)-1`` are the dimensions 
            of the mask. Regions that are masked should be
            set to True and regions not masked are False. 
            Because the different feedhorns see different regions
            of the sky for the same nominal R.A. and declination
            of the telescope it can be useful to specify 
            different masks for each feedhorn. If ``mask.ndim==2``
            the dimensions of the mask should be ``(16,nra,ndec)``
            where the first dimension corresponds to the feed
            number, and detectors from the first feedhorn will
            be masked by the mask in mask[0], detectors from 
            the second feedhorn will be masked using the mask in
            mask[1], and so on. This can be extended to defining
            a unique mask for each detector, in which case the 
            shape should be ``(16,60,nra,ndec)`` and the first
            two indices match the xf coordinates of the detectors.

        
        Returns
        -------
        None

        Notes
        -----

        ``mode='max_mask'`` will attempt to identify the
        peak in the timestream and mask a region around it which
        is excluded from the fits. This is meant to a bright
        source in the map from biasing the polynomial fits.
        Note that this is only likely to work well for compact 
        sources, and even then is only a rough tool. Works well
        specifically for planets when getting offsets in the 
        calibration stage.
        
        ``mode='circular_mask'`` essentailly takes the methods
        used in ``mode='max_mask'`` except using a defined 
        central RA and dec and just applies them to data that 
        has already been treated with offsets.
        
        ``mode='square_mask'`` defines a square mask cetnered on 
        ra_peak and dec_peak, the size of the mask is determined 
        by the angular_size given. The size of the mask will be
        (angular_size) by (angular_size).
        
        ``mode='mask'`` allows for the specification of a 
        masked region to exclude from the polynomial fitting 
        so that bright sources to not bias the fits. The mask i
        s specified in terms of a 2D array - `mask_ra_bins` and 
        `mask_dec_bins` give a the R.A. and declination of
        the edges of this array, while `mask` is a boolean
        array which specifies which cells of the grid are to 
        be masked (regions to be excluded should be ``True``).
        The size and resolution of the mask can be arbitrary.
        Regions outside the defined mask will be assumed to 
        need no masking. The coordinates system of the mask
        should match that of the data (which is in apparent
        coordinates by default but can be transformed to other
        coordinate systems using ``Timestream.convert_coordinates()``).
        """
        
        if make_stds:
            self.scan_stds = np.zeros((self.data.shape)) ### data holder 

        if mode not in ['no_mask','max_mask', 'circular_mask', 'square_mask', 'mask']:
            raise ValueError("Filtering mode not recognized")
        if self.header['flags']['filtering_applied'] and _warn_extra_filter:
            warnings.warn("Filtering already applied, this filter will be applied on top of previous results")
        
        # Iterate through detectors and channels
        for deti,xf in enumerate(self.header['xf_coords']):
            x = xf[0]
            f = xf[1]
            if mode == 'no_mask':
                self.filter_scan_det(x,f,n,exclude_max=False,max_sep=max_sep,nscans=nscans, save_dir=save_dir)
            if mode == 'max_mask':
                self.filter_scan_det(x,f,n,exclude_max=True,max_sep=max_sep,nscans=nscans, save_dir=save_dir)

            if mode == 'mask':
                if mask is None or mask_ra_bins is None or mask_dec_bins is None:
                    raise ValueError("Mask parameters not given for filter type 'mask'")
                else:
                    if mask.ndim == 4:
                        self.filter_scan_det_mask(x, f, n=n, mask_ra_bins=mask_ra_bins, mask_dec_bins=mask_dec_bins, mask=mask[x,f], use_offsets=use_offsets, save_dir=save_dir)
                    elif mask.ndim == 3:
                        self.filter_scan_det_mask(x, f, n=n, mask_ra_bins=mask_ra_bins, mask_dec_bins=mask_dec_bins, mask=mask[deti], use_offsets=use_offsets, save_dir=save_dir)
                    else:
                        self.filter_scan_det_mask(x, f, n=n, mask_ra_bins=mask_ra_bins, mask_dec_bins=mask_dec_bins, mask=mask, use_offsets=use_offsets, save_dir=save_dir)

        if mode == 'no_mask':
            self.header['filter_type'] = '{} degree poly'.format(n)
        if mode == 'max_mask':
            self.header['filter_type'] = '{} degree poly + max mask'.format(n)
        if mode == 'mask':
            self.header['filter_type'] = '{} degree poly + mask'.format(n)
        self.header['flags']['filtering_applied'] = True
        
    def offset_pos(self, feed, raise_nan=True):
        """Apply the appropriate positional offsets for a specified
        feedhorn to the self.ra and self.dec timestream arrays and
        return the correct position arrays.

        This uses the feedhorn offsets for the timestream to
        determine the R.A. and declination of a specified 
        feedhorn from the telescope R.A. and declination data
        stored in self.ra and self.dec. It returns these positions
        to the user.

        In order to use this method the feedhorn offsets must be
        specified. This can be done by passing at ``timesoft.calibration.Offsets``
        object to ``Timestream.set_offsets``

        Parameters
        ----------
        feed : int (0 to 15)
            The feedhorn number to get positions for.
        raise_nan : bool, default=True
            If the feed offsets for a given detector are nans
            (ie the ``Offsets`` object does not contain valid
            measurements for those feeds), then an error will
            be raised. If this is set to false, then the telescope
            position, uncorrected for offsets, will be returned
            instead.

        Returns
        -------
        ra_offset : ndarray
            The R.A. of the feed at each sample in the timestream,
            accounting for the offset between the telescope pointing
            and the feedhorn pointing
        dec_offset : ndarray
            The declination of the feed at each sample in the timestream,
            accounting for the offset between the telescope pointing
            and the feedhorn pointing
        """
        if not self.header['flags']['has_feed_offsets']:
            raise ValueError('No offsets specified, cannot apply corrections')

        xoff,yoff = self.get_feed_offsets(feed,det_coord_mode='x',raise_nan=raise_nan)

        # print('Getting Offsets')
        # fig, axs = plt.subplots(1, figsize=(6,6))
        # axs.plot(xoff,yoff)
        # axs.set_xlabel('XOFF')
        # axs.set_ylabel('YOFF')
        # fig.savefig('OFFSETSINTIMESTREAM')
        # # exit()
        # exit()
        dec_offset = self.dec-yoff
        ra_offset = self.ra-xoff/np.cos((dec_offset)*np.pi/180)

        return ra_offset, dec_offset


    #########################
    #### FOR CALIBRATION ####
    #########################
    def set_tau(self,tau,overwrite=False):
        """Set the atmospheric opacity for the timestream

        Older datasets do not have tau stored with them. 
        This function makes it possible to add an array of
        tau values to the timestream. tau can be specified as
        a single value, or as an array of the same length
        as the timestream (try interpolating it to the 
        Timestream.t array if the sampling differs). 

        Parameters
        ----------
        tau : float or array-like
            The tau_0 values at 225 GHz to be added to the 
            timestream. Should either be a single value or
            matched in length to the other timestream variables
        overwrite : bool, default=False
            If True, the current values of tau will be overwritten
            even if they are believed to be valid data. This
            defaults to False to prevent overwriting good data by 
            accident.
        """

        if self.header['flags']['has_tau_data'] and not overwrite:
            raise ValueError("Timestream already has tau data, use overwrite=True to overwrite.")

        if np.isscalar(tau):
            tau = np.ones(len(self.t)) * tau
        elif len(tau) != len(self.t):
            raise ValueError("Length of tau must match length of timestream")
        else:
            tau = np.array(tau)

        self.tau = tau
        self.header['flags']['has_tau_data'] = True


    def correct_tau(self,atm_function,reverse=False):
        """Apply correction for atmospheric extinction

        Use the tau values stored in ``Timestream.tau`` 
        to determine the extinction correction and apply
        it to each detector. ``atm_function`` should be 
        a function which takes three arguments: x - the
        x coordinate of a detector, f - the f coordinate
        of a detector, and tau_0 - a value of tau_0 at 225
        GHz (these are the values stored in ``Timestream.tau``),
        and return the tau_0 value at the frequency of 
        the detector. 

        Parameters
        ----------
        atm_function : function
            The function should take three arguments: detector
            x, detector f, and an array of tau_0,225 values
            and return an array of tau_0 values for the frequency 
            of the specified detector.
        reverse : bool, default=False
            If True, the atmospheric correction will be reversed.
        
        Notes
        -----

        This method is so that an arbitrary function for
        converting tau_225 to tau_detector can be specified
        because a) multiple atmospheric models are possible
        and it may be valuable to be able to try more than one,
        b) each detector can in theory have its own value
        of tau because of different effective frequency, so
        having a flexible way to handle this is useful.

        I have implemented a simple atmospheric model which
        consists of a generic AM model for the elevation of
        Kitt Peak (courtesy of Dan Marrone), convolved with
        a Gaussian kernel of FWHM = 2 GHz at the nominal
        frequency of each frequency channel. This can be accessed
        by importing ``timesoft.calibration.setup_simple_atm_model``
        and running ``atm_function = timesoft.calibration.setup_simple_atm_model()``.
        ``atm_function`` can then be passed to ``Timestream.correct_tau``.

        It may be advisable to implement more sophisticaded models
        once we have better characterized the frequency response
        of the detectors and behavior of the atmosphere.
        """

        if not self.header['flags']['has_tau_data']:
            raise ValueError("No tau data available for correction")

        if not reverse:
            if self.header['flags']['corrected_tau']:
                raise ValueError("Atmosphere correction already applied")
        if reverse:
            if not self.header['flags']['corrected_tau']:
                raise ValueError("No tau correction to remove")

        # Note sec(z) is not valid bellow about 30 degrees, so we may want
        # to implement something smarter
        if np.any(self.el < 25):
            warnings.warn("Airmass computations using young 1994 model")
            def am_func(z):
                z_rad = np.deg2rad(z)
                c = np.cos(z_rad)
                numerator = 1.002432 * (c**2) + 0.148386 * c + 0.0096467
                denominator = (c**3) + 0.149864 * (c**2) + 0.0102963 * c + 0.000303978
                return numerator / denominator
        else: 
            def am_func(z):
                return 1 / np.cos(np.pi/180*(z))

        # airmass = 1/np.cos(np.pi/180*(90-self.el))
        airmass = am_func(90 - self.el)
        for xf in self.header['xf_coords']:
            tau_det = atm_function(xf[0],xf[1],self.tau)
            if not reverse:
                self.data[self.get_xf(*xf)] *= np.exp(tau_det * airmass)
            if reverse:
                self.data[self.get_xf(*xf)] /= np.exp(tau_det * airmass)

        if not reverse:
            self.header['flags']['corrected_tau'] = True
        if reverse:
            self.header['flags']['corrected_tau'] = False


    def correct_gains(self,gains=None,gains_e=None,check_dets=True,drop_missing_dets=True,overwrite=False,reverse=False):
        """Apply counts to intensity conversion

        Convert from counts to units of intensity by applying
        the measured gains to the ``Timestream.data`` array.
        If the gains have already been specified using 
        ``Timestream.set_gains``, then no arguments are needed.
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
            then it will be used to initialize the gains for the ``Timestream``
            instance and the remaining parameters are passed to 
            ``Timestream.set_gains``.
        reverse : bool, default=False
            If True, the gains correction will be reversed.
        """

        if gains is None and not self.header['flags']['has_gains']:
            raise ValueError("No Gain values specified")
        elif gains is not None:
            self.set_gains(gains=gains,gains_e=gains_e,check_dets=check_dets,drop_missing_dets=drop_missing_dets,overwrite=overwrite)
        
        if not reverse:
            if self.header['flags']['corrected_gains']:
                raise ValueError("Gain correction already applied")
            self.data *= self.header['gains'].reshape(-1,1)
            self.header['flags']['corrected_gains'] = True
        if reverse:
            if not self.header['flags']['corrected_gains']:
                raise ValueError("No gain correction to remove")
            self.data /= self.header['gains'].reshape(-1,1)
            self.header['flags']['corrected_gains'] = False


    def correct_time_constants(self,time_constants=None,check_dets=True,drop_missing_dets=True,overwrite=False):
        """Deconvolve time constants to correc slow detectors

        This function is not implemented - need to figure out
        how to do the deconvolution.

        Parameters
        ----------
        time_constants : ``timesoft.calibration.DetectorConstants`` instance, optional
            An object containing time constant measurements for a subset of the
            detectors. If constant for the timesream have already been 
            set then this argument is not necessary. If it is specified
            then it will be used to initialize the time constants for the ``Timestream``
            instance and the remaining parameters are passed to 
            ``Timestream.set_time_constants``.
        """

        if self.header['flags']['corrected_time_constants']:
            raise ValueError("Time constant correction already applied")

        if time_constants is None and not self.header['flags']['has_time_constants']:
            raise ValueError("No time constant values specified")
        elif time_constants is not None:
            self.set_time_constants(time_constants=time_constants,check_dets=check_dets,drop_missing_dets=drop_missing_dets,overwrite=overwrite)

        # Need to fix this code - if you know how, be my guest
        raise ValueError("This feature doesn't work - someone needs to get the deconvolution to look right")

        time_step = np.median(np.gradient(self.t))
        if np.any(np.unique(np.abs(np.gradient(self.t)-time_step)/time_step) > 0.05):
            raise ValueError("Data is not spaced consistently in time")
        l = 20*np.max(self.header['time_constants'])

        for i in range(self.header['n_detectors']):
            tc = self.header['time_constants'][i]
            impulse = np.exp(-(self.t-self.t[0])/tc)
            impulse = impulse[self.t-self.t[0] < l] # Shorten the impulse function
            impulse /= np.sum(impulse)
            result,_ = deconvolve(self.data[i],impulse)
            self.data[i] = 0
            self.data[i][:len(result)] = result
        
        self.header['flags']['corrected_time_constants'] = True


    #######################################
    #### FOR MAPMAKING (KARTO-GRAPHY?) ####
    #######################################
    def make_map(self, pixel=1./60./2.,correct_xpix=True,coordinate_system=None,dims=None,center=None,use_offsets=False,use_pointing_center=True):
        """Make maps of all detectors
        
        This method maps all detectors simultaneously.
        Maps are constructed by inserting each sample for
        a given detector into a grid and then averaging all 
        samples in a given grid cell. By default the map dimensions
        are guessed from the range of R.A. and declination 
        (or Galactic l and b, or azimuth and elevation)
        in the timestream, but they can optionally
        be specified.

        Once created, maps are stored in the ``Timestream.Maps``
        attribute, which is in turn an instance of the 
        ``timesoft.Maps`` class. They can be interacted 
        with following the documentation for that class,
        as well as a few convenience functions defined for
        the ``Timestream`` class itself.

        Parameters
        ----------
        pixel : float or list
            The size of each pixel specified in degrees. A single
            number will result in square pixels. Two values can
            be given in a list to specify the size of the pixels 
            in R.A. and declination separately.
        correct_xpix : bool, default=True
            Away from 0 declination/elevation, changing an object's
            coordinates by 1 degree in R.A./azimuth is not equivalent
            to moving the object by 1 degree (instead it is a move 
            of 1 degree * cos(declination/elevation)). By default, the
            pixel size is assumed to be specified such that it is the
            same at any point on the sky, thus when binning the 
            x-coordinate of the data (R.A. or azimuth), the bins 
            used are spaced by pixel / cos(y coordinate)). To turn
            this behavior off set this parameter to False. This
            will cause the bins to be defined directly by the pixel
            parameter with no correction.
        coordinate_system : {'ra-dec','az-el','l-b'}, optional
            The coordinates to map in. By default either 
            'ra-dec' or 'l-b' will be used, depending on what
            coordinate system the timestream data is currently 
            represented in.
        dims : array_like, optional
            Two element array giving the dimensions of the map 
            in degrees. If not specified the dimensions will be
            determined automatically. If specified, a center for
            the map must also be given.
        center : array_like, optional
            Two element array giving the location of the map 
            center in degrees. If not specified, it will be
            determined automatically.
        use_offsets : bool, default=False
            If True, apply offset corrections to the sample positions
            before mapping.
        use_pointing_center : bool, default=True
            If True, the center passed to the Map object will 
            be the commanded position of the observation (otherwise it
            will be the center of the map grid)

        Returns
        -------
        None

        Notes
        -----
        **To Do**        
        Add support for weighted averages
        """

        if self.header['scan_pars']['type'] in ['1d','1D']:
            raise ValueError('Timestream is for a 1d dataset, mapping not available')

        if use_offsets and not self.header['flags']['has_feed_offsets']:
            raise ValueError('No offsets specified, cannot apply corrections')

        # Check coordinate system setup
        if coordinate_system is None:
            if self.header['epoch'] not in ['Galactic','galactic']:
                coordinate_system = 'ra-dec'
            else:
                coordinate_system = 'l-b'
        
        if coordinate_system in ['l-b','ra-dec']:
            if coordinate_system == 'ra-dec' and self.header['epoch'] in ['Galactic','galactic']:
                warnings.warn("Timestream is currently represented in 'l-b' coordinates, maps will appear in these coordinates, not 'ra-dec'")
                coordinate_system == 'l-b'
            if coordinate_system == 'l-b' and self.header['epoch'] not in ['Galactic','galactic']:
                warnings.warn("Timestream is currently represented in 'ra-dec' coordinates, maps will appear in these coordinates, not 'l-b'")
                coordinate_system == 'ra-dec'
        elif coordinate_system == 'az-el':
            if use_offsets:
                raise ValueError("Offsets not implemented for az-el maps")
        else:
            raise ValueError("Coordinate system not be recognized")

        # Check dimension inputs
        if dims is not None or center is not None:
            if dims is None or center is None:
                raise ValueError("Must specify both center and dimensions, or neither")

        # Set up pixels
        if len(np.array(pixel,ndmin=1)) > 1:
            x_pixel = pixel[0]
            y_pixel = pixel[1]
        else:
            x_pixel = pixel
            y_pixel = pixel

        # Implementation for mapping all detectors simultaneously
        if not use_offsets:
            if dims is not None:
                fit_dims = False
            else:
                fit_dims = True

            # need an array of the positions for each datapoint
            if coordinate_system is None:
                if self.header['epoch'] not in ['Galactic','galactic']:
                    coordinate_system = 'ra-dec'
                else:
                    coordinate_system = 'l-b'
            
            if coordinate_system in ['l-b','ra-dec']:
                pos = np.array([self.ra,self.dec]).T
                yc = np.median(self.dec)
            elif coordinate_system == 'az-el':
                pos = np.array([self.az,self.el]).T
                yc = np.median(self.el)
            if correct_xpix:
                x_pixel = x_pixel / np.cos(yc * np.pi/180)

            # and a matched array for the values - it needs to be two dimensional
            # with one dimesnion being n_datapoints and the other n_det x n_chan,
            # so we unwrap the 2d array of detector and channel.
            vals = np.array([self.data[i] for i in range(len(self.data)) if ~np.all(self.data[i]==0)])
            inds = np.array([i for i in range(len(self.data)) if ~np.all(self.data[i]==0)])

            # next we'll tack on an array of ones so we can count the data points
            # used for each pixel as well and then divide at the end to go from sums
            # to averages
            to_grid = np.concatenate([vals,np.ones((1,len(vals[0])))]).T
            
            # make the grids
            grids, ax = mk_grid(pos, to_grid, calc_density=False, voxel_length=(x_pixel,y_pixel), fit_dims=fit_dims, dims=dims, center=center)

            # Here's a set of pixel edges in RA and Dec
            map_x = ax[0].flatten() - x_pixel/2
            map_x = np.concatenate((map_x,[map_x[-1]+x_pixel]))
            map_y = ax[1].flatten() - y_pixel/2
            map_y = np.concatenate((map_y,[map_y[-1]+y_pixel]))

            # we then re-wrap our maps into a 16 x 60 grid
            maps = np.zeros((grids[:,:,0].shape[0],grids[:,:,0].shape[1],len(self.data)))
            for i in range(len(inds)):
                maps[:,:,inds[i]] = grids[:,:,i]
            maps = maps.transpose((2,1,0))

            # and finally divide to get average counts instead of sums
            maps /= grids[:,:,-1].T
            maps_count = grids[:,:,-1].T
            maps_count = np.array([maps_count for i in range(nfeed)])

        # Loop over individual feeds, applying offsets when mapping
        elif use_offsets:
            ra = []
            dec = []
            for i_x in range(nfeed):
                x,y = self.offset_pos(i_x,raise_nan=False)
                ra.append(x)
                dec.append(y)

            if dims is not None:
                fit_dims = False
            else:
                dims = [np.ptp(np.concatenate(ra)), np.ptp(np.concatenate(dec))]
                center = [np.min(np.concatenate(ra)) + np.ptp(np.concatenate(ra))/2,
                          np.min(np.concatenate(dec)) + np.ptp(np.concatenate(dec))/2]
                fit_dims = False

            if correct_xpix:
                yc = np.median(np.concatenate(dec))
                x_pixel = x_pixel / np.cos(yc * np.pi/180)

            do_setup = True
            for i_x in range(nfeed):
                inds_x = self.get_x(i_x,raise_nodet=False)
                
                # If no detectors in this index, continue
                if len(inds_x) == 0:
                    continue

                pos = np.array([ra[i_x],dec[i_x]]).T
                vals = np.array([self.data[i] for i in inds_x if ~np.all(self.data[i]==0)])
                to_grid = np.concatenate([vals,np.ones((1,len(vals[0])))]).T
                grids, ax = mk_grid(pos, to_grid, calc_density=False, voxel_length=(x_pixel,y_pixel), fit_dims=fit_dims, dims=dims, center=center)

                # make the final map array and the coordinates the first time a set of maps is generated, but don't need to do it again.
                if do_setup:
                    maps = np.zeros((len(self.data),grids[:,:,0].shape[1],grids[:,:,0].shape[0]))

                    maps_count = np.zeros((nfeed,grids[:,:,0].shape[1],grids[:,:,0].shape[0]))

                    map_x = ax[0].flatten() - x_pixel/2
                    map_x = np.concatenate((map_x,[map_x[-1]+x_pixel]))
                    map_y = ax[1].flatten() - y_pixel/2
                    map_y = np.concatenate((map_y,[map_y[-1]+y_pixel]))
                    do_setup = False

                # we then re-wrap our maps into a 16 x 60 grid
                maps_count[i_x] = grids[:,:,-1].T
                for i in range(len(inds_x)):
                    maps[inds_x[i]] = grids[:,:,i].T
                    maps[inds_x[i]] /= grids[:,:,-1].T
        
        map_pars = {'coords':coordinate_system,
            'x_pixel':x_pixel,
            'y_pixel':y_pixel,
            'x_dim':np.abs(map_x[0]-map_x[-1]),
            'y_dim':np.abs(map_y[0]-map_y[-1]),
            'map_angle_offset':self.header['scan_pars']['map_angle_offset']
            }
        if use_pointing_center:
            map_pars['x_center'] = self.header['center_ra']
            map_pars['y_center'] = self.header['center_dec']
        else:
            map_pars['x_center'] = (map_x[0]+map_x[-1])/2
            map_pars['y_center'] = (map_y[0]+map_y[-1])/2

        if coordinate_system in ['l-b','ra-dec']:
            map_pars['coord_epoch'] = self.header['epoch']
            map_pars['dx_pixel'] = map_pars['x_pixel'] * np.cos(np.median(self.dec) * np.pi/180)
            map_pars['dx_dim'] = map_pars['x_dim'] * np.cos(np.median(self.dec) * np.pi/180)
        else:
            map_pars['coord_epoch'] = 'undefined'
            map_pars['x_pixel'] = map_pars['dx_pixel'] * np.cos(np.median(self.el) * np.pi/180)
            map_pars['x_dim'] = map_pars['dx_dim'] * np.cos(np.median(self.el) * np.pi/180)

        self.Maps = MapConstructor(self.header,map_pars,map_x,map_y,maps,maps_count)
        self.header['flags']['maps_initialized'] = True

    def var_weighted_make_map(self, pixel=1./60./2.,correct_xpix=True,coordinate_system=None,dims=None,center=None,use_offsets=False,use_pointing_center=True):
        """Make maps of all detectors
        
        This method maps all detectors simultaneously.
        Maps are constructed by inserting each sample for
        a given detector into a grid and then averaging all 
        samples in a given grid cell. By default the map dimensions
        are guessed from the range of R.A. and declination 
        (or Galactic l and b, or azimuth and elevation)
        in the timestream, but they can optionally
        be specified.

        Once created, maps are stored in the ``Timestream.Maps``
        attribute, which is in turn an instance of the 
        ``timesoft.Maps`` class. They can be interacted 
        with following the documentation for that class,
        as well as a few convenience functions defined for
        the ``Timestream`` class itself.

        Parameters
        ----------
        pixel : float or list
            The size of each pixel specified in degrees. A single
            number will result in square pixels. Two values can
            be given in a list to specify the size of the pixels 
            in R.A. and declination separately.
        correct_xpix : bool, default=True
            Away from 0 declination/elevation, changing an object's
            coordinates by 1 degree in R.A./azimuth is not equivalent
            to moving the object by 1 degree (instead it is a move 
            of 1 degree * cos(declination/elevation)). By default, the
            pixel size is assumed to be specified such that it is the
            same at any point on the sky, thus when binning the 
            x-coordinate of the data (R.A. or azimuth), the bins 
            used are spaced by pixel / cos(y coordinate)). To turn
            this behavior off set this parameter to False. This
            will cause the bins to be defined directly by the pixel
            parameter with no correction.
        coordinate_system : {'ra-dec','az-el','l-b'}, optional
            The coordinates to map in. By default either 
            'ra-dec' or 'l-b' will be used, depending on what
            coordinate system the timestream data is currently 
            represented in.
        dims : array_like, optional
            Two element array giving the dimensions of the map 
            in degrees. If not specified the dimensions will be
            determined automatically. If specified, a center for
            the map must also be given.
        center : array_like, optional
            Two element array giving the location of the map 
            center in degrees. If not specified, it will be
            determined automatically.
        use_offsets : bool, default=False
            If True, apply offset corrections to the sample positions
            before mapping.
        use_pointing_center : bool, default=True
            If True, the center passed to the Map object will 
            be the commanded position of the observation (otherwise it
            will be the center of the map grid)

        Returns
        -------
        None

        Notes
        -----
        **To Do**        
        Add support for weighted averages
        """

        print('- Making filtered var weighted maps')

        if self.header['scan_pars']['type'] in ['1d','1D']:
            raise ValueError('Timestream is for a 1d dataset, mapping not available')

        if use_offsets and not self.header['flags']['has_feed_offsets']:
            raise ValueError('No offsets specified, cannot apply corrections')

        # Check coordinate system setup
        if coordinate_system is None:
            if self.header['epoch'] not in ['Galactic','galactic']:
                coordinate_system = 'ra-dec'
            else:
                coordinate_system = 'l-b'
        
        if coordinate_system in ['l-b','ra-dec']:
            if coordinate_system == 'ra-dec' and self.header['epoch'] in ['Galactic','galactic']:
                warnings.warn("Timestream is currently represented in 'l-b' coordinates, maps will appear in these coordinates, not 'ra-dec'")
                coordinate_system == 'l-b'
            if coordinate_system == 'l-b' and self.header['epoch'] not in ['Galactic','galactic']:
                warnings.warn("Timestream is currently represented in 'ra-dec' coordinates, maps will appear in these coordinates, not 'l-b'")
                coordinate_system == 'ra-dec'
        elif coordinate_system == 'az-el':
            if use_offsets:
                raise ValueError("Offsets not implemented for az-el maps")
        else:
            raise ValueError("Coordinate system not be recognized")

        # Check dimension inputs
        if dims is not None or center is not None:
            if dims is None or center is None:
                raise ValueError("Must specify both center and dimensions, or neither")

        # Set up pixels
        if len(np.array(pixel,ndmin=1)) > 1:
            x_pixel = pixel[0]
            y_pixel = pixel[1]
        else:
            x_pixel = pixel
            y_pixel = pixel

        # Implementation for mapping all detectors simultaneously
        if not use_offsets:
            if dims is not None:
                fit_dims = False
            else:
                fit_dims = True

            # need an array of the positions for each datapoint
            if coordinate_system is None:
                if self.header['epoch'] not in ['Galactic','galactic']:
                    coordinate_system = 'ra-dec'
                else:
                    coordinate_system = 'l-b'
            
            if coordinate_system in ['l-b','ra-dec']:
                pos = np.array([self.ra,self.dec]).T
                yc = np.median(self.dec)
            elif coordinate_system == 'az-el':
                pos = np.array([self.az,self.el]).T
                yc = np.median(self.el)
            if correct_xpix:
                x_pixel = x_pixel / np.cos(yc * np.pi/180)

            # and a matched array for the values - it needs to be two dimensional
            # with one dimesnion being n_datapoints and the other n_det x n_chan,
            # so we unwrap the 2d array of detector and channel.
            f_vals = np.array([self.data[i]/self.scan_stds[i]**2 for i in range(len(self.data)) if ~np.all(self.data[i]==0)])
            w_vals = np.array([1/self.scan_stds[i]**2 for i in range(len(self.data)) if ~np.all(self.data[i]==0)])
            inds = np.array([i for i in range(len(self.data)) if ~np.all(self.data[i]==0)])

            # next we'll tack on an array of ones so we can count the data points
            # used for each pixel as well and then divide at the end to go from sums
            # to averages
            f_to_grid = np.concatenate([f_vals,np.ones((1,len(f_vals[0])))]).T
            w_to_grid = np.concatenate([w_vals,np.ones((1,len(w_vals[0])))]).T


            # make the grids
            f_grids, f_ax = mk_grid(pos, f_to_grid, calc_density=False, voxel_length=(x_pixel,y_pixel), fit_dims=fit_dims, dims=dims, center=center)
            w_grids, w_ax = mk_grid(pos, w_to_grid, calc_density=False, voxel_length=(x_pixel,y_pixel), fit_dims=fit_dims, dims=dims, center=center)

            # Here's a set of pixel edges in RA and Dec
            map_x = f_ax[0].flatten() - x_pixel/2
            map_x = np.concatenate((map_x,[map_x[-1]+x_pixel]))
            map_y = f_ax[1].flatten() - y_pixel/2
            map_y = np.concatenate((map_y,[map_y[-1]+y_pixel]))

            # we then re-wrap our maps into a 16 x 60 grid
            f_maps = np.zeros((f_grids[:,:,0].shape[0],f_grids[:,:,0].shape[1],len(self.data)))
            w_maps = np.zeros((w_grids[:,:,0].shape[0],w_grids[:,:,0].shape[1],len(self.data)))
            for i in range(len(inds)):
                f_maps[:,:,inds[i]] = f_grids[:,:,i]
                w_maps[:,:,inds[i]] = w_grids[:,:,i]
            f_maps = f_maps.transpose((2,1,0))
            w_maps = w_maps.transpose((2,1,0))

            # and finally divide to get average counts instead of sums
            # maps /= grids[:,:,-1].T
            maps = f_maps / w_maps
            maps_count = f_grids[:,:,-1].T
            maps_count = np.array([maps_count for i in range(nfeed)])

        # Loop over individual feeds, applying offsets when mapping
        elif use_offsets:
            ra = []
            dec = []
            for i_x in range(nfeed):
                x,y = self.offset_pos(i_x,raise_nan=False)
                ra.append(x)
                dec.append(y)

            if dims is not None:
                fit_dims = False
            else:
                dims = [np.ptp(np.concatenate(ra)), np.ptp(np.concatenate(dec))]
                center = [np.min(np.concatenate(ra)) + np.ptp(np.concatenate(ra))/2,
                          np.min(np.concatenate(dec)) + np.ptp(np.concatenate(dec))/2]
                fit_dims = False

            if correct_xpix:
                yc = np.median(np.concatenate(dec))
                x_pixel = x_pixel / np.cos(yc * np.pi/180)

            do_setup = True
            for i_x in range(nfeed):
                inds_x = self.get_x(i_x,raise_nodet=False)
                
                # If no detectors in this index, continue
                if len(inds_x) == 0:
                    continue

                pos = np.array([ra[i_x],dec[i_x]]).T

                f_vals = np.array([self.data[i]/self.scan_stds[i]**2 for i in inds_x if ~np.all(self.data[i]==0)])
                w_vals = np.array([1/self.scan_stds[i]**2 for i in inds_x if ~np.all(self.data[i]==0)])
                c_vals = np.ones((w_vals.shape))

                f_to_grid = np.concatenate([f_vals,np.ones((1,len(f_vals[0])))]).T
                w_to_grid = np.concatenate([w_vals,np.ones((1,len(w_vals[0])))]).T
                c_to_grid = np.concatenate([c_vals,np.ones((1,len(w_vals[0])))]).T
                f_grids, f_ax = mk_grid(pos, f_to_grid, calc_density=False, voxel_length=(x_pixel,y_pixel), fit_dims=fit_dims, dims=dims, center=center)
                w_grids, w_ax = mk_grid(pos, w_to_grid, calc_density=False, voxel_length=(x_pixel,y_pixel), fit_dims=fit_dims, dims=dims, center=center)
                c_grids, c_ax = mk_grid(pos, c_to_grid, calc_density=False, voxel_length=(x_pixel,y_pixel), fit_dims=fit_dims, dims=dims, center=center)

                # make the final map array and the coordinates the first time a set of maps is generated, but don't need to do it again.
                if do_setup:
                    maps = np.zeros((len(self.data),f_grids[:,:,0].shape[1],f_grids[:,:,0].shape[0]))
                    e_maps = np.zeros((len(self.data),w_grids[:,:,0].shape[1],w_grids[:,:,0].shape[0]))
                    h_maps = np.zeros((len(self.data),w_grids[:,:,0].shape[1],w_grids[:,:,0].shape[0]))

                    maps_count = np.zeros((nfeed,f_grids[:,:,0].shape[1],f_grids[:,:,0].shape[0]))

                    map_x = f_ax[0].flatten() - x_pixel/2
                    map_x = np.concatenate((map_x,[map_x[-1]+x_pixel]))
                    map_y = f_ax[1].flatten() - y_pixel/2
                    map_y = np.concatenate((map_y,[map_y[-1]+y_pixel]))
                    do_setup = False

                # we then re-wrap our maps into a 16 x 60 grid
                maps_count[i_x] = f_grids[:,:,-1].T


                # maps_count[i_x] = grids[:,:,-1].T
                # for i in range(len(inds_x)):
                    # maps[inds_x[i]] = grids[:,:,i].T
                    # maps[inds_x[i]] /= grids[:,:,-1].T


                for i in range(len(inds_x)):
                    maps[inds_x[i]] = f_grids[:,:,i].T / w_grids[:,:,i].T
                    e_maps[inds_x[i]] = 1 / np.sqrt(w_grids[:,:,i].T)

                    h_maps[inds_x[i]] = c_grids[:,:,i].T
                    # maps[inds_x[i]] /= grids[:,:,-1].T
        
        map_pars = {'coords':coordinate_system,
            'x_pixel':x_pixel,
            'y_pixel':y_pixel,
            'x_dim':np.abs(map_x[0]-map_x[-1]),
            'y_dim':np.abs(map_y[0]-map_y[-1]),
            'map_angle_offset':self.header['scan_pars']['map_angle_offset']
            }
        if use_pointing_center:
            map_pars['x_center'] = self.header['center_ra']
            map_pars['y_center'] = self.header['center_dec']
        else:
            map_pars['x_center'] = (map_x[0]+map_x[-1])/2
            map_pars['y_center'] = (map_y[0]+map_y[-1])/2

        if coordinate_system in ['l-b','ra-dec']:
            map_pars['coord_epoch'] = self.header['epoch']
            map_pars['dx_pixel'] = map_pars['x_pixel'] * np.cos(np.median(self.dec) * np.pi/180)
            map_pars['dx_dim'] = map_pars['x_dim'] * np.cos(np.median(self.dec) * np.pi/180)
        else:
            map_pars['coord_epoch'] = 'undefined'
            map_pars['x_pixel'] = map_pars['dx_pixel'] * np.cos(np.median(self.el) * np.pi/180)
            map_pars['x_dim'] = map_pars['dx_dim'] * np.cos(np.median(self.el) * np.pi/180)

        self.Maps = MapConstructor(self.header,map_pars,map_x,map_y,maps,maps_count,e_maps, h_maps)
        self.header['flags']['maps_initialized'] = True
        
    def make_map_det(self, c1, c2, det_coord_mode='xf', plot=True, cbar=True, show=True, pixel=1./60./2., use_offsets=False):
        """Quicklook map of a single detector
        
        This method maps a single detector. It is meant to
        provide a quick way to visualize the data and takes
        less time to run than mapping all detectors in a 
        large dataset. By default, this method plots of the 
        resulting map, although this can be turned off.
        Making final maps should be done with
        ``Timestream.make_map()`` which provides more options
        and processes all detectors uniformly. 

        Once created, the map is stored in the ``Timestream.map_det``
        attribute. The R.A. and declination
        coordinates of the maps are saved in ``Timestream.map_x_det``
        and ``Timestream.map_y_det`` attributes and specify the 
        coordinates at the edges of each map cell.

        Parameters
        ----------
        c1, c2 : int
            The coordinates in either xf (default) or muxcr space
            of the detector to process. The coordinates system to 
            use is controlled by `det_coord_mode`.
        det_coord_mode : {'xf','cr'}, default='xf'
            Coordinate system used to identify detectors.
            'xf' (default) or 'cr' are accepted.
        plot : bool, default=True
            If True, a plot of the resulting map will be displayed.
        cbar : bool, default=True
            If True, the displayed plot will include a colorbar
        show : bool, default=True
            If True, plt.show() will be called (set to False if you
            don't want the plot shown immediately)
        pixel : float or list
            The size of each pixel specified in degrees. A single
            number will result in square pixels. Two values can
            be given in a list to specify the size of the pixels 
            in R.A. and declination separately.
        use_offsets : bool, default=False
            If True, apply offset corrections to the sample positions
            before mapping.

        Returns
        -------
        None        
        """

        if self.header['scan_pars']['type'] in ['1d','1D']:
            raise ValueError('Timestream is for a 1d dataset, mapping not available')

        if use_offsets and not self.header['flags']['has_feed_offsets']:
            raise ValueError('No offsets specified, cannot apply corrections')

        index = self._get_coord(c1,c2,det_coord_mode)

        if use_offsets:
            ra, dec = self.offset_pos(self.header['xf_coords'][index][0])
        else:
            ra = self.ra
            dec = self.dec

        if len(np.array(pixel,ndmin=1)) > 1:
            self.pixel_x_det = pixel[0] / np.cos(np.median(dec) * np.pi/180)
            self.pixel_y_det = pixel[1]
        else:
            self.pixel_x_det = pixel / np.cos(np.median(dec) * np.pi/180)
            self.pixel_y_det = pixel

        # need an array of the positions for each datapoint
        pos = np.array([ra,dec]).T

        # and a matched array for the values
        vals = self.data[index]

        # next we'll tack on an array of ones so we can count the data points
        # used for each pixel as well and then divide at the end to go from sums
        # to averages
        to_grid = np.array([np.ones(len(vals)),vals]).T

        # make the grids
        grids, ax = mk_grid(pos, to_grid, calc_density=False, voxel_length=(self.pixel_x_det,self.pixel_y_det), fit_dims=True)

        # Here's a set of pixel edges in RA and Dec
        self.map_x_det = ax[0].flatten() - self.pixel_x_det/2
        self.map_x_det = np.concatenate((self.map_x_det,[self.map_x_det[-1]+self.pixel_x_det]))
        self.map_y_det = ax[1].flatten() - self.pixel_y_det/2
        self.map_y_det = np.concatenate((self.map_y_det,[self.map_y_det[-1]+self.pixel_y_det]))
        self.map_det = (grids[:,:,1]/grids[:,:,0]).T
        self.map_det_count = grids[:,:,-1].T

        if plot:
            xx,yy = np.meshgrid(self.map_x_det,self.map_y_det)
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111)
            c = ax.pcolormesh(xx,yy,self.map_det)
            if cbar:
                fig.colorbar(c)
            ax.set(xlabel='RA (degrees)', ylabel='Dec (degrees)')
            if show:
                plt.show()

    def save_map_level1(self,Maps):

        from datetime import date
        import netCDF4 as nc

        # make a full copy of the dictionary
        time_head = copy.deepcopy(Maps.header)

        # delete the other keys that are nested dictionaries
        # netcdf doesn't like those...
        keys_to_remove = ['flags','scan_pars','detector_pars','map_pars']
        for key in keys_to_remove:
            try:
                del time_head[key]
            except KeyError:
                pass

        # create new file and open in write mode
        f = nc.Dataset(config.data_save + f'level1_maps_{date.today()}.nc',"w")

        # add map parameters
        parms = f.createGroup('map_pars')
        for k,v in Maps.header['map_pars'].items():
            if isinstance(v, bool) :
                setattr(parms,k,int(v)) # converts bool to int
            elif isinstance(v,type(None)) :
                setattr(parms,k,'None')
            else :
                setattr(parms,k,v)

        # add detector parameters
        det_parm = f.createGroup('det_pars')
        for k,v in Maps.header['detector_pars'].items():
            if isinstance(v, bool) :
                setattr(det_parm,k,int(v)) # converts bool to int
            elif isinstance(v,type(None)) :
                setattr(det_parm,k,'None')
            else :
                setattr(det_parm,k,v)

        flags_parm = f.createGroup('flags')
        for k,v in Maps.header['flags'].items():
            if isinstance(v, bool) :
                setattr(flags_parm,k,int(v)) # converts bool to int
            elif isinstance(v,type(None)) :
                setattr(flags_parm,k,'None')
            else :
                setattr(flags_parm,k,v)

        scan_parm = f.createGroup('scan_pars')
        for k,v in Maps.header['scan_pars'].items():
            if isinstance(v, bool) :
                setattr(scan_parm,k,int(v)) # converts bool to int
            elif isinstance(v,type(None)) :
                setattr(scan_parm,k,'None')
            else :
                setattr(scan_parm,k,v)

        # save remaining timestream parameters
        time_parm = f.createGroup('timestream_pars')
        # netcdf does not let attributes be arrays, so we need to save these as variables
        # convert list of tuples to list of list, because netcdf said NO
        n_xf = np.array([list(ele) for ele in time_head['xf_coords']])
        n_cr = np.array([list(ele) for ele in time_head['cr_coords']])
        f.createDimension('num_det',n_xf.shape[0])
        f.createDimension('coor',2)
        xf_coords = time_parm.createVariable('xf_coords','f8',('num_det','coor'))
        xf_coords[:,:] = n_xf
        cr_coords = time_parm.createVariable('cr_coords','f8',('num_det','coor'))
        cr_coords[:,:] = n_cr

        f.createDimension('num_feed',time_head['feed_offsets'].shape[0])
        feed_offsets = time_parm.createVariable('feed_offsets','f8',('num_feed','coor'))
        feed_offsets[:,:] = time_head['feed_offsets']

        f.createDimension('num_gain',time_head['gains'].shape[0])
        gains = time_parm.createVariable('gains','f8',('num_gain'))
        gains[:] = time_head['gains']

        # then we remove those keys from the dictionary before adding the rest as attributes
        keys_to_remove = ['xf_coords','cr_coords','feed_offsets','gains']
        for key in keys_to_remove:
            try:
                del time_head[key]
            except KeyError:
                pass

        # add remaining timestream header info
        for k,v in time_head.items():
            if isinstance(v, bool) :
                setattr(time_parm,k,int(v)) # converts bool to int
            elif isinstance(v,type(None)) :
                setattr(time_parm,k,'None')
            else :
                setattr(time_parm,k,v)

        # create variable with dimensions to store map arrays
        map_params = Maps.header['map_pars']
        ''' This one wasn't working. For some reason x_dim and y_dim 
            are the RA and DEC centers which doesn't make a lot of sense...
        '''
        # f.createDimension('pix_x',map_params['x_dim'])
        # f.createDimension('pix_y',map_params['y_dim'])
        f.createDimension('pix_x',Maps.maps[:,:,0].shape[0])
        f.createDimension('pix_y',Maps.maps[:,:,0].shape[1])
        f.createDimension('num_map',Maps.maps.shape[2]) # (x,y,num)
        m = f.createVariable('maps','f8',('pix_x','pix_y','num_map'))
        # add maps to file
        m[:,:,:] = Maps.maps

        # close the file
        f.close()

        return config.data_save + f'level1_maps_{date.today()}.nc'

    def make_fits(self,Maps):
        """Save calibrated maps in a fits file with header information
           For continued use throughout the rest of the pipeline
           Gets called within make_map

           Parameters
           ----------


           Returns
           ---------
           Fits file containing one or all detector maps for a given source
        """
        from astropy.io import fits
        from astropy import wcs
        from datetime import date

        map_head = Maps.header['map_pars']

        wp = wcs.WCS(naxis=2)
        wp.wcs.cdelt = [map_head['x_pixel'],map_head['y_pixel']]
        wp.wcs.crval = [map_head['x_center'], map_head['y_center']]
        wp.wcs.ctype = ["RA", "DEC"]
        # wp.wcs.jepoch = Maps.header['epoch']
        # wp.wcs.dateavg = Maps.header['datetime'] # in UTC

        # create header information for file
        hdu = wp.to_header()
        cr = Maps.header['cr_coords']

        hda = []
        hdul = fits.HDUList(hdus=hda)
        d = 0
        for map in Maps.maps :
            # find which detector this map is for
            det = cr[d]
            hdul.append(fits.ImageHDU(data=map,header=hdu,name=f'c{det[0]}r{det[1]}'))
            d += 1
        hdul.writeto(config.data_save + f'level1_maps_{date.today()}.fits',overwrite=True)
        return config.data_save + f'level1_maps_{date.today()}.fits', wp

    def plot_map(self,c1,c2,det_coord_mode='xf',savepath=None,show=True,**kwargs):
        """View map of single detector
        
        This method plots the map of a single detector. It is
        a wrapper of ``Timestream.Map.plot``.

        Parameters
        ----------
        c1, c2 : int
            The coordinates in either xf (default) or muxcr space
            of the detector to process. The coordinates system to 
            use is controlled by `det_coord_mode`.
        det_coord_mode : {'xf','cr'}, default='xf'
            Coordinate system used to identify detectors.
            'xf' (default) or 'cr' are accepted.
        show : bool, default=True
            If True, plt.show() will be called (set to False if you
            don't want the plot shown immediately)
        savepath : str, optional
            If specified, the resulting plot will be saved in the 
            path provided.
        kwargs : 
            See ``timesfoft.Map.plot`` for additional options

        Returns
        -------
        None        
        """

        if not self.header['flags']['maps_initialized']:
            raise ValueError('Detectors have not been mapped')

        self.Maps.plot(c1=c1, c2=c2, det_coord_mode=det_coord_mode, savepath=savepath, show=show, **kwargs)

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

        This is a wrapper around ``Timestream.Maps.detector_grid``

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

        if not self.header['flags']['maps_initialized']:
            raise ValueError('Detectors have not been mapped')

        self.Maps.detector_grid(savepath=savepath, show=show)

    def make_1d(self, pixel=1./60./2., axis=None, extras={}, use_offsets=False):
        """Bin samples along a specified axis

        This method bins scans along a specified axis.
        It is designed for processing line scan data,
        but can also be used to bin 2D data along a 
        single axis in cases where that is useful for
        some reason.

        By default the binning is done along the scan
        direction (if it is available in the header).
        If no direction is in the header then the default
        is to bin in R.A. It is possible to manually 
        set the binning direction, and binning by 
        azimuth and elevation are supported in additon
        to R.A. and declination.

        The resulting binned data is stored in the 
        ``Timestream.LineMap`` attribute, which is an
        instance of the ``timesoft.maps.LineMap`` class.

        Optionally, a list of additional timestreams to
        be binned can be specified. These should be 1d 
        arrays of the same length as ``self.t``. They will
        be binned along the same axis and stored in
        ``Timestream.LineMaps.extras``. This is useful if you
        want to examine the average value of some other
        timestream element in each bin (e.g. some flag
        you've defined for each point in time).

        Parameters
        ----------
        pixel : float
            The size of the bins to use, specified in 
            degrees.
        axis : {'ra','dec','az','el'}, optional
            The axis along which to bin the data. By
            default the scan axis is used if it can be
            determined from the headers, or 'ra' otherwise.
        extras : dict, optional
            If specified, it should be a dictionary. Each 
            item in the dictionary should be an array
            which will be binned along with the data, while
            the keys should provide a descriptive name for
            this item. Each element of also_bin should be 
            an array of the same length as ``Timestream.t`` 
            (i.e. have the same number of samples as the 
            timestream). The data in each array should be 
            castable to a float.
        use_offsets : bool, default=False
            If True, apply offset corrections to the sample positions
            before mapping.

        Returns
        -------
        None

        Notes
        -----
        **To Do**
        Add support for weighted averages
        """

        if axis is None:
            axis = self._get_scan_direction()

        if axis in ['ra','RA','r.a.','R.A.']:
            pos = np.array([self.ra]).T
            if self.header['epoch'] == 'Galactic':
                axis = 'Galactic Longitude'
            else:
                axis = 'R.A.'
        elif axis in ['dec','Dec','DEC','declimation','Declination']:
            pos = np.array([self.dec]).T
            if self.header['epoch'] == 'Galactic':
                axis = 'Galactic Latitude'
            else:
                axis = 'Declination'
        elif axis in ['az','Az','azimuth','Azimuth']:
            if use_offsets:
                raise ValueError("Offsets not implemented for azimuth")
            pos = np.array([self.az]).T
            axis = 'Azimuth'
        elif axis in ['el','El','elevation','Elevation','alt','altitude','Alt','Altitude']:
            if use_offsets:
                raise ValueError("Offsets not implemented for elevation")
            pos = np.array([self.el]).T
            axis = 'Elevation'
        else:
            raise ValueError("Scan direction not recognized")
        
        # we'll tack on an array of ones so we can count the data points
        # used for each pixel then divide at the end to go from sums
        # to averages
        props=[]
        keys=[]
        n_extra = len(extras)
        if n_extra > 0:
            for key in extras.keys():
                if len(extras[key]) != len(self.t):
                    raise ValueError("Properties in extras do not have the correct length")
                props.append(extras[key].reshape(1,-1))
                keys.append(key)

        if not use_offsets:
            if axis in ['Galactic Longitude','R.A.']:
                pos = np.array([self.ra]).T
            elif axis in ['Galactic Latitude','Declination']:
                pos = np.array([self.dec]).T
            elif axis in ['Azimuth']:
                pos = np.array([self.az]).T
            elif axis in ['Elevation']:
                pos = np.array([self.el]).T

            # make the grids
            to_grid = np.concatenate([self.data,np.ones((1,len(self.t)))]+props).T
            grids, ax = mk_grid(pos, to_grid, calc_density=False, voxel_length=pixel, fit_dims=True, one_d=True)

            # Here's a set of pixel edges in RA and Dec
            m1d_ax = ax[0].flatten() - pixel/2
            m1d_ax = np.concatenate((m1d_ax,[m1d_ax[-1]+pixel]))

            # Extract gridded values and divide by nuber of points
            m1d_count = grids[:,-1-n_extra].T
            m1d = grids[:,:-1-n_extra].T / grids[:,-1-n_extra].T
            m1d_count = np.array([grids[:,-1-n_extra].T for i_x in range(nfeed)])

            if n_extra>0:
                m1d_extras = {keys[i]:np.array([grids[:,-n_extra + i].T / grids[:,-1-n_extra].T for i_x in range(nfeed)]) for i in range(n_extra)}
            else:
                m1d_extras = {}

        if use_offsets:
            if axis in ['Galactic Longitude','R.A.']:
                pos = [self.offset_pos(i_x,raise_nan=False)[0] for i_x in range(nfeed)]
            elif axis in ['Galactic Latitude','Declination']:
                pos = [self.offset_pos(i_x,raise_nan=False)[1] for i_x in range(nfeed)]

            dims = [np.ptp(pos)]
            center = [np.min(pos)+np.ptp(pos)/2]

            do_setup = True
            for i_x in range(nfeed):
                inds_x = self.get_x(i_x,raise_nodet=False)
                
                # If no detectors in this index, continue
                if len(inds_x) == 0:
                    continue

                # make the grids
                to_grid = np.concatenate([self.data[i].reshape(1,-1) for i in inds_x]+[np.ones((1,len(self.t)))]+props).T
                grids, ax = mk_grid(pos[i_x].reshape(-1,1), to_grid, calc_density=False, voxel_length=pixel, fit_dims=False, dims=dims, center=center)

                # make the final arrays and the coordinates the first time, but don't need to do it again.
                if do_setup:
                    # Here's a set of pixel edges in RA and Dec
                    m1d_ax = ax[0].flatten() - pixel/2
                    m1d_ax = np.concatenate((m1d_ax,[m1d_ax[-1]+pixel]))

                    m1d_count = np.zeros((nfeed,grids[:,0].shape[0]))
                    m1d = np.zeros((len(self.data),grids[:,0].shape[0]))
                    if n_extra>0:
                        m1d_extras = {keys[i]:np.zeros((nfeed,grids[:,0].shape[0])) for i in range(n_extra)}
                    else:
                        m1d_extras = {}
                    do_setup = False

                # Extract gridded values and divide by nuber of points
                m1d_count[i_x] = grids[:,-1-n_extra].flatten()
                for i in range(len(inds_x)):
                    m1d[inds_x[i]] = grids[:,i].flatten() / m1d_count[i_x]

                if n_extra>0:
                    for i in range(n_extra):
                        m1d_extras[keys[i]][i_x] = grids[:,-n_extra + i].flatten() / m1d_count[i_x]

        m1d_pars = {'axis':axis,
                    'pixel':pixel,
                    'center':(m1d_ax[0]+m1d_ax[-1])/2,
                    'length':np.abs(m1d_ax[0]-m1d_ax[-1])
                    }
        
        if axis in ['R.A.','Declination','Galactic Longitude','Galactic Latitude']:
            m1d_pars['coord_epoch'] = self.header['epoch']
        else:
            m1d_pars['coord_epoch'] = 'undefined'

        self.header['flags']['1d_initialized'] = True
        self.LineMaps = LineMapConstructor(self.header,m1d_pars,m1d_ax,m1d,m1d_count,m1d_extras)

    def make_1d_with_var_weighting(self, pixel=1./60./2., axis=None, extras={}, use_offsets=False):
        """Bin samples along a specified axis

        This method bins scans along a specified axis.
        It is designed for processing line scan data,
        but can also be used to bin 2D data along a 
        single axis in cases where that is useful for
        some reason.

        By default the binning is done along the scan
        direction (if it is available in the header).
        If no direction is in the header then the default
        is to bin in R.A. It is possible to manually 
        set the binning direction, and binning by 
        azimuth and elevation are supported in additon
        to R.A. and declination.

        The resulting binned data is stored in the 
        ``Timestream.LineMap`` attribute, which is an
        instance of the ``timesoft.maps.LineMap`` class.

        Optionally, a list of additional timestreams to
        be binned can be specified. These should be 1d 
        arrays of the same length as ``self.t``. They will
        be binned along the same axis and stored in
        ``Timestream.LineMaps.extras``. This is useful if you
        want to examine the average value of some other
        timestream element in each bin (e.g. some flag
        you've defined for each point in time).

        Parameters
        ----------
        pixel : float
            The size of the bins to use, specified in 
            degrees.
        axis : {'ra','dec','az','el'}, optional
            The axis along which to bin the data. By
            default the scan axis is used if it can be
            determined from the headers, or 'ra' otherwise.
        extras : dict, optional
            If specified, it should be a dictionary. Each 
            item in the dictionary should be an array
            which will be binned along with the data, while
            the keys should provide a descriptive name for
            this item. Each element of also_bin should be 
            an array of the same length as ``Timestream.t`` 
            (i.e. have the same number of samples as the 
            timestream). The data in each array should be 
            castable to a float.
        use_offsets : bool, default=False
            If True, apply offset corrections to the sample positions
            before mapping.

        Returns
        -------
        None

        Notes
        -----
        **To Do**
        Add support for weighted averages
        """

        if axis is None:
            axis = self._get_scan_direction()

        if axis in ['ra','RA','r.a.','R.A.']:
            pos = np.array([self.ra]).T
            if self.header['epoch'] == 'Galactic':
                axis = 'Galactic Longitude'
            else:
                axis = 'R.A.'
        elif axis in ['dec','Dec','DEC','declimation','Declination']:
            pos = np.array([self.dec]).T
            if self.header['epoch'] == 'Galactic':
                axis = 'Galactic Latitude'
            else:
                axis = 'Declination'
        elif axis in ['az','Az','azimuth','Azimuth']:
            if use_offsets:
                raise ValueError("Offsets not implemented for azimuth")
            pos = np.array([self.az]).T
            axis = 'Azimuth'
        elif axis in ['el','El','elevation','Elevation','alt','altitude','Alt','Altitude']:
            if use_offsets:
                raise ValueError("Offsets not implemented for elevation")
            pos = np.array([self.el]).T
            axis = 'Elevation'
        else:
            raise ValueError("Scan direction not recognized")
        
        # we'll tack on an array of ones so we can count the data points
        # used for each pixel then divide at the end to go from sums
        # to averages
        props=[]
        keys=[]
        n_extra = len(extras)
        if n_extra > 0:
            for key in extras.keys():
                if len(extras[key]) != len(self.t):
                    raise ValueError("Properties in extras do not have the correct length")
                props.append(extras[key].reshape(1,-1))
                keys.append(key)

        if not use_offsets:
            if axis in ['Galactic Longitude','R.A.']:
                pos = np.array([self.ra]).T
            elif axis in ['Galactic Latitude','Declination']:
                pos = np.array([self.dec]).T
            elif axis in ['Azimuth']:
                pos = np.array([self.az]).T
            elif axis in ['Elevation']:
                pos = np.array([self.el]).T

            # make the grids
            # #------ 2d map example to copy from...
            # f_vals = np.array([self.data[i]/self.scan_stds[i]**2 for i in range(len(self.data)) if ~np.all(self.data[i]==0)])
            # w_vals = np.array([1/self.scan_stds[i]**2 for i in range(len(self.data)) if ~np.all(self.data[i]==0)])
            # inds = np.array([i for i in range(len(self.data)) if ~np.all(self.data[i]==0)])

            # # next we'll tack on an array of ones so we can count the data points
            # # used for each pixel as well and then divide at the end to go from sums
            # # to averages
            # f_to_grid = np.concatenate([f_vals,np.ones((1,len(f_vals[0])))]).T
            # w_to_grid = np.concatenate([w_vals,np.ones((1,len(w_vals[0])))]).T

            # #------
            #### note that self.scan_stds here has to be the same shape as the data array, meaning that
            #### in general it will be a list of [x,x,x,x,x,x,x,y,y,y,y,y,y,y,z,z,z,z,z,z,z] where x,y,z are dfferent scans

            f_to_grid = np.concatenate([self.data/self.scan_stds**2,np.ones((1,len(self.t)))]+props).T
            w_to_grid = np.concatenate([1/self.scan_stds**2,np.ones((1,len(self.t)))]+props).T
            f_grids, ax = mk_grid(pos, to_grid, calc_density=False, voxel_length=pixel, fit_dims=True)
            w_grids, w_ax = mk_grid(pos, to_grid, calc_density=False, voxel_length=pixel, fit_dims=True)

            # Here's a set of pixel edges in RA and Dec
            m1d_ax = ax[0].flatten() - pixel/2
            m1d_ax = np.concatenate((m1d_ax,[m1d_ax[-1]+pixel]))

            # Extract gridded values and divide by weights
            m1d_count = f_grids[:,-1-n_extra].T
            m1d = f_grids[:,:-1-n_extra].T / w_grids[:,:-1-n_extra].T
            m1d_count = np.array([f_grids[:,-1-n_extra].T for i_x in range(nfeed)])

            if n_extra>0:
                m1d_extras = {keys[i]:np.array([f_grids[:,-n_extra + i].T / w_grids[:,-n_extra + i].T for i_x in range(nfeed)]) for i in range(n_extra)}
            else:
                m1d_extras = {}

        if use_offsets:
            if axis in ['Galactic Longitude','R.A.']:
                pos = [self.offset_pos(i_x,raise_nan=False)[0] for i_x in range(nfeed)]
            elif axis in ['Galactic Latitude','Declination']:
                pos = [self.offset_pos(i_x,raise_nan=False)[1] for i_x in range(nfeed)]

            dims = [np.ptp(pos)]
            center = [np.min(pos)+np.ptp(pos)/2]

            do_setup = True
            for i_x in range(nfeed):
                inds_x = self.get_x(i_x,raise_nodet=False)
                
                # If no detectors in this index, continue
                if len(inds_x) == 0:
                    continue

                # make the grids
                f_to_grid = np.concatenate([self.data[i].reshape(1,-1) /self.scan_stds[i].reshape(1,-1)  for i in inds_x]+[np.ones((1,len(self.t)))]+props).T
                w_to_grid = np.concatenate([1 /self.scan_stds[i].reshape(1,-1)  for i in inds_x]+[np.ones((1,len(self.t)))]+props).T
                f_grids, ax = mk_grid(pos[i_x].reshape(-1,1), to_grid, calc_density=False, voxel_length=pixel, fit_dims=False, dims=dims, center=center)
                w_grids, w_ax = mk_grid(pos[i_x].reshape(-1,1), to_grid, calc_density=False, voxel_length=pixel, fit_dims=False, dims=dims, center=center)

                # make the final arrays and the coordinates the first time, but don't need to do it again.
                if do_setup:
                    # Here's a set of pixel edges in RA and Dec
                    m1d_ax = ax[0].flatten() - pixel/2
                    m1d_ax = np.concatenate((m1d_ax,[m1d_ax[-1]+pixel]))

                    m1d_count = np.zeros((nfeed,f_grids[:,0].shape[0]))
                    m1d = np.zeros((len(self.data),f_grids[:,0].shape[0]))
                    if n_extra>0:
                        m1d_extras = {keys[i]:np.zeros((nfeed,f_grids[:,0].shape[0])) for i in range(n_extra)}
                    else:
                        m1d_extras = {}
                    do_setup = False

                # Extract gridded values and divide by nuber of points
                m1d_count[i_x] = f_grids[:,-1-n_extra].flatten()
                for i in range(len(inds_x)):
                    m1d[inds_x[i]] = f_grids[:,i].flatten() / w_grids[:,i]

                if n_extra>0:
                    for i in range(n_extra):
                        m1d_extras[keys[i]][i_x] = f_grids[:,-n_extra + i].flatten() / w_grids[:,-n_extra + i].flatten()

        m1d_pars = {'axis':axis,
                    'pixel':pixel,
                    'center':(m1d_ax[0]+m1d_ax[-1])/2,
                    'length':np.abs(m1d_ax[0]-m1d_ax[-1])
                    }
        
        if axis in ['R.A.','Declination','Galactic Longitude','Galactic Latitude']:
            m1d_pars['coord_epoch'] = self.header['epoch']
        else:
            m1d_pars['coord_epoch'] = 'undefined'

        self.header['flags']['1d_initialized'] = True
        self.LineMaps = LineMapConstructor(self.header,m1d_pars,m1d_ax,m1d,m1d_count,m1d_extras)

    def plot_1d(self,*coords,det_coord_mode='xf',savepath=None,show=True):
        """Make plots of detectors 1d binned data

        This is a wrapper around the ``Timestream.LineMaps.plot``
        method.
        """

        if not self.header['flags']['1d_initialized']:
            raise ValueError('Timestreams have not been binned')

        self.LineMaps.plot(*coords,det_coord_mode=det_coord_mode,savepath=savepath,show=show)


    ###################################################
    #### Backwards Compatibility - Don't Use These ####
    ###################################################
    def remove_obs_flag(self,flags=[0,1]):
        """Remove samples with specified values of ``Timestream.flag``
        
        .. deprecated:: 
          `remove_obs_flag` has been renamed remove_tel_flag for clarity.
          This method is now a wrapper for it and is kept for backwards
          compatibility.

        This function removes samples where the ``Timestream.flag`` 
        attribute (the telescope direction information) is in a provided
        list of values.

        The flag values are 0 if telescope is not on source, 1 if telescope
        is tracking field center, 2 if telescope is tracking in the
        negative RA direction, 3 if telescope is tracking in the positive
        RA direction, 4 if telescope is turning around.

        Parameters
        ----------
        flags : list
            A list of one or more flag values to be filtered from the
            timestream

        Returns
        -------
        None
        """
        
        self.remove_tel_flag(flags)
        
    def make_map_az_el(self, pixel=1./60./2.):
        """Make az-el space maps of all detectors
        
        .. deprecated:: 
          `make_map_az_el` functionality is now included in `make_map`.
          This method is now a wrapper kept for backwards compatibility.

        This method maps all detectors simultaneously.
        Maps are constructed by inserting each sample for
        a given detector into a grid of azimuth and 
        elevation and then averaging all samples in 
        a given grid cell. 

        Once created, maps are stored in the ``Timestream.maps``
        attribute, and are ordered in the same was as the
        ``Timestream.data`` attributes. The azimuth and elevation
        coordinates of the maps are saved in ``Timestream.map_x``
        and ``Timestream.map_y`` attributes and specify the 
        coordinates at the edges of each map cell.

        Parameters
        ----------
        pixel : float or list
            The size of each pixel specified in degrees. A single
            number will result in square pixels. Two values can
            be given in a list to specify the size of the pixels 
            in azimuth and elevation separately.

        Returns
        -------
        None
        """

        self.make_map(pixel=pixel,coordinate_system='az-el')

    def make_calibrated_maps(self, planet_cal_path, date_override=None, obj_override=None, set_raise_frame=True,debug_save=False, **kwargs):
        """ A wrapper for the absolute map calibration 
    
        Parameters
        ----------
        planet_cal_path : str
            a str that points to the planet data folder being used for calibration
        date_override : str
            a str that overrides the date in the observation header, used for 2022 maps since header
            info is broken for them, example uses '2021-02-09T05:00:00' IN UTC !
        obj_override : str
            a str that overrides the obj name in the header map

        Returns
        -------
        None
        """

        channels = []
        for i in range(60):
            channels.append(f_ind_to_val(i))
        channels = np.array(channels)
        
        abs_cal = absolute_calibration(channels, self.header)
        offsets, gains, planet_ts, errmsg = abs_cal.compute_gains(planet_cal_path, date_override=date_override, obj_name_override=obj_override,debug_save=debug_save)            
    
        if errmsg:
            return 'bad', errmsg
        else:
            errmsg = False
        #the offsets can be applied to omc data with 
        self.set_feed_offsets(offsets,overwrite=True,raise_frame= set_raise_frame)
        self.correct_gains(gains)
        self.remove_obs_flag() # Get rid of points where the telescope wasn't in an "observing" mode
        self.flag_scans() # Identify individual scans across the map
        self.remove_scan_flag() # If data didn't appear to belong to a scan, drop it
        self.remove_short_scans(thresh=100) # Remove anything that appears to short to be a real scan
        self.remove_end_scans(n_start=4,n_finish=4) # Remove the first few scans and last few scans (ie top and bottom of the map)
        self.remove_scan_edge(n_start=50, n_finish=50) # Remove a few data points from the edges of the scan, where the telescope may still be accelerating
        self.filter_scan(n=5)
        self.make_map(pixel=0.0055,use_offsets=True,center=(84.10,-5.37), dims=(0.3,0.3))

        ###convert gains and detector images to a 16x60 array for vectorized processing.
        self.gains_to_focal_grid()
        self.maps_to_focal_grid()

        return errmsg
    
    
    ###################
    #### Ben WIP?? ####
    ###################
    def gains_to_focal_grid(self):
        gains = self.Maps.header['gains']
        focal_grid_gains = np.zeros((16,60))
        focal_grid_gains[:,:] = np.nan 
        xf_coords = self.header['xf_coords']
        x_coords = np.array([xf[0] for xf in xf_coords])
        f_coords = np.array([xf[1] for xf in xf_coords])
        focal_grid_gains[x_coords,f_coords] = gains 
        self.Maps.header['focal_grid_gains'] = focal_grid_gains
        
    def maps_to_focal_grid(self):
        ndetector, x_size, y_size = self.Maps.maps.shape
        focal_grid = np.zeros((16,60,x_size,y_size))
        focal_grid[:,:,:,:] = np.nan
        xf_coords = self.header['xf_coords']
        x_coords = np.array([xf[0] for xf in xf_coords])
        f_coords = np.array([xf[1] for xf in xf_coords])
        focal_grid[x_coords,f_coords,:,:] = self.Maps.maps
        self.Maps.focal_grid_maps = focal_grid
        self.Maps.header['focal_grid_maps'] = focal_grid

    def make_f_co_add(self, sanity_check=True):
        ndet,nf,x_size,y_size = self.Maps.focal_grid_maps.shape
        weights = 1 / self.Maps.header['focal_grid_gains']
        ca = (np.nansum((self.Maps.focal_grid_maps.reshape(ndet*nf,x_size,y_size).T * weights.flatten()).T.reshape((ndet,nf,x_size,y_size)),axis=0).T / np.nansum(weights,axis=0)).T
        self.Maps.co_add = ca 
        self.Maps.header['co_add'] = ca 

        ###sanity check
        if sanity_check:
            map_x_weight = (self.Maps.focal_grid_maps.reshape(ndet*nf,x_size,y_size).T * weights.flatten()).T.reshape((ndet,nf,x_size,y_size))
            for xf in self.Maps.header['xf_coords']:
                spec_map = map_x_weight[xf[0],xf[1]]
                assert spec_map.all() == (self.Maps.focal_grid_maps[xf[0],xf[1]] * weights[xf[0],xf[1]]).all()

    

    ####################################################################################################################################
    ####################################################################################################################################
    ####################################                                     ###########################################################
    #################################### Common Mode Noise Removal Functions ###########################################################
    ####################################                                     ###########################################################
    ####################################################################################################################################
    ####################################################################################################################################
    ####################################################################################################################################
    ####################################################################################################################################
    ####################################                                     ###########################################################
    #################################### Common Mode Noise Removal Functions ###########################################################
    ####################################                                     ###########################################################
    ####################################################################################################################################
    ####################################################################################################################################

    def remove_common_mode_noise(self,save_path,pixel,debugging_plots=True,common_mode_params={'remove_cm':True,'strength':1, 'mux_block':4,'freq_block':[0,8,16,24,36,48,60],'subtract_which':'mean',},**kwargs):
        """
        wrapper function that carries out the common mode noise removal on the timestream data
        
        """


        print('\tbegin setting up for common mode noise removal')
        
        strength_param = common_mode_params['strength']
        subtract_which = common_mode_params['subtract_which']
        if subtract_which not in ['mean','lowpass']:
            raise ValueError(f"subtract_which={subtract_which} must be either 'mean' or 'lowpass'.")
        mux_block_param = common_mode_params['mux_block']
        if 32 % mux_block_param != 0:
            raise ValueError(f"mux_block_param={mux_block_param} must divide 32 evenly.")
        freq_block_param = common_mode_params['freq_block']

        cm_savepath = os.path.join(save_path,'common_mode_plots')
        ts_savepath = os.path.join(save_path,'timestreams')
        os.makedirs(cm_savepath, exist_ok=True)
        os.makedirs(ts_savepath, exist_ok=True)
       

        #paramters used globally in this: 
        SPECTRAL_CHANNELS = (19,22,23,24,40)

        #make a list of xfs that's good so that we can put it into the df 

        #_,xf_good = self.get_and_sort_xf(fx_sort=True)
        #xf_good = set(map(tuple, xf_good))
        #xf_good_array = np.array([(i[0],i[1]) for i in self.header["xf_coords"]]) 
        
        # making some common mode files to keep track of the common mode removal process 
        # one file to save right from beginning the detector common mode metadata 
        cm_df_columns = ['detector_x','detector_f','detector_c','detector_r','tod_idx','is_used_to_make_template',
                                      'which_mux_block','which_freq_block',
                                      'mux_correlation_with_template','freq_correlation_with_template',
                                      'mux_scale_factor_base','freq_scale_factor_base',
                                      'mux_scale_factor','freq_scale_factor',
                                      'tod_normalization_factor_initial','tod_normalization_factor_interm']
        
        cm_df = pd.DataFrame(index=np.arange(len(self.header["xf_coords"])),columns=cm_df_columns)
     
        print('\tassigning detectors to a mux template and a frequency template')

        for i, xf in enumerate(self.header["xf_coords"]):
            x,f = xf
            c,r = coordinates.xf_to_muxcr(x,f)
            c = int(c); r = int(r)
            #in_xf_good = (int(x), int(f)) in xf_good
            is_spec = f in SPECTRAL_CHANNELS
            is_used = not is_spec
            #compute which mux block and frequency block this detector belongs to
            which_mux_block = c // mux_block_param
            #now also for the frequency block template 
            which_freq_block = np.searchsorted(freq_block_param, f, side="right") - 1

            cm_df.iloc[i] = [x,f,c,r,i,bool(is_used), 
                                which_mux_block, which_freq_block] + [np.nan]*(len(cm_df_columns) - 8) #placeholder values for the rest of the columns that will be filled in later during the common mode removal process.

        
        if debugging_plots:
            print('\t\tplotting the mux and frequency blocks for the detectors')
            from matplotlib.colors import ListedColormap

            #x_all = np.arange(16)
            #f_all = np.arange(60)

            #plot_mux = (cm_df.pivot(index="detector_x", columns="detector_f", values="which_mux_block").reindex(index=x_all, columns=f_all).astype(float).values)
            #plot_freq = (cm_df.pivot(index="detector_x", columns="detector_f", values="which_freq_block").reindex(index=x_all, columns=f_all).astype(float).values)
            
            plot_mux  = np.full((16, 60), np.nan, dtype=float)
            plot_freq = np.full((16, 60), np.nan, dtype=float)

            xs = cm_df["detector_x"].to_numpy(dtype=int)
            fs = cm_df["detector_f"].to_numpy(dtype=int)

            plot_mux[xs, fs]  = cm_df["which_mux_block"].to_numpy(dtype=float)
            plot_freq[xs, fs] = cm_df["which_freq_block"].to_numpy(dtype=float)

            #only plot colors for the detectors with is_good flag true by setting the plotting array to nan for bad detectors 
            colors32 = [
            '#1f77b4','#669fce','#aec7e8','#ff7f0e','#ff9d43','#ffbb78',
            '#2ca02c','#62c05b','#98df8a','#d62728','#eb605f','#ff9896',
            '#9467bd','#ac8cc9','#c5b0d5','#8c564b','#a87970','#c49c94',
            '#e377c2','#ed96ca','#f7b6d2','#7f7f7f','#a3a3a3','#c7c7c7',
            '#bcbd22','#cccc58','#dbdb8d','#17becf','#5accda','#9edae5']

            colors_plot = colors32[:32//mux_block_param]
            cmap = ListedColormap(colors_plot)

            fig, ax = plt.subplots(figsize=(15,6))

            im = ax.imshow(plot_mux, origin='lower', aspect='auto', cmap=cmap)

            ax.set_xticks(np.arange(-0.5, 60, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

            cbar = plt.colorbar(im, ax=ax, ticks=np.arange(32//mux_block_param))
            cbar.set_label('Mux CM Template Index')

            ax.set_xlabel('f coordinate')
            ax.set_ylabel('x coordinate')
            ax.set_title('Detector to Mux CM Template Mapping')

            plt.tight_layout()
            plt.savefig(os.path.join(cm_savepath,'detector_to_muxc_template_mapping.png'))
            plt.close()


            colors_plot = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
            cmap = ListedColormap(colors_plot[:len(freq_block_param)-1])

            fig, ax = plt.subplots(figsize=(15,6))

            im = ax.imshow(plot_freq, origin='lower', aspect='auto', cmap=cmap)

            ax.set_xticks(np.arange(-0.5, 60, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

            cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(freq_block_param)-1))
            cbar.set_label('Frequency CM Template Index')

            ax.set_xlabel('f coordinate')
            ax.set_ylabel('x coordinate')
            ax.set_title('Detector to Frequency CM Template Mapping')

            plt.tight_layout()
            plt.savefig(os.path.join(cm_savepath,'detector_to_freq_template_mapping.png'))
            plt.close()


        print('\tnormalizing the data by the standard deviation of each scan to put them on the same footing for the common mode template construction')
        good_tods = self.data.copy() 

        # if subtract_which == 'mean':
        #     print('\t\tsubtracting the mean from each TOD before making the common mode templates')
        #     good_tods = good_tods - np.nanmean(good_tods, axis=1)[:,None]
        # if subtract_which == 'lowpass':
        #     print('\t\tsubtracting a lowpass of 0.01Hz from each TOD before making the common mode templates')
        #     rff = np.fft.rfftfreq(len(self.t),d=0.01)
        #     lowpass = np.fft.irfft(np.fft.rfft(good_tods-(good_tods[:,-1]-good_tods[:,0])[:,None]/len(self.t)*np.arange(len(self.t))[None,:],axis=-1)*np.where(rff>0.01,np.exp(-(rff-0.01)**2/0.08**2),1)[None,:],axis=-1)+(good_tods[:,-1]-good_tods[:,0])[:,None]/len(self.t)*np.arange(len(self.t))[None,:]
        #     good_tods = good_tods - lowpass
            #np.savez(os.path.join(ts_savepath,'good_tods_before_subtracting_lowpass.npz'),t=self.t,good_tods=good_tods)


        tod_normalization_factor_initial = strength_param * np.nanstd(good_tods,axis=-1)[:,None]
        cm_df['tod_normalization_factor_initial'] = tod_normalization_factor_initial.flatten()
        good_tods_normed = good_tods / tod_normalization_factor_initial
        #print(good_tods_normed.shape)

        print('\tplotting the correlation matrix before common mode template subtraction')
        

        order_c_r = cm_df.sort_values(["detector_c", "detector_r"]).index.to_numpy(dtype=int)
        order_r_c = cm_df.sort_values(["detector_r", "detector_c"]).index.to_numpy(dtype=int)
        order_f_x = cm_df.sort_values(["detector_f", "detector_x"]).index.to_numpy(dtype=int)
        order_x_f = cm_df.sort_values(["detector_x", "detector_f"]).index.to_numpy(dtype=int)


        corr = np.corrcoef(good_tods_normed)
        corr_c_r = corr[np.ix_(order_c_r, order_c_r)]
        corr_r_c = corr[np.ix_(order_r_c, order_r_c)]
        corr_f_x = corr[np.ix_(order_f_x, order_f_x)]
        corr_x_f = corr[np.ix_(order_x_f, order_x_f)]

        if debugging_plots:
            print('\t\tdoing a few sanity checks to make sure the ordering of the cm_df matches the ordering of the TODs and the header info')

            # each row should be a unique detector
            assert cm_df[["detector_x","detector_f"]].duplicated().sum() == 0, "Duplicate (x,f) rows found in cm_df"

            # cm_df (x,f) ordering matches header['xf_coords'] ordering
            xf_header = np.array([(int(x), int(f)) for (x, f) in self.header["xf_coords"]], dtype=int)
            xf_df = cm_df[["detector_x","detector_f"]].to_numpy(dtype=int)
            assert np.array_equal(xf_df, xf_header), \
                "cm_df (x,f) rows do NOT match header['xf_coords'] order"

            print("\t\t\t(c,r) sort head:")
            print(cm_df.loc[order_c_r[:10], ["detector_x","detector_f","detector_c","detector_r"]])
            print("\t\t\t(c,r) sort tail:")
            print(cm_df.loc[order_c_r[-10:], ["detector_x","detector_f","detector_c","detector_r"]])

            print("\t\t\t(f,x) sort head:")
            print(cm_df.loc[order_f_x[:10], ["detector_x","detector_f","detector_c","detector_r"]])
            print("\t\t\t(f,x) sort tail:")
            print(cm_df.loc[order_f_x[-10:], ["detector_x","detector_f","detector_c","detector_r"]])

        
        fig, axes = plt.subplots(2, 2, figsize=(10,8), constrained_layout=True)

        im00 = axes[0,0].imshow(corr_c_r, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
        axes[0,0].set_title("Sorted by mux c then mux r")

        im01 = axes[0,1].imshow(corr_r_c, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
        axes[0,1].set_title("Sorted by mux r then mux c")

        im10 = axes[1,0].imshow(corr_f_x, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
        axes[1,0].set_title("Sorted by f then x")

        im11 = axes[1,1].imshow(corr_x_f, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
        axes[1,1].set_title("Sorted by x then f")

        cbar = fig.colorbar(im11, ax=axes[:, 1], location='right', shrink=0.95, pad=0.02)
        cbar.set_label("Correlation coefficient")

        plt.savefig(os.path.join(cm_savepath, 'pairwise_corr_matrix_before.jpg'), dpi=120)
        plt.close()

        print('\tconstructing the common mode templates')
        print('\t\tconstructing mux-column block templates')
        #doing mux block first 
        template_per_muxc = []
        for mux_block in range(32 // mux_block_param):
            block_detectors = cm_df[cm_df['which_mux_block'] == mux_block]
            #only detectors with is_used flag true should be used to make the template
            block_detectors_make_template = block_detectors[block_detectors['is_used_to_make_template']]

            #print('block_detectors.index.to_numpy(dtype=int)', block_detectors.index.to_numpy(dtype=int))
            block_tods = good_tods_normed[block_detectors.index.to_numpy(dtype=int)]
            block_tods_make_template = good_tods_normed[block_detectors_make_template.index.to_numpy(dtype=int)]
            template = np.nanmedian(block_tods_make_template, axis=0)
            template = np.nan_to_num(template, nan=0.0) #because the templates get subtracted, it's okay to replace the nans with zeros cuz then that just mean nothing get subtracted
            template_per_muxc.append(template)

            cm_df.loc[block_detectors.index, 'mux_correlation_with_template'] = [np.corrcoef(block_tods[i], template)[0,1] for i in range(len(block_detectors))]
        
            cm_df.loc[block_detectors.index, 'mux_scale_factor_base'] = self.corrcoef_to_scaling(cm_df.loc[block_detectors.index, 'mux_correlation_with_template'])
            scaling_base = cm_df.loc[block_detectors.index, 'mux_scale_factor_base'].to_numpy(dtype=float)
            assert np.all(np.isfinite(scaling_base)), 'the scaling BASE has some nans!!!!'
            
            # choose exponent based on is_used flag (is_used is a channel that's not a line channel!)
            exponent = np.where(cm_df.loc[block_detectors.index, 'is_used_to_make_template'].to_numpy(),1.0,0.25)
            scaling = np.sign(scaling_base) * np.abs(scaling_base) ** exponent
            assert np.all(np.isfinite(scaling)), 'the scaling FACTOR has some nans!!!!'
            cm_df.loc[block_detectors.index, 'mux_scale_factor'] = scaling

        if debugging_plots:
            print('\t\tplotting the mux block templates')
            self.plot_commonmode_template(templates = template_per_muxc, pixel = pixel, savepath = cm_savepath, template_kind = 'mux')
        
        
        print('\tsubtracting the mux block templates') 
        if debugging_plots:
             print(f'\tand plotting the tods before the mux block subtraction and the template')
        good_tods_normed_interim = good_tods_normed.copy() 
        assert np.all(np.isfinite(good_tods_normed_interim)), 'the tods before mux block subtraction has nans'
        for mux_block in range(32 // mux_block_param):
            block_detectors = cm_df[cm_df['which_mux_block'] == mux_block]
            block_indices = block_detectors.index.to_numpy(dtype=int)
            block_tods = good_tods_normed[block_indices]
            template = template_per_muxc[mux_block]
            #if mux_block in [1,3,4,9,10,16,17]: 
            #    np.savez(os.path.join(ts_savepath,f'mux_block_{mux_block}_tods_and_template_before_subtraction.npz'), block_tods=block_tods, template=template)
            scale_factors = cm_df.loc[block_indices, 'mux_scale_factor'].to_numpy(dtype=float)
            #print(block_tods.shape) #shaped (num_detectors_in_block, num_time_samples)
            #print(scale_factors.shape) #shaped (num_detectors_in_block,)
            #print(template.shape) #shaped (num_time_samples,)
            scaled_template = scale_factors[:, None] * template[None, :]
            #print(scaled_template.shape) #shaped (num_detectors_in_block, num_time_samples)
            #print('-----------')
            #assert (sanity check) that the first scaling factor times the first template gets you scaled template for the first detector:
            for i in range(len(block_indices)):
                assert np.array_equal(scaled_template[i], scale_factors[i] * template), f"sanity check: scaled template for detector index {i} doesn't match scaling factor times template"
            #assert that there is nothing in block tods, scaled template, or block_tods_subtracted that is nan (there shouldn't be because we replaced nans in the template with zeros and the scaling factors should be finite (and in fact between -1 and 1))
            assert np.all(np.isfinite(block_tods)), f"block_tods has some nans or infs for mux block {mux_block}!"
            assert np.all(np.isfinite(scaled_template)), f"scaled_template has some nans or infs for mux block {mux_block}!"
            block_tods_subtracted = block_tods - scaled_template
            assert np.all(np.isfinite(block_tods_subtracted)), f"block_tods_subtracted has some nans or infs for mux block {mux_block}!"
            good_tods_normed_interim[block_indices] = block_tods_subtracted

            if debugging_plots:
                nrows = len(block_indices) + 1
                #if tod - template  = all zeros for that template, color the plot's background red 
                fig, axes = plt.subplots(nrows, 2,figsize=(10, 2*nrows),sharex='col',sharey='row',squeeze=False)
                fig.suptitle(f'Mux Block {mux_block} | TODs before subtraction and template', fontsize=16)
                for i in range(nrows):
                    if i < len(block_indices):
                        if np.all(block_tods_subtracted[i] == 0.0) and not block_detectors.iloc[i]['is_used_to_make_template']:
                            background_color = '#ff6961'
                        elif np.all(block_tods_subtracted[i] == 0.0): 
                            background_color = '#ffe6ee'
                        elif not block_detectors.iloc[i]['is_used_to_make_template']:
                            background_color = '#fdfd96'
                        else: 
                            background_color = 'white'
                    else:
                        background_color = 'white'
                    
                    ax = axes[i, 0]#all the tods
                    ax.set_facecolor(background_color)
                    if i < len(block_indices):
                        ax.plot(block_tods[i], label=f'detector x={block_detectors.iloc[i]["detector_x"]} f={block_detectors.iloc[i]["detector_f"]}')
                    else:
                        ax.plot(template, label='mux block template', color='black', linewidth=2)

                    #ax.legend()

                    ax = axes[i, 1] #zoomed in 
                    ax.set_facecolor(background_color)

                    if i < len(block_indices):
                        ax.plot(block_tods[i], label=f'detector x={block_detectors.iloc[i]["detector_x"]} f={block_detectors.iloc[i]["detector_f"]}')
                    else:
                        ax.plot(template, label='mux block template', color='black', linewidth=2)

                    ax.set_xlim(10000, 14000)
                    ax.legend()

                plt.tight_layout()
                mux_c_ts_savepath = os.path.join(ts_savepath,'mux_c_block')
                if not os.path.exists(mux_c_ts_savepath):
                    os.makedirs(mux_c_ts_savepath)
                plt.savefig(os.path.join(mux_c_ts_savepath,f'mux_block_{mux_block}_tods_and_template_before_subtraction.png'))
                plt.close()

        

        print('\t\tcheck if the detectors with all zeros after mux block subtraction correspond to those with a correlation to template of 1.0')
        #print(cm_df.index)
        all_zero_after_mux_subtraction = (good_tods_normed_interim == 0.0).all(axis=-1)
        correlation_one = (cm_df['mux_correlation_with_template'].to_numpy() == 1.0)

        print(f'\t\t\t{correlation_one.sum()} detectors have a correlation of exactly 1.0 to the mux template')
        print(f"\t\t\t{all_zero_after_mux_subtraction.sum()} detectors have all zeros after mux block subtraction")
        print(f"\t\t\tout of those, {(all_zero_after_mux_subtraction & correlation_one).sum()} detectors have a correlation of exactly 1.0 to the mux template")
        print("\t\t\tare there any detectors that have all zeros after mux block subtraction but do NOT have a correlation of exactly 1.0 to the mux template?")
        

        for i in range(len(all_zero_after_mux_subtraction)): 
            if all_zero_after_mux_subtraction[i] and not correlation_one[i]:
                x = cm_df.iloc[i]['detector_x']
                f = cm_df.iloc[i]['detector_f']
                corr = cm_df.iloc[i]['mux_correlation_with_template']
                print(f"\t\t\t\tdetector (x={x}, f={f}) | corr={corr} | all_zero={all_zero_after_mux_subtraction[i]}")
                
        print('\t\t\tprinting out the metadata for the detectors that have all zeros after mux block subtraction to see if there is any pattern to them')
        for i in range(len(all_zero_after_mux_subtraction)):
            if all_zero_after_mux_subtraction[i]:
                x = cm_df.iloc[i]['detector_x']
                f = cm_df.iloc[i]['detector_f']
                c = cm_df.iloc[i]['detector_c']
                r = cm_df.iloc[i]['detector_r']
                which_mux_block = cm_df.iloc[i]['which_mux_block']
                #get how many other detectors are in the same mux block as this one:
                num_detectors_in_mux_block = len(cm_df[cm_df['which_mux_block'] == which_mux_block])
                #how many other detectors in the same mux block contribute in making the template 
                num_contributing_detectors_in_mux_block = len(cm_df[(cm_df['which_mux_block'] == which_mux_block) & (cm_df['is_used_to_make_template'])])

                print(f"\t\t\t\tdetector (x={x}, f={f}) | c={c} | r={r} | mux_block={which_mux_block} | num_detectors_in_mux_block={num_detectors_in_mux_block} | num_contributing_detectors_in_mux_block={num_contributing_detectors_in_mux_block}")
            
        #exit()
        assert np.all(np.isfinite(good_tods_normed_interim)), 'the tods after mux block subtraction has nans'

        print('\tnormalizing the tod one more time after mux block subtraction before making the frequency block templates')
        tod_normalization_factor_interm = strength_param * np.nanstd(good_tods_normed_interim,axis=-1)[:,None]
        cm_df['tod_normalization_factor_interm'] = tod_normalization_factor_interm.flatten()
        good_tods_normed_interim /= tod_normalization_factor_interm
        assert np.all(np.isfinite(tod_normalization_factor_interm)), 'the interim normalization factor has nans'
        assert np.all(np.isfinite(good_tods_normed_interim)), 'the tods after interium normalization has nans'
        
        print('\tconstructing frequency block templates')
        template_per_freq_block = []
        for freq_block_index in range(len(freq_block_param)-1):
            block_detectors = cm_df[cm_df['which_freq_block'] == freq_block_index]
            block_detectors_make_template = block_detectors[block_detectors['is_used_to_make_template']]
            block_tods = good_tods_normed_interim[block_detectors.index.to_numpy(dtype=int)]
            block_tods_make_template = good_tods_normed_interim[block_detectors_make_template.index.to_numpy(dtype=int)]
            template = np.nanmedian(block_tods_make_template, axis=0)
            template = np.nan_to_num(template, nan=0.0) #because the templates get subtracted, it's okay to replace the nans with zeros cuz then that just mean nothing get subtracted
            template_per_freq_block.append(template)

            cm_df.loc[block_detectors.index, 'freq_correlation_with_template'] = [np.corrcoef(block_tods[i], template)[0,1] for i in range(len(block_detectors))]
        
            cm_df.loc[block_detectors.index, 'freq_scale_factor_base'] = self.corrcoef_to_scaling(cm_df.loc[block_detectors.index, 'freq_correlation_with_template'])
            scaling_base = cm_df.loc[block_detectors.index, 'freq_scale_factor_base'].to_numpy(dtype=float)
            assert np.all(np.isfinite(scaling_base)), 'the frequency block scaling BASE has some nans!!!!'
            
            # choose exponent based on is_used flag (is_used is a channel that's not a line channel!)
            exponent = np.where(cm_df.loc[block_detectors.index, 'is_used_to_make_template'].to_numpy(),1.0,0.25)
            scaling = np.sign(scaling_base) * np.abs(scaling_base) ** exponent
            assert np.all(np.isfinite(scaling)), 'the frequency block scaling FACTOR has some nans!!!!'
            cm_df.loc[block_detectors.index, 'freq_scale_factor'] = scaling

        if debugging_plots:
            print('\t\tplotting the frequency block templates')
            self.plot_commonmode_template(templates = template_per_freq_block, pixel = pixel, savepath = cm_savepath, template_kind = 'freq')

        print('\tsubtracting the frequency block templates')
        if debugging_plots:
            print(f'\tand plotting the tods before the freq block subtraction and the template')
        
        for freq_block_index in range(len(freq_block_param)-1):
            block_detectors = cm_df[cm_df['which_freq_block'] == freq_block_index]
            block_indices = block_detectors.index.to_numpy(dtype=int)
            block_tods = good_tods_normed_interim[block_indices]
            template = template_per_freq_block[freq_block_index]
            for i in range(len(block_indices)):
                assert np.array_equal(block_tods[i], good_tods_normed_interim[block_indices[i]]), f"sanity check: block_tods for detector index {i} doesn't match good_tods_normed_interim for that detector index"
            assert np.all(np.isfinite(block_tods)), f"block_tods has some nans or infs for frequency block {freq_block_index}!"
            assert np.all(np.isfinite(template)), f"template has some nans or infs for frequency block {freq_block_index}!"
            scale_factors = cm_df.loc[block_indices, 'freq_scale_factor'].to_numpy(dtype=float)
            scaled_template = scale_factors[:, None] * template[None, :]
            block_tods_subtracted = block_tods - scaled_template
            assert np.all(np.isfinite(block_tods_subtracted)), f"block_tods_subtracted has some nans or infs for frequency block {freq_block_index}!"
            good_tods_normed_interim[block_indices] = block_tods_subtracted

            if debugging_plots:
                nrows = len(block_indices) + 1
                fig, axes = plt.subplots(nrows, 2,figsize=(10, 2*nrows),sharex='col',sharey='row',squeeze=False)
                fig.suptitle(f'Frequency Block {freq_block_index} | TODs before subtraction and template', fontsize=16)
                for i in range(nrows):
                    if i < len(block_indices):
                        ax = axes[i, 0]
                        ax.plot(block_tods[i], label=f'detector x={block_detectors.iloc[i]["detector_x"]} f={block_detectors.iloc[i]["detector_f"]}')
                    else:
                        ax = axes[i, 0]
                        ax.plot(template, label='frequency block template', color='black', linewidth=2)

                    #ax.legend()

                    ax = axes[i, 1] #zoomed in 
                    if i < len(block_indices):
                        ax.plot(block_tods[i], label=f'detector x={block_detectors.iloc[i]["detector_x"]} f={block_detectors.iloc[i]["detector_f"]}')
                    else:
                        ax.plot(template, label='frequency block template', color='black', linewidth=2)

                    ax.set_xlim(10000, 14000)
                    ax.legend()

                plt.tight_layout()
                freq_block_ts_savepath = os.path.join(ts_savepath,'freq_block')
                if not os.path.exists(freq_block_ts_savepath):
                    os.makedirs(freq_block_ts_savepath)
                plt.savefig(os.path.join(freq_block_ts_savepath,f'freq_block_{freq_block_index}_tods_and_template_before_subtraction.png'))
                plt.close()

        assert np.all(np.isfinite(good_tods_normed_interim)), 'the tods after frequency block subtraction has nans'


        print('\t\tplotting the correlation matrix after common mode template subtraction')
        corr_after = np.corrcoef(good_tods_normed_interim)
        corr_c_r_after = corr_after[np.ix_(order_c_r, order_c_r)]
        corr_r_c_after = corr_after[np.ix_(order_r_c, order_r_c)]
        corr_f_x_after = corr_after[np.ix_(order_f_x, order_f_x)]
        corr_x_f_after = corr_after[np.ix_(order_x_f, order_x_f)]
        if debugging_plots:
            fig, axes = plt.subplots(2, 2, figsize=(10,8), constrained_layout=True)

            im00 = axes[0,0].imshow(corr_c_r_after, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
            axes[0,0].set_title("Sorted by mux c then mux r")

            im01 = axes[0,1].imshow(corr_r_c_after, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
            axes[0,1].set_title("Sorted by mux r then mux c")

            im10 = axes[1,0].imshow(corr_f_x_after, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
            axes[1,0].set_title("Sorted by f then x")

            im11 = axes[1,1].imshow(corr_x_f_after, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
            axes[1,1].set_title("Sorted by x then f")

            cbar = fig.colorbar(im11, ax=axes[:, 1], location='right', shrink=0.95, pad=0.02)
            cbar.set_label("Correlation coefficient")

            plt.savefig(os.path.join(cm_savepath,'pairwise_corr_matrix_after.jpg'), dpi=120)
            plt.close()
        
        print('\trestoring the overall normalization factors to the tods after common mode subtraction so that they are on the same footing as the original tods for any downstream analysis')
        good_tods_final = good_tods_normed_interim * tod_normalization_factor_initial * tod_normalization_factor_interm

        self.data = good_tods_final
        
        #save the cm_df stuff to a csv so that we can look at it 
        cm_df.to_csv(os.path.join(cm_savepath,'detector_common_mode_metadata.csv'),index=False)

    def plot_commonmode_template(self,templates,pixel,savepath,template_kind = 'mux'):
        """
        fold then plot the common mode template per mux-column block for visualization. 

        template_kind: str: 'mux' for mux-column block template, 'freq' for full array template

        """ 

        ra = self.ra
        dec = self.dec 

        if len(np.array(pixel,ndmin=1)) > 1:
            x_pixel = pixel[0]
            y_pixel = pixel[1]
        else:
            x_pixel = pixel
            y_pixel = pixel

        rabins = np.arange(np.nanmin(ra),np.nanmax(ra),x_pixel)
        decbins = np.arange(np.nanmin(dec),np.nanmax(dec),y_pixel)
        #ractrs = 0.5*(rabins[:-1]+rabins[1:])
        #decctrs = 0.5*(decbins[:-1]+decbins[1:])
        
        unweighted_hist, _, _ = np.histogram2d(ra, dec, bins=(rabins, decbins)) 
        N_templates = np.asarray(templates).shape[0]

        fig, axes = plt.subplots(8, 4, figsize=(8*4, 4*3), constrained_layout=True)
        axes_flat = axes.ravel()

        for i in range(N_templates):
            test_template = templates[i]

            weighted_hist, _, _ = np.histogram2d(
                ra,
                dec,
                weights=test_template,
                bins=(rabins, decbins)
            )
            ratio = np.divide(weighted_hist, unweighted_hist) 

            ax = axes_flat[i]

            pc = ax.pcolormesh(
                rabins,
                decbins,
                ratio.T
            )

            ax.set_aspect(1.5)
            if template_kind == 'mux':
                ax.set_title(f"MUX {i}")
            elif template_kind == 'freq':
                ax.set_title(f"FREQ BLOCK {i}")
            fig.colorbar(pc, ax=ax, orientation='vertical')

        plt.suptitle("COMMON MODE TEMPLATES", fontsize=16)
        plt.savefig(savepath+'/'+template_kind+'_common_mode_templates.png',dpi=300,bbox_inches='tight')

        plt.close()

    def corrcoef_to_scaling(self,r,a=20,b=6):
        """
        Map correlation coefficient(s) to a signed scaling factor using a sigmoid. 

        r : Correlation coefficient(s).
        
        a : optional, controls the steepness of the transition (default: 20).
        Larger a -> sharper threshold.
        
        b : optional, controls the horizontal shift of the transition (default: 6).
        Larger b -> threshold occurs at larger |x|.
        """
        r = np.asarray(r, dtype=float)
        return np.sign(r) * (erf(a * np.abs(r) - b) / 2 + 0.5) 