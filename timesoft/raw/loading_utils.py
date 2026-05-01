"""Utilities for loading data
"""


from timesoft.helpers import coordinates

import os
import itertools

import numpy as np
from scipy import interpolate as interp
from astropy.coordinates import Angle
from astropy.time import Time
import astropy.units as u

from netCDF4 import MFDataset
import json
pi = np.pi
import warnings
warnings.resetwarnings()

nchan = 60
ndet = 16

############################################
#### HELPER FUNCTIONS FOR VARIOUS TASKS ####
############################################

# Timing fix for oldest data
def _sync2utc(sync, ref_sync, ref_epoch_local, freq=100.0, utc_offset=-7):
    t0 = ref_epoch_local - (ref_sync / freq)
    t = sync / freq + t0
    t -= utc_offset * 60 * 60
    return t

def _get_files(path, hk_path, mce_files_in, hk_files_in, version, type=None, skip_last=True):
    """Helper function to get the desired files

    Given a path to a directory and a list of filenames from that directory
    this code will identify the netCDF files and construct
    a list of absolute paths to pass to the netCDF data
    loader. It aslo ensures that the path exists. Filenames
    not found in the directory will be ignored
    
    Parameters
    ----------
    path : str
        Path to direcotry containing desired data
    files_in : list of strs
        Can be used to restrict the files loaded to only those listed. If set 
        as None, all netcdf files in the path will be loaded.

    Returns
    -------
    files : list
        A list of absolute file paths to all .nc files to be loaded.
    """

    # Load data
    if mce_files_in is not None:
        if type == 'nc' :
            files = [os.path.join(path, i) for i in mce_files_in if i.endswith(".nc") and i.startswith("raw_mce")]
        else :
            files = [os.path.join(path, i) for i in mce_files_in if i.endswith(".run")]
        if hk_path is not None:
            hk_files = [os.path.join(hk_path, i) for i in hk_files_in if i.endswith(".nc") and i.startswith("raw_hk")]
    else:
        if type == 'nc' :
            files = [os.path.join(path, i) for i in os.listdir(path) if i.endswith(".nc") and i.startswith("raw_mce")]
        else : 
            files = [os.path.join(path, i) for i in os.listdir(path) if i.endswith(".run")]
        if hk_path is not None:
            hk_files = [os.path.join(hk_path, i) for i in os.listdir(hk_path) if i.endswith(".nc") and i.startswith("raw_hk")]

    files = sorted(files)

    # If it's been more than 150 seconds since a file was created, assume it's safe to load
    if skip_last:
        # Has to be a smarter way to do this, but I'm tired - get the unix time of the last scan:
        last_scan = int(files[-1].split('/')[-1].split('.')[0].split('_')[-1])
        if (Time.now().unix - last_scan) > 150:
            skip_last = False

    if skip_last:
        files = files[:-1]

    if version in ['2026.dev.0','2024.dev.1',]:
        if hk_files is not None:
            hk_files = sorted(hk_files)
            if skip_last:
                hk_files = hk_files[:-1]
            return files, hk_files
    else :
        return files


#####################
#### DATA LOADER ####
#####################

def get_data(path, hk_path = None, mc=0, version='2026.dev.0', files_in=None, xf=None, cr=None, raisenotel=False, skip_last=True):
    """Load and pre-process raw data form netCDF files

    This function loads raw data from netCDF files, performs the necessary 
    interpolations to make the sampling of telescope telemetry and flagging 
    match the sampling of the detector data, and then returns timestreams 
    of telemetry, flagging, and detector data.

    Parameters
    ----------
    path : str
        Path to direcotry containing desired data
    mc : {0, 1}, default=0 
        Determines which mce to load
    version : {'2026.dev.0','2024.dev.1','2022.dev.1', '2022.dev.0', '2019.final', '2019.dev.1', '2019.dev.0'}, optional
        Specifies which version of the raw data file structure is in use.
        The default is '2026.dev.0' which works for most data taken during 
        the 2026 run. Other versions may be needed for older data:
        '2024.dev.1' mostly vestigial, 2026.dev.0 should be fine
        '2022.dev.1' for most 2022 data;
        '2022.dev.0' for a few very early observations from the 2022 run; 
        '2019.final' for data from 2019-03-15 to end of 2019 engineering run;
        '2019.dev.1' for data from 2019-03-14 or earlier; and '2019.dev.0' for
        very first data from 2019 run
    files_in : list of strs, optional
        Can be used to restrict the files loaded to only those listed. If not
        specified, all netcdf files in the path will be loaded.
    xf : list of tuples, optional
        Can be used to specify the detectors to load in terms of the xf coordinates.
        If specified, it should be a list of xf pairs e.g. [(1,1), (15,2), (12,29)], and
        only data for those detectors will be returned. If not specified data for all 
        detectors of a given MCE will be returned. Only one of xf or cr can be used
        (the other should be set as None or left out of the function call).
    cr : list of tuples, optional
        Can be used to specify the detectors to load in terms of the muxcr coordinates.
        If specified, it should be a list of muxcr pairs, and
        only data for those detectors will be returned. If not specified data for all 
        detectors of a given MCE will be returned. Only one of xf or cr can be used
        (the other should be set to None or left out of the function call).
    raisenotel : bool, default=True
        Determines the handling of files with no valid telescope telemetry data.
        If `raisenotel=True`, a ValueError will be raised. If `raisenotel=False`
        no error will be raised, but the returned telescope data (ra, dec, az, el,
        and flags) will be arrays of filled with `np.nan`.

    Returns
    -------
    time : array
        Timestamps for each point in the timestream
    ra : array
        Telescope pointing R.A. values for each point in the timestream
    dec : array
        Telescope pointing declination values for each point in the timestream
    az : array
        Telescope pointing azimuth values for each point in the timestream
    elevation : array
        Telescope pointing elevation values for each point in the timestream
    data : array
        Data from each detector at each point in the timestream. This is given
        as a 2D array with the first index referring to a detector and the second
        index referring to a specific point in the timestream. The order of
        detectors is either by increasing x then f, or, if `xf` is specified,
        matches the input order
    flag : array
        Telescope flag values for each point in the timestream
    tau : array
        Atmospheric opacity values
    header : dict
        A dictionary containing metadata for the observation.

    Notes
    -----

    The data is returned as a 2D array of N_detectors x N_points. 
    If you have specified a set of detectors (ie used the xf argument), then 
    the order of detectors will match the order specified. Otherwise it will 
    be `xf = (0,0), (0,1), ... (0,15), (1,0), ...`. You can use 
    `val.reshape(16,60,-1)` to reshape the `val` array so that the first two 
    indices correspond to x and f respectively.

    The header information differs from observation to observation, 
    particularly for different versions of the file, but also depending
    whether there is telescope data available for a given dataset.

    **To Do**

    If you need a particular timestream added to the data retrieval system
    please contact Ryan Keenan about how to modify this code.

    We can also consider adding options to return additional telescope, 
    housekeeping, and/or KMS timestreams.
    """
    # Check version
    if version not in ['2026.dev.0','2024.dev.1','2022.dev.2','2022.dev.1','2022.dev.0','2019.final','2019.dev.1','2019.dev.0']:
        raise ValueError('Version not recognized')

    # Make sure path exists
    if not os.path.exists(path):
        raise ValueError('file not found', path)

    # Check that a maximum of one of xf and cr have been provided
    if xf is not None and cr is not None:
        raise ValueError('cannot specify both xf and cr indexing')

    # Get file names
    if version in ['2026.dev.0','2024.dev.1']:
        files, hk_files = _get_files(path,hk_path,None,None, version,type='nc',skip_last=skip_last)
    else: 
        files = _get_files(path,None,None,None,version,type='nc',skip_last=skip_last)

    # For data where telescope data wasn't appended correctly, also find the telescope data
    if version == '2022.dev.0':
        tel_files = np.array([path + i for i in os.listdir(path) if i.endswith(".npy")])
        tel_nums = np.array([int(f.split('/')[-1].split('.')[-2].split('t')[-1])  for f in tel_files])
        tel_files = tel_files[np.argsort(tel_nums)]
        tel_data = np.array([np.load(f) for f in tel_files])

    # Generate MFDataset object
    data_file = MFDataset(files)
    if version in ['2026.dev.0','2024.dev.1']:
        hk_data_files = MFDataset(hk_files)

    # Generate header information:
    header = {}
    header['data_version'] = version
        
    # Version reading in .json header info
    if version in ['2026.dev.0','2024.dev.1']:
        with open(path / "init_params.json", 'r') as f:
            input = json.load(f)

        header = {'mc':mc,
            'observer':str(input['Observer']),
            'object':str(input['Object Catalog Name']),
            'datetime':None,
            'command_ra':Angle(input['Source Coord 1'], unit=u.hourangle).deg,
            'command_dec':Angle(input['Source Coord 2'], unit=u.degree).deg,
            'command_epoch':str(input['Epoch']),
            'detector_pars':{'datamode':str(input['Datamode']),
                'readout':str(input['Readout Card']),
                'live_detectors':None
                },
            'scan_pars':{'type':str(input['Scan Strategy']),
                'direction':str(input['Variable Coordinate']),
                'crossing_time':float(input['Scan Traversal Time (sec)']),
                'scan_width':float(input['1st Dimension (Scan Length)']),
                'map_angle_offset':float(input['Angle of Map Offset']),
                },
            'original_data_path':path,
            'original_files_in':files_in,
            'filesystem_version':version
            }
        
        if input['Activate KMS'] == 'Yes':
            header['kms_on'] = True
        else:
            header['kms_on'] = False
        if input['Scan Strategy'] == '1D Raster':
            header['scan_pars']['number_repeats'] = int(input['Number of Scans (1D Only)'])
        else:
            header['scan_pars']['number_repeats'] = 1
            
        # If 2d scan:
        if header['scan_pars']['type'] in ['2D Raster','LIM Scan']:
            header['scan_pars']['map_height'] = float(input['2nd Dimension (2D Only)'])
            header['scan_pars']['map_row_spacing'] = float(input['Size of 2D Vertical Step'])

    else:
        header = {'mc':mc,
            'observer':data_file.variables['observer'][:],
            'object':'unknown',
            'datetime':data_file.variables['datetime'][:],
            'command_ra':np.nan,
            'command_dec':np.nan,
            'command_epoch':'unknown',
            'detector_pars':{'datamode':data_file.variables['datamode'][:],
                'readout':data_file.variables['rc'][:],
                'live_detectors':data_file.variables['detector'][:]
                },
            'scan_pars':{'type':'unknown',
                'direction':'unknown',
                'crossing_time':np.nan,
                'scan_width':np.nan,
                'map_angle_offset':0,
                'number_repeats':np.nan,
                },
            'original_data_path':path,
            'original_files_in':files_in,
            'filesystem_version':version
            }
        header['kms_on'] = True # Not actually known

        # If 2d scan:
        if header['scan_pars']['type'].lower() in ['2d','2d raster''unknown']:
            header['scan_pars']['map_height'] = np.nan
            header['scan_pars']['map_row_spacing'] = np.nan

    # Extract sync time from MCE header
    nvar = 17
    time_ind = 15
    time_get_inds = [time_ind + i for i in np.arange(100)*nvar]
    mce_sync_time = data_file.variables['mce'+str(mc)+'_header'][:,time_get_inds,0].flatten()
    mce_sync_time += np.int64(np.arange(len(mce_sync_time))%100)

    # Extract counts from MCE data
    mce_counts_all = data_file.variables['mce'+str(mc)+'_raw_data_all'][:]

    if cr is not None:
        if mc == 0:
            xf = [coordinates.muxcr_to_xf(cri[0],cri[1],p=mc) for cri in cr]
        elif mc == 1:
            xf = [coordinates.muxcr_to_xf_B(cri[0],cri[1]) for cri in cr]

    if xf is None:
        xf = [(x,f) for x,f in itertools.product(np.arange(16),np.arange(60))]
    xf_new = np.empty(len(xf), dtype=object)
    xf_new[:] = xf
    xf = xf_new

    # Extract time and pointing from telescope data - the indices have changed over time
    data = np.zeros((len(xf),len(mce_sync_time)))
    for i in range(len(xf)):
        if mc == 0:
            cr = coordinates.xf_to_muxcr(xf[i][0],xf[i][1],mc)
        elif mc == 1:
            cr = coordinates.xf_to_muxcr_B(xf[i][0],xf[i][1])
        data[i] = mce_counts_all[:,cr[1],cr[0],:].flatten()

        center_ra_ind = 7
        center_dec_ind = 8
        ra_ind = 9
        dec_ind = 10
        az_ind = 13
        el_ind = 14
        flag_ind = 1
    # Extract time and pointing from telescope data - the indices have changed over time
    data = np.zeros((len(xf),len(mce_sync_time)))
    for i in range(len(xf)):
        if mc == 0:
            cr = coordinates.xf_to_muxcr(xf[i][0],xf[i][1],mc)
        elif mc == 1:
            cr = coordinates.xf_to_muxcr_B(xf[i][0],xf[i][1])
        data[i] = mce_counts_all[:,cr[1],cr[0],:].flatten()

        center_ra_ind = 7
        center_dec_ind = 8
        ra_ind = 9
        dec_ind = 10
        az_ind = 13
        el_ind = 14
        flag_ind = 1
        tau_ind = 22
    if version == '2019.dev.0':
        time_ind = 4
    else:
        time_ind = 20

    if version == '2022.dev.0':
        telescope_time = tel_data[:,:,time_ind].flatten()
        telescope_center_ra = tel_data['tel'][:,:,center_ra_ind].flatten()
        telescope_center_dec = tel_data['tel'][:,:,center_dec_ind].flatten()
        telescope_ra = tel_data[:,:,ra_ind].flatten()
        telescope_dec = tel_data[:,:,dec_ind].flatten()
        telescope_az = tel_data[:,:,az_ind].flatten()
        telescope_el = tel_data[:,:,el_ind].flatten()
        telescope_flag = tel_data[:,:,flag_ind].flatten()
    else:
        telescope_time = data_file.variables['tel'][:,:,time_ind].flatten()
        telescope_center_ra = data_file.variables['tel'][:,:,center_ra_ind].flatten()
        telescope_center_dec = data_file.variables['tel'][:,:,center_dec_ind].flatten()
        telescope_ra = data_file.variables['tel'][:,:,ra_ind].flatten()
        telescope_dec = data_file.variables['tel'][:,:,dec_ind].flatten()
        telescope_az = data_file.variables['tel'][:,:,az_ind].flatten()
        telescope_el = data_file.variables['tel'][:,:,el_ind].flatten()
        telescope_flag = data_file.variables['tel'][:,:,flag_ind].flatten()
    
    # Older data doesn't have tau information - use zeros in that case
    if version in ['2026.dev.0','2024.dev.1','2022.dev.0', '2022.dev.1','2022.dev.1','2022.dev.0','2019.final','2019.dev.1','2019.dev.0']:
        telescope_tau = np.zeros(telescope_time.shape)
        header['has_tau_data'] = False
    else:
        telescope_tau = data_file.variables['tel'][:,:,tau_ind].flatten()
        header['has_tau_data'] = True

    if np.all(telescope_time == 0.0) or len(telescope_time)==0:
        if raisenotel:
            raise ValueError("No valid telescope data")
        usetel = False
    else:
        usetel = True
    header['has_tel_data'] = usetel

    if usetel:
        # This should be constant - get the first value and use that as the pointing center
        ## Actually it's not constant, updating it to be the median, only using if we can't 
        ## find a value in the init pars file
        header['center_ra'] = np.nanmedian(telescope_center_ra)
        header['center_dec'] = np.nanmedian(telescope_center_dec)
        header['epoch'] = 'apparent'

        # Generate pointing spline for telescope
        telescope_data = np.squeeze(np.dstack((telescope_time, telescope_ra, telescope_dec, telescope_az, telescope_el, telescope_flag, telescope_tau)))
        telescope_data = telescope_data[telescope_data[:,0]<1e10]
        telescope_data = telescope_data[telescope_data[:,0]>0]

        if version == '2019.dev.0':
            good_inds = np.where(telescope_data[1:,0]-telescope_data[:-1,0] > 0)[0]
            telescope_data = telescope_data[good_inds]

        ra_spline = interp.UnivariateSpline(telescope_data[:,0], telescope_data[:,1], s=0, ext=3)

        dec_spline = interp.UnivariateSpline(telescope_data[:,0], telescope_data[:,2], s=0, ext=3)
        az_spline = interp.UnivariateSpline(telescope_data[:,0], telescope_data[:,3], s=0, ext=3)
        el_spline = interp.UnivariateSpline(telescope_data[:,0], telescope_data[:,4], s=0, ext=3)
        flag_spline = interp.UnivariateSpline(telescope_data[:,0], telescope_data[:,5], s=0, ext=3)
        tau_spline = interp.UnivariateSpline(telescope_data[:,0], telescope_data[:,6], s=0, ext=3)
    else:
        header['center_ra'] = np.nan
        header['center_dec'] = np.nan
        header['epoch'] = 'unknown'


    # Get mapping from sync time to UTC in order to match telescope and detector data
    if version == '2019.dev.0':
        sync = data_file.variables['time'][:][:,1]
        sync = np.sort(sync)
        sync = sync[sync<1e10]
        times = (_sync2utc(sync, ref_sync=58782635, ref_epoch_local=1552238365.000, as_datetime=False) - 1552201200.0 - 1.08)/3600
        sync_time_spline = interp.UnivariateSpline(sync, times, s=0)
        mce_unix_time = sync_time_spline(mce_sync_time)
    elif version == '2019.dev.1':
        times = data_file.variables['time'][:]
        times = np.sort(times, axis=0)
        sync_time_spline = interp.UnivariateSpline(times[:,1], times[:,0], s=0)
        mce_unix_time = sync_time_spline(mce_sync_time)
    elif version == '2019.final':
        mce_unix_time = data_file.variables['time'][:,:,0].flatten()
    elif version in ['2022.dev.1','2022.dev.0']:
        to_spline_sync = data_file.variables['hk_time'][:,1,:].flatten()
        to_spline_unix = data_file.variables['hk_time'][:,0,:].flatten()
        inds = np.where((~np.isnan(to_spline_sync) & (to_spline_unix>0)))[0]
        to_spline_sync = to_spline_sync[inds]
        to_spline_unix = to_spline_unix[inds]
        to_spline_unix = to_spline_unix[np.argsort(to_spline_sync)]
        to_spline_sync = to_spline_sync[np.argsort(to_spline_sync)]
        sync_unix_spline = interp.UnivariateSpline(to_spline_sync, to_spline_unix, s=0)
        mce_unix_time = sync_unix_spline(mce_sync_time)
    # Later versions have hk in separate file:
    else:
        to_spline_sync = hk_data_files.variables['hk_time'][:,1,:].flatten()
        to_spline_unix = hk_data_files.variables['hk_time'][:,0,:].flatten()

        inds = np.where((~np.isnan(to_spline_sync) & (to_spline_unix>0)))[0]
        to_spline_sync = to_spline_sync[inds]
        to_spline_unix = to_spline_unix[inds]
        to_spline_unix = to_spline_unix[np.argsort(to_spline_sync)]
        to_spline_sync = to_spline_sync[np.argsort(to_spline_sync)]

        sync_unix_spline = interp.UnivariateSpline(to_spline_sync, to_spline_unix, s=0)
        mce_unix_time = sync_unix_spline(mce_sync_time)

    # sort everything by time
    sorter = np.argsort(mce_unix_time)

    if usetel:
        # Map MCE time to ra, dec, az, and el
        ra = ra_spline(mce_unix_time)[sorter]
        dec = dec_spline(mce_unix_time)[sorter]
        az = az_spline(mce_unix_time)[sorter]
        el = el_spline(mce_unix_time)[sorter]
        # Determine telescope state
        flag = flag_spline(mce_unix_time)[sorter]
        flag = np.around(flag).astype('int')
        tau = tau_spline(mce_unix_time)[sorter]
    else:
        ra = np.empty(sorter.shape)
        ra[:] = np.nan
        dec = np.copy(ra)
        az = np.copy(ra)
        el = np.copy(ra)
        flag = np.copy(ra)
        tau = np.copy(ra)

    values = data[:,sorter]
    time = mce_unix_time[sorter]

    return time, ra, dec, az, el, values, flag, tau, header

def get_data_lab(path, mc=0, files_in=None, xf=None, cr=None):
    """Load and pre-process raw lab data from netCDF files, creating using the DAQ GUI

    This function loads raw lab data from MCE files. This function should only be used if it is 
    uneccessary to synchronize timestamps across multiple devices, or read in any other type of data.
    The _get_files function has a flag that will read in the default MCE .run file format when using this 
    function.
    
    Parameters
    ----------
    path : str
        Path to direcotry containing desired data
    mc : {0, 1}, default=0 
        Determines which mce to load
    files_in : list of strs, optional
        Can be used to restrict the files loaded to only those listed. If not
        specified, all netcdf files in the path will be loaded.
    xf : list of tuples, optional
        Can be used to specify the detectors to load in terms of the xf coordinates.
        If specified, it should be a list of xf pairs e.g. [(1,1), (15,2), (12,29)], and
        only data for those detectors will be returned. If not specified data for all 
        detectors of a given MCE will be returned. Only one of xf or cr can be used
        (the other should be set as None or left out of the function call).
    cr : list of tuples, optional
        Can be used to specify the detectors to load in terms of the muxcr coordinates.
        If specified, it should be a list of muxcr pairs, and
        only data for those detectors will be returned. If not specified data for all 
        detectors of a given MCE will be returned. Only one of xf or cr can be used
        (the other should be set to None or left out of the function call).

    Returns
    -------
    data : array
        Data from each detector at each point in the timestream. This is given
        as a 2D array with the first index referring to a detector and the second
        index referring to a specific point in the timestream. The order of
        detectors is either by increasing x then f, or, if `xf` is specified,
        matches the input order

    Notes
    -----

    The data is returned as a 2D array of N_detectors x N_points. 
    If you have specified a set of detectors (ie used the xf argument), then 
    the order of detectors will match the order specified. Otherwise it will 
    be `xf = (0,0), (0,1), ... (0,15), (1,0), ...`. You can use 
    `val.reshape(16,60,-1)` to reshape the `val` array so that the first two 
    indices correspond to x and f respectively.

    There is no header information for this data. 

    **To Do**

    The beammapper code includes significant header data which is saved within a textfile.
    We could consider a feature update where the MCE data is combined with this header information
    within a Netcdf file. Final beammaps could be saved as extra variables within the same Netcdf file
    after they are created.

    """

    # Make sure path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found for path {path}.')

    # Check that a maximum of one of xf and cr have been provided
    if xf is not None and cr is not None:
        raise ValueError('cannot specify both xf and cr indexing')

    # Get file names
    files = _get_files(path,files_in,type='run')

    # Extract counts from MCE data
    mce_counts_all = []
    for file in files :
        f = mce_data.MCEFile(file)
        # reads in the MCE data in row/col format
        mce_counts_one = f.Read(row_col=True, unfilter='DC', all_headers=True).data
        mce_counts_all.append(mce_counts_one)

    mce_counts_all = np.array(mce_counts_all)

    # converts the detector coordinates to position and frequency
    if cr is not None:
        xf = [coordinates.muxcr_to_xf(cri[0],cri[1],p=mc) for cri in cr]
    if xf is None:
        xf = [(x,f) for x,f in itertools.product(np.arange(16),np.arange(60))]
    xf_new = np.empty(len(xf), dtype=object)
    xf_new[:] = xf
    xf = xf_new

    data = np.zeros((len(xf),len(mce_counts_all.shape[2])))
    for i in range(len(xf)):
        cr = coordinates.xf_to_muxcr(xf[i][0],xf[i][1],mc)
        data[i] = mce_counts_all[:,cr[1],cr[0],:].flatten()

    return data