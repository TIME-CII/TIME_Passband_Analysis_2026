import numpy as np
import warnings

from timesoft.calibration.detector_cal import Offsets, DetectorConstants

class _Datastruct_base:
    """Class with some data handling tools common across timestreams, maps, and 1d data"""

    def get_xf(self,x,f,raise_nodet=True):
        """Determine the index corresponding to the detector with xf indices (x,f)

        The order of scans in the ``class.data`` array
        varries depending on how the timestream is initialized,
        and does not always have an obvious trivial mapping to 
        detector coordinates in feed-frequency (xf) or readout
        colum-row (muxcr) space. This function keeps track of the
        mapping between a detectors index in ``class.data`` 
        and its xf coordinates: given a detector x and f value it
        returns the corresponding index in ``class.data`` array.

        Parameters
        ----------
        x : int
            An integer from 0 to 15 specifying the feedhorn number
            of the target detector
        f : int
            An integer from 0 to 60 specifying the frequency channel
        raise_nodet : bool, default=True
            If True an error will be raised when no detector is found
        
        Returns
        -------
        index : int
            The index of the requested detector in ``class.data`` array

        Examples
        --------
        >>> index_of_x7f47 = class.get_xf(7,47)
        >>> data_for_x7f47 = class.data[index_of_x7f47,:]

        Or
        >>> data_for_x7f47 = class.data[class.get_xf(7,47),:]
        """
        
        matches = [v == (x,f) for v in self.header['xf_coords']]
        indices = np.nonzero(matches)[0]
        if len(indices) == 0:
            if raise_nodet:
                raise ValueError('xf pair not found')
            else:
                return None
        if len(indices) > 1:
            warnings.warn("More than one instance of xf=({},{}) found".format(x,f))

        return indices[0]


    def get_cr(self,c,r,raise_nodet=True):
        """Determine the index in the ``class.data`` array corresponding to the detector with muxcr indices (c,r)

        The order of scans in the ``class.data`` array
        varries depending on how the timestream is initialized,
        and does not always have an obvious trivial mapping to 
        detector coordinates in feed-frequency (xf) or readout
        colum-row (muxcr) space. This function keeps track of the
        mapping between a detectors index in ``class.data`` 
        and its muxcr coordinates: given a detector c and r value it
        returns the corresponding index in ``class.data`` array.

        Parameters
        ----------
        c : int
            An integer specifying the readout column number
            of the target detector
        r : int
            An integer specifying the readout row number
            of the target detector
        raise_nodet : bool, default=True
            If True an error will be raised when no detector is found

        Returns
        -------
        index : int
            The index of the requested detector in ``class.data`` array

        Examples
        --------
        >>> index_of_c1r5 = class.get_crf(1,5)
        >>> data_for_c1r5 = class.data[index_of_c1r5,:]

        Or
        >>> data_for_c1r5 = class.data[class.get_cr(1,5)),:]
        """

        matches = [v == (c,r) for v in self.header['cr_coords']]
        indices = np.nonzero(matches)[0]
        if len(indices) == 0:
            if raise_nodet:
                raise ValueError('xf pair not found')
            else:
                return None
        if len(indices) > 1:
            warnings.warn("More than one instance of muxcr=({},{}) found".format(c,r))

        return indices[0]


    def _get_coord(self,c1,c2,det_coord_mode,raise_nodet=True):
        """Wraper for get_xf and get_cr (internal use)"""
        if det_coord_mode in ['xf','XF']:
            index = self.get_xf(c1,c2,raise_nodet=raise_nodet)
        elif det_coord_mode in ['cr','CR','MuxCR','muxcr','MUXCR']:
            index = self.get_cr(c1,c2,raise_nodet=raise_nodet)
        else:
            raise ValueError("detector_coords not recognized - specify one of 'xf' or 'muxcr'")

        return index


    # Get all detectors from a feedhorn
    def get_x(self,x,raise_nodet=True):
        """Get all detectors for feedhorn x

        The order of scans in the ``class.data`` array
        varries depending on how the timestream is initialized,
        and does not always have an obvious trivial mapping to 
        detector coordinates in feed-frequency (xf) or readout
        colum-row (muxcr) space. This function keeps track of the
        mapping between a detectors index in ``class.data`` 
        and its xf coordinates: given a feed x it
        returns the indices of all detectors for that feedhorn
        in the ``class.data`` array.

        Parameters
        ----------
        x : int
            An integer from 0 to 15 specifying the feedhorn number
        raise_nodet : bool, default=True
            If True an error will be raised when no detectors are found
        
        Returns
        -------
        index : int
            The indices of the requested feedhorn in ``class.data`` array
        """

        matches = [v[0]==x for v in self.header['xf_coords']]
        indices = np.nonzero(matches)[0]

        if len(indices) == 0 and raise_nodet:
            raise ValueError('No detectors in this feed found')

        return indices


    # Get all detectors from a frequency index
    def get_f(self,f,raise_nodet=True):
        """Get all detectors for frequency index f

        The order of scans in the ``class.data`` array
        varries depending on how the timestream is initialized,
        and does not always have an obvious trivial mapping to 
        detector coordinates in feed-frequency (xf) or readout
        colum-row (muxcr) space. This function keeps track of the
        mapping between a detectors index in ``class.data`` 
        and its xf coordinates: given a frequency index f it
        returns the indices of all detectors for that index
        in the ``class.data`` array.

        Parameters
        ----------
        f : int
            An integer from 0 to 60 specifying the frequency channel
        raise_nodet : bool, default=True
            If True an error will be raised when no detectors are found
        
        Returns
        -------
        index : int
            The indices of the requested frequency index in ``class.data`` array
        """

        matches = [v[1]==f for v in self.header['xf_coords']]
        indices = np.nonzero(matches)[0]

        if len(indices) == 0 and raise_nodet:
            raise ValueError('No detectors in this frequency found')

        return indices


    def get_c(self,c,raise_nodet=True):
        """Get all detectors for mux column c

        The order of scans in the ``class.data`` array
        varries depending on how the timestream is initialized,
        and does not always have an obvious trivial mapping to 
        detector coordinates in feed-frequency (xf) or readout
        colum-row (muxcr) space. This function keeps track of the
        mapping between a detectors index in ``class.data`` 
        and its cr coordinates: given a mux column c it
        returns the indices of all detectors for that index
        in the ``class.data`` array.

        Parameters
        ----------
        c : int
            An integer specifying the readout column number
            of the target detector
        raise_nodet : bool, default=True
            If True an error will be raised when no detectors are found
        
        Returns
        -------
        index : int
            The indices of the requested column in ``class.data`` array
        """

        matches = [v[0]==c for v in self.header['cr_coords']]
        indices = np.nonzero(matches)[0]

        if len(indices) == 0 and raise_nodet:
            raise ValueError('No detectors in this column found')

        return indices


    def get_r(self,c,raise_nodet=True):
        """Get all detectors for mux row r

        The order of scans in the ``class.data`` array
        varries depending on how the timestream is initialized,
        and does not always have an obvious trivial mapping to 
        detector coordinates in feed-frequency (xf) or readout
        colum-row (muxcr) space. This function keeps track of the
        mapping between a detectors index in ``class.data`` 
        and its cr coordinates: given a mux row r it
        returns the indices of all detectors for that index
        in the ``class.data`` array.

        Parameters
        ----------
        r : int
            An integer specifying the readout row number
            of the target detector
        raise_nodet : bool, default=True
            If True an error will be raised when no detectors are found
        
        Returns
        -------
        index : int
            The indices of the requested row in ``class.data`` array
        """

        matches = [v[1]==r for v in self.header['cr_coords']]
        indices = np.nonzero(matches)[0]

        if len(indices) == 0 and raise_nodet:
            raise ValueError('No detectors in this column found')

        return indices


    def header_entry(self,key,value,protect=True):
        """Change the value of an item in the header

        Parameters
        ----------
        key : str
            Name of the new header entry.
        value : any
            Value of the new header entry.
        protect : bool, default=True
            If True, trying to change existing header values will
            raise an error. This is to prevent accidentally 
            overwriting header information.

        Returns
        -------
        None
        """

        if protect:
            if key in self.header.keys():
                raise ValueError("Specified key already in use")
        
        self.header[key] = value


    def _check_header(self,header):
        """Checks whether a minimum set of keys are present in a header"""

        checks = {'flags':'Header missing flags to track analysis steps',
                  'xf_coords':'Header missing xf coordinate IDs',
                  'cr_coords':'Header missing muxcr coordinate IDs'}

        for key in checks.keys():
            if key not in header.keys():
                raise ValueError(checks[key])


    def _get_det_inds(self,dets,det_coord_mode='xf'):
        """Find the index in the data arrays corresponding to a list of detectors in xf, cr, or index space"""
        if det_coord_mode in ['xf','XF']:
            coord_key = 'xf_coords'
        elif det_coord_mode in ['cr','CR','MuxCR','muxcr','MUXCR']:
            coord_key = 'cr_coords'
        elif det_coord_mode in ['idx']:
            coord_key = 'idx'
        else:
            raise ValueError("det_coord_mode not recognized")


        inds = []

        if coord_key == 'idx':
            for id in dets:
                if id > len(self.header['xf_coords'])-1 or id<0:
                    warnings.warn("Detector {}={} is not in the dataset, skipped when loading".format(det_coord_mode,id))
                else:
                    inds.append(id)

        else:
            for id in dets:
                matches = [id==v for v in self.header[coord_key]]
                if np.any(matches):
                    inds.append(np.nonzero(matches)[0][0])
                else:
                    warnings.warn("Detector {}={} is not in the dataset, skipped when loading".format(det_coord_mode,id))

        if len(inds)==0:
            raise ValueError("No valid detectors specified")

        return inds


    def set_feed_offsets(self,offset,rotate_feeds=0,raise_spec=True,raise_frame=True,overwrite=False):
        """Set the offsets between feedhorns when the data was 
        collected.

        Parameters
        ----------
        offset : ``timesoft.calibration.Offsets`` instance
            An object containing offset measurements for each feedhorn.
        rotate_feeds : float
            Angle by which to rotate the specified offsets to match 
            dataset position angle. Generally this should be 
            self.header['scan_pars']['map_angle_offset'] for Timestreams or
            self.header['map_pars']['map_angle_offset'] for Maps
        raise_spec : bool, default=True
            Check that the offset object is for the same spectrometer 
            as the dataset and raise an error if not
        raise_frame : bool, default=True
            Check that the coordinate frame for the offset object
            and the dataset are the same and raise an error if not
        overwrite : bool, default=False
            If the dataset already has offsets specified, then
            this method will raise an error unless overwrite=True
            is specified.
        """

        if not overwrite and self.header['flags']['has_feed_offsets']:
            raise ValueError("Feed offsets already set for this object. Use overwrite=True to overwrite them.")

        if not isinstance(offset,Offsets):
            raise ValueError("offsets must be an instance of timesoft.calibration.Offsets")

        # if self.header['mc'] != offset.spec:
        #     raise ValueError("Offsets are for the wrong spectrometer. Set raise_spec=False to ignore.")
        if raise_frame:
            if self.header['epoch'] != offset.frame:
                raise ValueError("Offset frame does not match data epoch. Set raise_frame=False to ignore.")

        self.header['feed_offsets'] = np.array([np.array(offset.get(x,rotate_feeds)) for x in range(16)])
        self.header['flags']['has_feed_offsets'] = True


    def get_feed_offsets(self,c1,c2=None,theta=0,det_coord_mode='x',raise_nan=True, map_style='2d'):
        """Get the offsets for a specified detector
        
        Parameters
        ----------
        c1 : int
            The first coordinate of the detector in the specified coordinate
            system. Default coordinate system is 'x', in which case this is 
            the feedhorn number and no second coordinate is needed.
        c2 : int, default=None
            The second coordinate of the detector in the specified coordinate
            system. Default coordinate system is 'x', in which case this is 
            this coordinate is not specified.
            Rotation angle between the observation and the offsets
        det_coord_mode : {'x','xf','cr'}, default='x'
            Coordinate system used to identify detectors. The default is
            'x' which requires only the feedhorn number (as all detectors
            of a given feedhorn have the same offsets). 'xf' (default) or 
            'cr' are accepted and can be used to specify a specific detector.
        raise_nan : bool, default=True
            Detectors with missing information have offsets recorded as 
            NaNs. If True, then attempting to get offsets for these 
            detectors will cause an error. If False, the offsets for these
            detectors will be returned as zeros.
        
        Returns
        -------
        x_offset, y_offset
            The offset between the ``self.ra`` and ``self.dec`` arrays
            and the R.A. and declination of the specified feedhorn.
        """

        if not self.header['flags']['has_feed_offsets']:
            raise ValueError("No feed offsets known")

        if det_coord_mode == 'x':
            xval = c1
        else:
            if c2 is None:
                raise ValueError("Must specify c2 for 'xf' or 'cr' coord mode")
            index = self._get_coord(c1,c2,det_coord_mode)
            xval = self.header['xf_coords'][index][0]

        if map_style == '2d':
            xoff = self.header['feed_offsets'][xval][0]
            yoff = self.header['feed_offsets'][xval][1]

            if np.isnan(xoff):
                if raise_nan:
                    raise ValueError("Offset is nan")
                warnings.warn("nan offset found for feed using 0")
                xoff = 0
            if np.isnan(yoff):
                if raise_nan:
                    raise ValueError("Offset is nan")
                warnings.warn("nan offset found for feed using 0")
                yoff = 0
            return xoff, yoff

        elif map_style == '1d':
            off = self.header['feed_offsets'][xval]

            if np.isnan(off):
                if raise_nan:
                    raise ValueError("Offset is nan")
                warnings.warn("nan offset found for feed using 0")
                off = 0
            return off


    def set_gains(self,gains,gains_e=None,check_dets=True,drop_missing_dets=True,overwrite=False):
        """Set the gains for the detectors.

        If the gains object does not contain data for all 
        detectors in the dataset, the default behavior is
        to remove the detectors with missing calibration 
        data. This can be changed by setting the `drop_missing_dets`
        argument to False, in which case the gain for these 
        detectors will be set to zero.

        Parameters
        ----------
        gains : ``timesoft.calibration.DetectorConstants`` instance
            An object containing gain measurements for a subset of the
            detectors. 
        gains_e : ``timesoft.calibration.DetectorConstants`` instance
            An object containing gain uncertainties. 
        check_dets : bool, default=True
            If set to False, no check will be performed to make sure
            that detectors in the dataset have a match in the gains
            object. The gains for missing detectors will be set to 
            zero with no warning provided.
        drop_missing_dets : bool, defaul=True
            If True then detectors that do not have gain data will be
            removed from the dataset. This is apropriate if you only
            want detectors that can be calibrated. If False, these 
            detectors will be kept but the gains for them will be zero.
        overwrite : bool, default=False
            If the dataset already has gains specified, then
            this method will raise an error unless overwrite=True
            is specified.
        """

        if not overwrite and self.header['flags']['has_gains']:
            raise ValueError("Gains already set for this object. Use overwrite=True to overwrite them.")

        if not isinstance(gains,DetectorConstants):
            raise ValueError("gains must be an instance of timesoft.calibration.DetectorConstants")

        has_xf = np.zeros(self.header['n_detectors'])
        gain_val = np.zeros(self.header['n_detectors'])
        for i in range(self.header['n_detectors']):
            xf = self.header['xf_coords'][i]
            if check_dets:
                has_xf[i] = gains.check(xf[0],xf[1],det_coord_mode='xf')
                if not has_xf[i] and not drop_missing_dets:
                    warnings.warn("No gain found for xf=({},{}). A value of 0 will be used.".format(xf[0],xf[1]))
            gain_val[i] = gains.get(xf[0],xf[1],det_coord_mode='xf',raise_noval=False)
        if drop_missing_dets:
            coords_keep = self.header['xf_coords'][np.nonzero(has_xf)]
            self.restrict_detectors(coords_keep,det_coord_mode='xf')
        self.header['gains'] = gain_val
        self.header['gains_e'] = np.zeros(self.header['gains'].shape)
        self.header['flags']['has_gains'] = True

        if gains_e is not None:
            gain_val_e = np.zeros((self.header['n_detectors']))
            for i in range(self.header['n_detectors']):
                xf = self.header['xf_coords'][i]
                if check_dets:
                    has_xf[i] = gains_e.check(xf[0],xf[1],det_coord_mode='xf')
                    if not has_xf[i] and not drop_missing_dets:
                        warnings.warn("No gain_e found for xf=({},{}). A value of 0 will be used.".format(xf[0],xf[1]))
                gain_val_e[i] = gains_e.get(xf[0],xf[1],det_coord_mode='xf', raise_noval=False)

            gain_val_e = gain_val_e[np.nonzero(has_xf)]
            if drop_missing_dets:
                coords_keep = self.header['xf_coords'][np.nonzero(has_xf)]
                self.restrict_detectors(coords_keep,det_coord_mode='xf')

            self.header['gains_e'] = gain_val_e

    def set_beams(self,beams,check_dets=True,drop_missing_dets=True,overwrite=False):
        """Set the beam FWHM for the detectors.

        If the beam object does not contain data for all 
        detectors in the dataset, the default behavior is
        to remove the detectors with missing calibration 
        data. This can be changed by setting the `drop_missing_dets`
        argument to False, in which case the beam size for these 
        detectors will be set to zero.

        Parameters
        ----------
        beams : ``timesoft.calibration.DetectorConstants`` instance
            An object containing beam size measurements for a subset of the
            detectors. 
        check_dets : bool, default=True
            If set to False, no check will be performed to make sure
            that detectors in the dataset have a match in the beams
            object. The beams for missing detectors will be set to 
            zero with no warning provided.
        drop_missing_dets : bool, defaul=True
            If True then detectors that do not have gain data will be
            removed from the dataset. This is apropriate if you only
            want detectors that can be calibrated. If False, these 
            detectors will be kept but the beams for them will be zero.
        overwrite : bool, default=False
            If the dataset already has beams specified, then
            this method will raise an error unless overwrite=True
            is specified.
        """

        if not overwrite and self.header['flags']['has_beams']:
            raise ValueError("Gains already set for this object. Use overwrite=True to overwrite them.")

        if not isinstance(beams,DetectorConstants):
            raise ValueError("beams must be an instance of timesoft.calibration.DetectorConstants")

        has_xf = np.zeros(self.header['n_detectors'])
        beams_val = np.zeros(self.header['n_detectors'])
        for i in range(self.header['n_detectors']):
            xf = self.header['xf_coords'][i]
            if check_dets:
                has_xf[i] = beams.check(xf[0],xf[1],det_coord_mode='xf')
                if not has_xf[i] and not drop_missing_dets:
                    warnings.warn("No gain found for xf=({},{}). A value of 0 will be used.".format(xf[0],xf[1]))
            beams_val[i] = beams.get(xf[0],xf[1],det_coord_mode='xf',raise_noval=False)

        if drop_missing_dets:
            coords_keep = self.header['xf_coords'][np.nonzero(has_xf)]
            self.restrict_detectors(coords_keep,det_coord_mode='xf')
            beams_val = beams_val[np.nonzero(has_xf)]

        self.header['beams'] = beams_val
        self.header['flags']['has_beams'] = True

    def set_time_constants(self,time_constants,check_dets=True,drop_missing_dets=True,overwrite=False):
        """Set the time constants for the detectors.

        If the time_constants object does not contain data for all 
        detectors in the dataset, the default behavior is
        to remove the detectors with missing calibration 
        data. This can be changed by setting the `drop_missing_dets`
        argument to False, in which case the gain for these 
        detectors will be set to zero.

        Parameters
        ----------
        gains : ``timesoft.calibration.DetectorConstants`` instance
            An object containing time constant measurements for a subset of the
            detectors. 
        check_dets : bool, default=True
            If set to False, no check will be performed to make sure
            that detectors in the dataset have a match in the time_constants
            object. The time constants for missing detectors will be set to 
            zero with no warning provided.
        drop_missing_dets : bool, defaul=True
            If True then detectors that do not have time constant data will be
            removed from the dataset. This is apropriate if you only
            want detectors that can be calibrated. If False, these 
            detectors will be kept but the time constants for them will be zero.
        overwrite : bool, default=False
            If the dataset already has time constants specified, then
            this method will raise an error unless overwrite=True
            is specified.
        """

        if not overwrite and self.header['flags']['has_time_constants']:
            raise ValueError("Time constants already set for this object. Use overwrite=True to overwrite them.")

        if not isinstance(time_constants,DetectorConstants):
            raise ValueError("time_constants must be an instance of timesoft.calibration.DetectorConstants")

        has_xf = np.zeros(self.header['n_detectors'])
        tc_val = np.ones(self.header['n_detectors'])
        for i in range(self.header['n_detectors']):
            xf = self.header['xf_coords'][i]
            if check_dets:
                has_xf[i] = time_constants.check(xf[0],xf[1],det_coord_mode='xf')
                if not has_xf[i] and not drop_missing_dets:
                    warnings.warn("No gain found for xf=({},{}). A value of 0 will be used.".format(xf[0],xf[1]))
            tc_val[i] = time_constants.get(xf[0],xf[1],det_coord_mode='xf',raise_noval=False)

        if drop_missing_dets:
            coords_keep = self.header['xf_coords'][np.nonzero(has_xf)]
            self.restrict_detectors(coords_keep,det_coord_mode='xf')
            tc_val = tc_val[np.nonzero(has_xf)]

        self.header['time_constants'] = tc_val
        self.header['flags']['has_time_constants'] = True

    def red_chi_square(self, model, data, uncertainty):
        return np.nansum( (model - data)**2 / uncertainty**2) / (data.size) #assume num of params << num of data points 
