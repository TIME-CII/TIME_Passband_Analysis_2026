import numpy as np
import warnings
from timesoft.helpers import coordinates
from astropy.table import QTable

nfeed = 16

class Offsets:

    def __init__(self, x_off, y_off=None, frame='apparent', spectrometer=0, map_style='2d'):
        """Create an object to contain feed offset information
        
        Parameters
        ----------
        x_off : array-like
            An array of the x-offsets of each detector to the
            pointing axis of the telescope
        y_off : array-like
            An array of the y-offsets of each detector to the
            pointing axis of the telescope
        frame : {'apparent','J2000','Galactic'}
            The frame in which the offsets are measured
        spectrometer : {0, 1}
            The spectrometer for which the offsets are applicable
        """

        if len(x_off) != len(y_off):
            raise ValueError("x_off and y_off must have the same length")
        if len(x_off) != nfeed or len(x_off) != nfeed:
            raise ValueError("x_off and y_off must have length 16")

        if map_style=='2d':
            if y_off.any() == None:
                raise ValueError('y_off needs to specified for 2d map offsets')
            self.x_off = x_off
            self.y_off = y_off
        elif map_style=='1d':
            self.off = x_off
        self.frame = frame
        self.spec = spectrometer

    def get(self, x, theta=0):
        """Get the offsets for a given detector, optionally specify a map PA theta
        Give theta in degrees
        """
        
        # rotate as needed: 
        theta_rad = np.pi/180*theta
        xoffp = self.x_off[x]*np.cos(theta_rad) - self.y_off[x]*np.sin(theta_rad)
        yoffp = self.x_off[x]*np.sin(theta_rad) + self.y_off[x]*np.cos(theta_rad)

        return xoffp, yoffp

    def get_1d(self, x):
        return self.off

class DetectorConstants:

    def __init__(self, vals, xf_coords, mc=0):
        """Create an object to contain information related to 
        detector constants (gains, time constants, efficiencies,
        etc.)

        Note that this class is designed to handle cases where
        constants are only known for a subset of detectors - 
        not all detectors need to be specified.
        
        Parameters
        ----------
        vals : array-like
            An array of values for associated with the detectors
            specified in the xf_coords array
        xf_coords : array-like
            An array of tuples corresponding to the xf coordinates
            of the detectors associated with the value array
        mc : {0, 1}
            The MCE for which the values are applicable
        """
        
        if len(vals) != len(xf_coords):
            raise ValueError("must specify one xf_coords value for each value")
        
        cr_coords = [coordinates.xf_to_muxcr(xfi[0],xfi[1],p=mc) for xfi in xf_coords]
        xf_coords = [(xfi[0],xfi[1]) for xfi in xf_coords]

        self.vals = vals

        self.header = {'mc': mc,
                       'n_detectors': len(vals),
                       'xf_coords': xf_coords}
        self.header['xf_coords'] = np.empty(self.header['n_detectors'], dtype=object)
        self.header['xf_coords'][:] = xf_coords
        self.header['cr_coords'] = np.empty(self.header['n_detectors'], dtype=object)
        self.header['cr_coords'][:] = cr_coords


    @classmethod
    def from_file(cls, filename, **kwargs):
        '''
        Initialize a `DetectorConstants` object from a file saved with the
        `DetectorConstants.write` method.
        '''
        t = QTable.read(filename, **kwargs)
        vals = t['vals'] # Jy/beam / count
        xf_coords = t['xf_coords']
        mc = t.meta['mc'.upper()]
        return cls(vals, xf_coords, mc=mc)


    def add(self, vals, xf_coords, overwrite=False):
        """Add additional values"""

        if len(vals) != len(xf_coords):
            raise ValueError("must specify one xf_coords value for each value")
        
        new_vals = []
        new_coords = []
        for i,xf in enumerate(xf_coords):
            matches = [tuple(v) == (xf[0],xf[1]) for v in self.header['xf_coords']]
            indices = np.nonzero(matches)[0]
            if len(indices) > 0:
                if not overwrite:
                    raise ValueError("Some specified detectors already have values")
                self.vals[indices[0]] = vals[i]
            else:
                new_vals.append(vals[i])
                new_coords.append(xf)

        xf_new = np.empty(len(new_vals),dtype=object)
        xf_new[:] = new_coords
        cr_new = np.empty(len(new_vals),dtype=object)
        for i in range(len(xf_new)):
            cr_new[i] = coordinates.xf_to_muxcr(xf_new[i][0],xf_new[i][1],p=self.header['mc'])
        self.vals = np.concatenate((self.vals,new_vals))
        self.header['xf_coords'] = np.concatenate((self.header['xf_coords'],xf_new))
        self.header['cr_coords'] = np.concatenate((self.header['cr_coords'],cr_new))
        

    def check(self, c1, c2, det_coord_mode='xf'):
        """Check whether a value exists for a specified detector"""
        
        if det_coord_mode in ['xf','XF']:
            matches = [tuple(v) == (c1,c2) for v in self.header['xf_coords']]
            indices = np.nonzero(matches)[0]
        elif det_coord_mode in ['cr','CR','MuxCR','muxcr','MUXCR']:
            matches = [tuple(v) == (c1,c2) for v in self.header['cr_coords']]
            indices = np.nonzero(matches)[0]
        else:
            raise ValueError("det_coord_mode not recognized - specify one of 'xf' or 'cr'")

        if len(indices) == 0:
            return False
        else:
            return True


    def get(self, c1, c2, det_coord_mode='xf',raise_noval=False):
        """Get the value for a specified detector"""
        
        if det_coord_mode in ['xf','XF']:
            matches = [tuple(v) == (c1,c2) for v in self.header['xf_coords']]
            indices = np.nonzero(matches)[0]
        elif det_coord_mode in ['cr','CR','MuxCR','muxcr','MUXCR']:
            matches = [tuple(v) == (c1,c2) for v in self.header['cr_coords']]
            indices = np.nonzero(matches)[0]
        else:
            raise ValueError("det_coord_mode not recognized - specify one of 'xf' or 'cr'")

        if len(indices) == 0:
            if raise_noval:
                raise ValueError('Detector not present in DetectorConstants object')
            else:
                warnings.warn("Detector not present in DetectorConstants object - using 0")
                return 0

        if len(indices) > 1:
            warnings.warn("More than one instance of detector ({},{}) found".format(c1,c2))

        return self.vals[indices[0]]


    def write(self, filename, **kwargs):
        '''
        Write DetectorConstants to a file using the astropy QTable interface.
        kwargs are passed to `Table.write`.

        Parameters
        ----------
            filename : str
                File to write out. Infers from extension, or supply with
                `format` kwarg.

        Returns
        -------
            None
        '''
        t = QTable()
        for key, val in self.header.items():
            print(key,val)
            if key in ['xf_coords', 'cr_coords']:
                t[key] = val
            else:
                t.meta[key] = val
        t['val'] = self.vals # Jy/beam / count
        t.write(filename, **kwargs)