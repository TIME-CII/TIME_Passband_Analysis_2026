# from timesoft.maps.map_tools import Map
import matplotlib.pyplot as plt 
import numpy as np 
# from timesoft import Timestream
import pandas as pd
# from timesoft.detector_cuts import *
from scipy.optimize import curve_fit
from astropy.modeling.models import BlackBody
# from timesoft.calibration import DetectorConstants
from timesoft.timestream.timestream_tools import Timestream
from timesoft.Utilities.config import *

# const_RA_data_path = '/Users/butler/Documents/omc-data/1644379146'
# # const_DEC_data_path = '/home/vaughan/omc_test_data/1644116132'
planet_cal_data_path = '/Users/butler/Documents/kms_p1_2'

print(const_RA_data_path)
dec_ts = Timestream(const_RA_data_path, mc=0, store_copy=False)
header, maps, channels, planet_ts = dec_ts.make_calibrated_maps(planet_cal_data_path,date_override='2021-02-09T05:00:00', obj_override='jupiter')

for map,coord in zip(maps, header['xf_coords']):
    plt.imshow(map, origin='lower')
    plt.colorbar(label='jy/beam')
    plt.savefig(data_save + 'test_data/%s_%s' % (coord[0],coord[1]))
    # plt.show()
    plt.clf()

