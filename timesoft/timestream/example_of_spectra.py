#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:38:10 2024

@author: dunn
"""

# from timesoft import Timestream
from timesoft.timestream.timestream_tools import spectra, make_fits, save_
import matplotlib.pyplot as plt
import numpy as np

const_RA_data_path = './omc_data/1644379146'
planet_cal_data_path = './planet_data/1644101949'


ra_ts = Timestream(const_RA_data_path, mc=0, store_copy=False)
Maps = ra_ts.make_calibrated_maps(planet_cal_data_path,date_override='2021-02-09T05:00:00', obj_override='jupiter')
make_fits(Maps) 
save_map_level1(Maps)
# spectra_per_pix = ra_ts.spectra(input_maps = None)

spectra_per_pix = spectra()

width, height, spec_size = spectra_per_pix.shape

# ra_ts.Maps.maps = ra_ts.pixel_cut(ra_ts.Maps.maps)

# plt.imshow(ra_ts.Maps.maps[0], origin='lower')
# plt.colorbar(label='Intensity [Jy/beam]')
# plt.show()


for i in range(width):
    for j in range(height):
        data = spectra_per_pix[i][j][:]
        plt.plot(channels, data)
        plt.scatter(channels, data)
plt.plot(channels, np.zeros(len(channels)), color = 'black', linestyle='dotted', label = 'Line at 0', alpha=0.5)
plt.xlabel('Freq [GHz]')
plt.ylabel('Intensity [Jy/beam]')
plt.legend()
plt.show()