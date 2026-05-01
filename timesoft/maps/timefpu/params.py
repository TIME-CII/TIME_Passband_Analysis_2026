from __future__ import division, print_function, unicode_literals, absolute_import

# Coordinate limits, see coordinates.py for more information

N_MODULES = 6 # det_m limit, modules per spectrometer bank
N_SUBARRAY = 4 # det_s limit, subarrays per module
N_DETBANK = 4 # det_b limit, detector banks per subarray
N_DETPERBANK_LF = 8 # LF det_d limit, detectors per subarray bank
N_DETPERBANK_HF = 12 # HF det_d limit, detectors per subarray bank
N_DETPERBANK = [N_DETPERBANK_LF]*3 + [N_DETPERBANK_HF]*3 # indexed by det_m

N_CHAN_SPATIAL = 16 # det_x limit, feedhorns per spectrometer bank
N_CHAN_SPECTRAL = 60 # det_f limit, spectral channels per single spectrometer

N_MUX_ROWS = 33 # det_r limit, MCE multiplexing rows
N_MUX_COLS = 32 # det_c limit, MCE multiplexing columns
