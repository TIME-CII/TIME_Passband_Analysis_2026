from __future__ import division, print_function, unicode_literals, absolute_import

### Detector Coordinate Conversion ###

# mux_c: multiplexing column (mce)
# mux_r: multiplexing row (mce)
# p: polarization (0 for no twist feed, 1 for twist)
# m: module, with 0 being the lowest low frequency
# s: sub array, with 0 nearest the cables
# b: bank, 0 at pin
# d: detector, 0 at pin
# f: frequency index (not value), with 0 being the lowest frequency
# x: spatial index, with 0 being the spectrometer closest to the motherboard

# For plots to help visualize these coordinate systems, see the 
# Mapping Visualization section in the following post:
# https://chronos.caltech.edu/time/logbook/instrument/20190531-detmapping/

def xf_to_muxcr(x,f,p=0):

	if p != 0:
		raise NotImplementedError("p=1 not implemented")
		
	v = _xf_to_muxcr.get((x,f), None)
	
	if v is None:
		raise NotImplementedError("Not implemented!")
	
	return v	
	
def muxcr_to_xf(c,r,p=0):
	
	if p != 0:
		raise NotImplementedError("p=1 not implemented")
	
	if r == 32:
		raise ValueError("The dark squid doesn't have an x,f value")
	
	if (c > 29):
		raise NotImplementedError("150 GHz not implemented")
	
	assert(c >= 0 and c < 30)
	assert(r >= 0 and r <= 32)
	
	(m,s,b,d) = muxcr_to_msbd(c,r,p)
	
	# Module contribution to f
	if m < 3:
		f = m * 8
	else:
		f = 3 * 8 + (m-3) * 12
		
	# Intra-module component of f
	if s > 1:
		f += d
	else:
		# Upside down subarrays
		if m < 3:
			f += 7-d
		else:
			f += 11-d
	
	# Subarray component of x
	x = 4 * s
	
	# Intra-subarray component of x
	if s < 2:
		x += 3 - b
	else:
		x += b
		
	assert(x >= 0 and x <= 15)
	assert(f >= 0 and f <= 60)
	
	return (x,f)	

def xf_to_msbd(x,f,p=0):
	
	c,r = xf_to_muxcr(x,f)
	return muxcr_to_msbd(c,r)
	
	
def muxcr_to_msbd(c,r,p=0):
	
	if p != 0:
		raise NotImplementedError
		
	if c % 10 <= 5:
		m = 5 - (c // 10)
	else:
		m = (c // 10)
	
	(s,b,d) = muxcr_to_sbd(c, r, p)
	
	return (m,s,b,d)
	
	
def msbd_to_muxcr(m,s,b,d,p=0):
	
	if p != 0:
		raise NotImplementedError
	
	assert(m >= 0 and s >= 0 and b >= 0 and d >= 0)
	assert(m < 6 and s < 4 and b < 4 and d < 12)
	assert(m >= 3 or d < 8)		
		
	c,r = _msbd_to_muxcr.get((m,s,b,d), (None, None))
	
	if c is None or r is None:
		raise ValueError("Invalid m,s,b,d")
		
	return (c,r)

def msbd_to_xf(m,s,b,d,p=0):
	c,r = msbd_to_muxcr(m,s,b,d,p)
	return muxcr_to_xf(c,r,p)

def muxcr_to_sbd(c,r,p=0):
	
	c %= 10 # The module assignment repeats every 10 col
	
	if p != 0:
		raise NotImplementedError("p=1 not implemented")

	s,d,b = _sdb_bycol[c][r]
	
	return (s,b,d)

########################################################################

# Engineering Run 2019 mux row mapping correction.  Returns None for
# rows that don't map (some data was not stored because of the MCE
# and Sync Box configuration).
#   row_eng19: the (incorrect) row stored in NetCDF files from 2019
#   row_phys: the physical multiplexing row (the one you convert to xf)
# More: https://chronos.caltech.edu/time/logbook/instrument/20200203-planetdetmap/

def row_eng19_to_phys(row_eng19):
	row_phys = (row_eng19 + 14) % 33 
	if (row_phys == 0) or (row_phys >= 14 and row_phys <= 32):
		return row_phys
	else:
		return None
		
def row_phys_to_eng19(row_phys):	
	row_eng19 = (row_phys - 14) % 33 
	if (row_phys == 0) or (row_phys >= 14 and row_phys <= 32):
		return row_eng19
	else:
		return None
	
########################################################################
	
# Main mapping from mux to detectors for pol 0.
# LF not well verified.
_sdb_bycol = {
	0: [	
			(2,8,0),(2,10,0),(2,11,0),(2,9,0),
			(2,8,1),(2,10,1),(2,11,1),(2,9,1),
			(2,8,2),(2,10,2),(2,11,2),(2,9,2),
			(2,8,3),(2,10,3),(2,11,3),(2,9,3),
			(3,8,0),(3,10,0),(3,11,0),(3,9,0),
			(3,8,1),(3,10,1),(3,11,1),(3,9,1),
			(3,8,2),(3,10,2),(3,11,2),(3,9,2),
			(3,8,3),(3,10,3),(3,11,3),(3,9,3), (None, None, None)
		],
		
	1: [(3,3,3),(3,1,3),(3,0,3),(3,2,3),(3,4,3),(3,6,3),(3,7,3),(3,5,3),(3,3,2),(3,1,2),(3,0,2),(3,2,2),(3,4,2),(3,6,2),(3,7,2),(3,5,2),(3,3,1),(3,1,1),(3,0,1),(3,2,1),(3,4,1),(3,6,1),(3,7,1),(3,5,1),(3,3,0),(3,1,0),(3,0,0),(3,2,0),(3,4,0),(3,6,0),(3,7,0),(3,5,0),("Dark",11,0)],
	2: [(2,3,3),(2,1,3),(2,0,3),(2,2,3),(2,4,3),(2,6,3),(2,7,3),(2,5,3),(2,3,2),(2,1,2),(2,0,2),(2,2,2),(2,4,2),(2,6,2),(2,7,2),(2,5,2),(2,3,1),(2,1,1),(2,0,1),(2,2,1),(2,4,1),(2,6,1),(2,7,1),(2,5,1),(2,3,0),(2,1,0),(2,0,0),(2,2,0),(2,4,0),(2,6,0),(2,7,0),(2,5,0),("Dark",12,0)],
	
	3: [	
			(1,8,0),(1,10,0),(1,11,0),(1,9,0),
			(1,8,1),(1,10,1),(1,11,1),(1,9,1),
			(1,8,2),(1,10,2),(1,11,2),(1,9,2),
			(1,8,3),(1,10,3),(1,11,3),(1,9,3),
			(0,8,0),(0,10,0),(0,11,0),(0,9,0),
			(0,8,1),(0,10,1),(0,11,1),(0,9,1),
			(0,8,2),(0,10,2),(0,11,2),(0,9,2),
			(0,8,3),(0,10,3),(0,11,3),(0,9,3), (None, None, None)
		],
	
	4: [(1,3,3),(1,1,3),(1,0,3),(1,2,3),(1,4,3),(1,6,3),(1,7,3),(1,5,3),(1,3,2),(1,1,2),(1,0,2),(1,2,2),(1,4,2),(1,6,2),(1,7,2),(1,5,2),(1,3,1),(1,1,1),(1,0,1),(1,2,1),(1,4,1),(1,6,1),(1,7,1),(1,5,1),(1,3,0),(1,1,0),(1,0,0),(1,2,0),(1,4,0),(1,6,0),(1,7,0),(1,5,0),("Dark",14,0)],
	5: [(0,3,3),(0,1,3),(0,0,3),(0,2,3),(0,4,3),(0,6,3),(0,7,3),(0,5,3),(0,3,2),(0,1,2),(0,0,2),(0,2,2),(0,4,2),(0,6,2),(0,7,2),(0,5,2),(0,3,1),(0,1,1),(0,0,1),(0,2,1),(0,4,1),(0,6,1),(0,7,1),(0,5,1),(0,3,0),(0,1,0),(0,0,0),(0,2,0),(0,4,0),(0,6,0),(0,7,0),(0,5,0),("Dark",15,0)],
	
	# LF layer 1
	9:[	
			(2,4,0),(2,6,0),(2,7,0),(2,5,0),
			(2,4,1),(2,6,1),(2,7,1),(2,5,1),
			(2,4,2),(2,6,2),(2,7,2),(2,5,2),
			(2,4,3),(2,6,3),(2,7,3),(2,5,3),
			(3,4,0),(3,6,0),(3,7,0),(3,5,0),
			(3,4,1),(3,6,1),(3,7,1),(3,5,1),
			(3,4,2),(3,6,2),(3,7,2),(3,5,2),
			(3,4,3),(3,6,3),(3,7,3),(3,5,3), (None, None, None)
		],
	
	# LF layer 2
	8: [
			(3,3,3),(3,1,3),(3,0,3),(3,2,3),
			(3,3,2),(3,1,2),(3,0,2),(3,2,2),
			(3,3,1),(3,1,1),(3,0,1),(3,2,1),
			(3,3,0),(3,1,0),(3,0,0),(3,2,0),
			(2,3,3),(2,1,3),(2,0,3),(2,2,3),
			(2,3,2),(2,1,2),(2,0,2),(2,2,2),
			(2,3,1),(2,1,1),(2,0,1),(2,2,1),
			(2,3,0),(2,1,0),(2,0,0),(2,2,0), (None, None, None)
		],
	
	# LF layer 3
	7: [
			(0,3,3),(0,1,3),(0,0,3),(0,2,3),
			(0,3,2),(0,1,2),(0,0,2),(0,2,2),
			(0,3,1),(0,1,1),(0,0,1),(0,2,1),
			(0,3,0),(0,1,0),(0,0,0),(0,2,0),
			(1,3,3),(1,1,3),(1,0,3),(1,2,3),
			(1,3,2),(1,1,2),(1,0,2),(1,2,2),
			(1,3,1),(1,1,1),(1,0,1),(1,2,1),
			(1,3,0),(1,1,0),(1,0,0),(1,2,0), (None, None, None)],
	
	# LF layer 4
	6: [	
			(1,4,0),(1,6,0),(1,7,0),(1,5,0),
			(1,4,1),(1,6,1),(1,7,1),(1,5,1),
			(1,4,2),(1,6,2),(1,7,2),(1,5,2),
			(1,4,3),(1,6,3),(1,7,3),(1,5,3),
			(0,4,0),(0,6,0),(0,7,0),(0,5,0),
			(0,4,1),(0,6,1),(0,7,1),(0,5,1),
			(0,4,2),(0,6,2),(0,7,2),(0,5,2),
			(0,4,3),(0,6,3),(0,7,3),(0,5,3), (None, None, None)
		],
	
	}

# Assemble reverse lookup tables

_xf_to_muxcr = {}
for c in range(30):
	for r in range(32):
		try:
			x,f = muxcr_to_xf(c,r)
			assert ((x,f) not in list(_xf_to_muxcr.keys()))
			_xf_to_muxcr[(x,f)] = (c,r)
		except NotImplementedError:
			continue
		except AssertionError:
			print("FAILURE: Duplicate when generating coordinate lookup tables: c%02ir%02i -> x%02if%02i" % (c,r,x,f))
			exit()

_msbd_to_muxcr = {}
for c in range(30):
	for r in range(32):
		try:
			m,s,b,d = muxcr_to_msbd(c,r)
			assert ((m,s,b,d) not in list(_msbd_to_muxcr.keys()))
			_msbd_to_muxcr[(m,s,b,d)] = (c,r)
		except NotImplementedError:
			continue
		except AssertionError:
			print("FAILURE: Duplicate when generating coordinate lookup tables: c%02ir%02i -> m%01is%01ib%01id%02i" % (c,r,m,s,b,d))
			exit()
