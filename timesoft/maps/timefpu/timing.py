from __future__ import division, print_function, unicode_literals, absolute_import

import time
import datetime
import numpy as np

# Returns UTC seconds since the epoch
def sync2utc(sync, ref_sync, ref_epoch_local, freq=100.0, as_datetime=True, utc_offset=-7):
	
	t0 = ref_epoch_local - (ref_sync / freq)
	t = sync / freq + t0
	t -= utc_offset * 60 * 60
	
	if as_datetime:
		return datetime.datetime.fromtimestamp(t)
	else:
		return t

# Convert a datetime.datetime to a unix timestamp (time.time() output).
# Works on 1D arrays or single elements.
def datetime2time(d):
	try:
		# Treat as a single datetime
		return time.mktime(d.timetuple())
	except AttributeError:
		# Treat as an array of datetimes
		return np.asarray([time.mktime(x.timetuple()) for x in d])
		
# Convert a unix timestamp (time.time() output) to a datetime.datetime.
# Works on 1D arrays or single elements.
def time2datetime(t):
	try:
		# Treat as an array of times
		return np.asarray([datetime.datetime.fromtimestamp(x) for x in t])
	except TypeError:
		# Treat as a single time
		return datetime.datetime.fromtimestamp(t)

if __name__ == '__main__':
	
	print(datetime2time([datetime.datetime.now()]), datetime2time(datetime.datetime.now()), time.time())
	print(time2datetime([time.time()]), time2datetime(time.time()), datetime.datetime.now())
	
	# ~ for sync in np.arange(58852835, 58852845, 1):
		# ~ print(sync2utc(sync, ref_sync=58782635, ref_epoch_local=1552238365.000))
