from __future__ import division, print_function

import time
import subprocess
import os
import serial

i = 0
runcode = int(time.time())
while True:
	
	#~ subprocess.call("bias_tess 30000 && sleep 10 && bias_tess 2000", shell=True)
	
	#~ time.sleep(60*6)

	fname = "iv_partial_%i_t%i" % (runcode, i)
	t_start = time.time()
	cmd = "/home/time/b3_analysis/load_curve/ivcurve.py -c 0 1 2 3 4 5 6 7 10 11 12 14 15 16 17 20 21 22 23 24 25 26 27 --bias_start 1500 --bias_step -10 --bias_count 82 --zap_bias 1500 --zap_time 1 --settle_time 1 --settle_bias 1500 --bias_pause 0.1 --bias_final 1500 --data_mode 10 -d " + fname
	subprocess.call(cmd, shell=True)	
	t_stop = time.time()

	with open('/data/cryo/current_data/' + fname + '/' + fname + ".time", 'wb') as f:
		f.write('avg=' + str((t_start+t_stop)/2.0) + '\nstart=' + str(t_start) + '\nstop=' + str(t_stop) + '\n')

	i += 1
