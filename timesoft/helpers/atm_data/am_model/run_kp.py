import os
import numpy as np

scale = np.arange(0.05,2.01,0.05)
for s in scale:
	print("Doint {}".format(s))
	os.system("am kp.amc 150 350 0 {} > kpam_scale_{:.2f}.dat".format(s,s))

