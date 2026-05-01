from importlib_resources import files
from numpy import genfromtxt
import numpy as np

def f_ind_to_val(i):
    if np.any(~np.isin(i,np.arange(60))):
        raise ValueError("invalid index")
    else:
        file = files('timesoft.helpers').joinpath('center_freq.txt')
        chan = genfromtxt(file,skip_header=1)
        return chan[i,1]
