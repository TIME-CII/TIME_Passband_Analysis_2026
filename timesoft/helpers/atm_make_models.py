import numpy as np
from scipy.signal import convolve
from scipy.interpolate import interp1d
from importlib_resources import files

def make_simple_atm_model():
    file = files('timesoft.helpers').joinpath('atm_data/am_model/kpam_scale_1.00.dat')
    nu,tau0,_ = np.genfromtxt(file,unpack=True)
    kernel_fwhm = 2 # GHz
    kernel = np.exp(-4*np.log(2)*((nu-np.mean(nu))/kernel_fwhm)**2)
    kernel /= np.sum(kernel)

    scale = np.arange(0.1,2.01,0.05)
    tau_225 = []
    tau_det = []
    for s in scale:
        file = files('timesoft.helpers').joinpath('atm_data/am_model/kpam_scale_{:.2f}.dat'.format(s))
        _,tau,_ = np.genfromtxt(file,unpack=True)
        tau = convolve(tau,kernel,mode='same')
        tau_func = interp1d(nu,tau)
        tau_225.append(tau_func(225))
        tau_det.append(tau_func(frequency_listing[:,1]))

    # First axis specifies frequency, second axis specifies tau
    tau_det = np.array(tau_det)
    np.save(files('timesoft.helpers').joinpath('atm_data/simple_atm_model/tau_det.npy'),tau_det)
    np.save(files('timesoft.helpers').joinpath('atm_data/simple_atm_model/tau_225.npy'),tau_225)