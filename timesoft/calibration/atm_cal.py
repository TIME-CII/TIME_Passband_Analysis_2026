# NOT USED IN CURRENT PIPELINE
# try:
#     from ryutils.plotutils import *
# except ModuleNotFoundError as e:
#     print(e)
#     print('Is ryutils installed on this system?')

import numpy as np
from scipy.interpolate import interp1d
from importlib_resources import files

def setup_simple_atm_model():
    if __name__ == '__main__':
        file_det = "/Users/rpkeenan/Dropbox/4_research/3.1_TIME_analysis/TIME-analysis/timesoft/helpers/atm_data/simple_atm_model/tau_det.npy"
        file_225 = "/Users/rpkeenan/Dropbox/4_research/3.1_TIME_analysis/TIME-analysis/timesoft/helpers/atm_data/simple_atm_model/tau_225.npy"
    else:
        file_det = files('timesoft.helpers').joinpath('atm_data/simple_atm_model/tau_det.npy')
        file_225 = files('timesoft.helpers').joinpath('atm_data/simple_atm_model/tau_225.npy')

    tau_det = np.load(file_det)
    tau_225 = np.load(file_225)
    functions = []
    for i in range(60):
        functions.append(interp1d(tau_225,tau_det[:,i]))

    def simple_atm_model(x,f,tau):
        if f>=60 or f<0:
            raise ValueError("Frequency index not recognized")
        return functions[f](tau)

    return simple_atm_model

