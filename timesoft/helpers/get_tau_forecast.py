import numpy as np 
import os 
from scipy.interpolate import interp1d

def get_tau(date, time, data_path, version='2024.dev.1'):
    '''
    get_tau - extracts the value of tau at 225 GHz from Dan's weather forecast.

    Parameters
    -----------
    date : (str) the date of the observation must be in (YYYYMMDD) format to match output files
    time : (str) the local time (MST) of the observation in HH:MM:SS form. 

    Returns
    --------
    Tau225 : Tau at the given time.
    '''

    if version == '2024.dev.1':
        tau_data = np.loadtxt('/data/vaughan/2025_test/tau_measurements.txt')

        tau_interp = interp1d(tau_data[:,0], tau_data[:,1])

        tau = np.nanmean(tau_interp(time))

        print(tau)
        return tau

    else:
        files = os.listdir(data_path)
        good_files = [date in file for file in files]
        correct_dates = np.array(files)[good_files]
        times = np.array([0,6,12,18])
        time_h = float(time[0:2]) + float(time[3:5]) / 60 + float(time[6:8]) / 3600
        time_greater = time_h > times
        time_str = np.array(['00:00:00','06:00:00','12:00:00','18:00:00'])
        corr_time = time_str[time_greater][-1]
        good_files = [corr_time in corr_date for corr_date in correct_dates]
        corr_file = correct_dates[good_files][0]
        data_times = np.loadtxt(data_path + corr_file, usecols=(0), dtype='str')
        data_taus  = np.loadtxt(data_path + corr_file, usecols=(1), dtype='float')
        good_date = np.array([date in data_time for data_time in data_times])
        good_dates = data_times[good_date]
        good_tau_dates = data_taus[good_date]
        time_hs = np.array([float(f[9:11]) for f in good_dates])
        time_greater = time_h > time_hs
        corr_time = good_dates[time_greater][-1]
        corr_tau = good_tau_dates[time_greater][-1]

if __name__ == '__main__':
    get_tau('20211201', '04:20:13', '/data/time/tau_forecasts/')
