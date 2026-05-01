import numpy as np

def spec1D(y,signal_rate):
    ''' This function calculates the power spectrum of a time stream, y, sampled at rate = signal_rate. 
    
    Parameters

    ----------
    y: list
        The timestream
    
    signal_rate: int    
        The sampling rate
    
    Returns

    ----------
    
    The frequencies comprising the signal
    
    The power spectral density profile.   
    
    
    '''
    
    dt=1/signal_rate
    ft = np.fft.rfft(y)*dt/len(y)
    psd = 2*np.abs(ft)**2*len(y)/dt
    return np.fft.rfftfreq(len(y),d=dt),psd
