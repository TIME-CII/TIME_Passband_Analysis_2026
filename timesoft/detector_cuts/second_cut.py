import pandas as pd
import numpy as np
from timesoft.timestream import timestream_tools as timeTools
from ast import literal_eval

def RMS_filter(good_detectors, ndevs):
    """
    This function performs RMS filtering: it discards any detectors whose time stream has RMS value that is outside +/- 'ndevs'-number of  standard deviations from the mean of the RMS values of all detectors.

    Parameters

    ----------

    good_detectors: list
        The detectors left behind after the cut performed in the 'first_cut' module.
 
    ndevs: int
        number of standard deviations from mean RMS to keep.

    Returns

    ----------

    detector_numbers_list: list
        a list, written in (x,f) format, of all the detectors left after RMS filtering.

    """
    ordatadf=pd.DataFrame()
    for (det_x,det_f) in good_detectors:
        idx = ts.get_xf(det_x, det_f)
        y = detData[idx]    
        ordatadf.insert(ordatadf.shape[1],'({}, {})'.format(det_x,det_f), list(y))
    RMSdf=pd.DataFrame((np.mean(ordatadf.apply(lambda x: x**2,axis=0))).apply(lambda x: x**1/2)) #computing RMS
    RMSdf.columns=['RMS values']
    stdev= RMSdf['RMS values'].std()
    RMSdf['Dev. from mean RMS']= RMSdf- RMSdf.mean()
    dffinal=RMSdf[(RMSdf['Dev. from mean RMS'] < ndevs*stdev) & (RMSdf['Dev. from mean RMS'] > -1* ndevs*stdev)]
    detector_numbers= dffinal.index.to_list()
    detector_numbers_list = [literal_eval(x) for x in detector_numbers] #Getting detector numbers written in the format (x,f)
    return detector_numbers_list
