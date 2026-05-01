"""
Overall, this code performs PCA on detectors, per value of feedhorn number (x), on the detectors left behind after the detector cuts. In the end, the timestreams and PSD's of the original time streams and the timestreams with a quadratic polynomial subtracted and PCA performed are plotted. The role of the individual functions is also specified below.
"""


from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from timesoft.timestream import timestream_tools as timeTools
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import numpy as np


def detector_sublist(det_x,detector_numbers_list):
    """This function returns all the values of 'f' corresponding to a certain value of 'x' from the detectors left after the first and second dectector cuts. 

    Parameters

    ----------
    det_x: int
        The feedhorn number under consideration.
       
    detctor_numbers_list: list
         A list, written in (x,f) format, of all the detectors left after the detector cuts.
       


    Returns

    ----------
    detector_sublist: list
        A list of all frequency channel labels ('f') corresponding to the chosen feedhorn number ('det_x') that are left after the detector cuts.

    """    
    detector_sublist=[]
    for i in range (len(detector_numbers_list)):
        if detector_numbers_list[i][0]==det_x:
            detector_sublist.append(detector_numbers_list[i][1])
    if len(detector_sublist)==0:
        raise Exception("This detector was discarded during detector cuts.")
    else:
        return detector_sublist
        
    

def xwise_pca(det_x,n_pca,detector_numbers_list): 
    """ This function does a quadratic polynomial subtraction from the timestream and then does PCA, on all 'f' labels, corresponding to a fixed 'det_x', keeping 'n_pca' principal components.

    Parameters

    ----------
    det_x: int
        The feedhorn number under consideration.

    n_pca: int
        The number of principal components to keep
     
    detector_numbers_list: list
        A list, written in (x,f) format, of all the detectors left after the detector cuts.
                        
       
       

    Returns

    ----------

    det_x: int
        The feedhorn number under consideration.
    
    detector_sublist(det_x,detector_numbers_list): list
        A list of all frequency channel labels ('f') corresponding to the chosen feedhorn number ('det_x') that are left behind.
    
    X_orig: list
        A list of the original timestreams for all the frequency channels correspond to the chosen feedhorn number ('det_x') that are left behind.
        
    X_pca: list
        A list of the timestreams after a quadratic polynomial subtraction and pca.   
        
    z: list
        A list of the timestreams after the just the quadractic polynormial subtraction.
    
""" 
    if (any(x[0]==det_x for x in detector_numbers_list)) and (sum([x==det_x for (x,y) in detector_numbers_list]) > n_pca):
        detrendeddf=pd.DataFrame()
        ordatadf=pd.DataFrame() 
        ordata=[]
        detrended=[]
        for det_f in detector_sublist(det_x,detector_numbers_list):
            idx = ts.get_xf(det_x, det_f)
            y = detData[idx]              
            c= np.polyfit(t,y,2) #least squares fit to a quadratic polynomial 
            detrendedts = y -(c[0]*t**2 + c[1]*t + c[2])  #Detrended time stream
            ordata.append(y)
            detrended.append(detrendedts)        
            detrendeddf.insert(detrendeddf.shape[1],'({}, {})'.format(det_x,det_f), list(detrendedts))
            ordatadf.insert(ordatadf.shape[1],'({}, {})'.format(det_x,det_f), list(y))             
        z=np.transpose(detrended)
        X_orig=np.transpose(ordata)    
        scaler = StandardScaler() 
        scaler.fit(z) 
        scaled_data = scaler.transform(z) 
        pca = PCA(n_components=n_pca)
        pca.fit(scaled_data) 
        x_pca = pca.transform(scaled_data)
        X_ori0=pca.inverse_transform(x_pca)
        X_pca= scaler.inverse_transform(X_ori0)
        return X_pca, X_orig, det_x,detector_sublist(det_x,detector_numbers_list),z
    elif (sum([x==det_x for (x,y) in detector_numbers_list]) <= n_pca):
        raise Exception("Not enough detectors for PCA")        
    else:
        raise Exception("This detector was discarded during detector cuts") 
    
def timestream_comparison(det_x,detector_numbers_list): 
    """
    This function allows a comparison of original timestreams with  timestreams after polynomial subtraction and PCA.

    Parameters

    ----------
    det_x: int
        The feedhorn number under consideration.
    
    detector_numbers_list: list
        A list, written in (x,f) format, of all the detectors left after the detector cuts.

    
    Returns

    ----------
    Plots of the original time streams and the time streams after polynomial subtraction and pca. 

    """
    X_pca,X_orig, det_x,detector_sublist_det_x,z=xwise_pca(det_x,n_pca,detector_numbers_list)
    difference=np.subtract(X_pca,z)
    
    for detectorno in detector_sublist_det_x: 
           
       
        plt.figure()           
        plt.plot(X_orig[:,detector_sublist_det_x.index(detectorno)],'r--', label='Original timestream') #can comment this out to better visualize the fluctations in the timestreams we ge after quadratic subtraction and PCA.
        plt.plot(difference[:,detector_sublist_det_x.index(detectorno)],'b--', label='Timestream after quadratic subtraction and PCA')  
        plt.xlabel('Time stamps')        
        plt.title('Timestreams for (x={}, f={})'.format(det_x, detectorno))
        plt.legend()
        plt.show()





def power_spectrum(det_x,detector_numbers_list, power_spectrum_func, *kwargs): 
    """
    This function computes power spectra using your favorite method (e.g Welch or spec 1D, which you specify in place of power_spectrum_func below) and then compares the PSD's of the original timestreams with  timestreams after polynomial subtraction and PCA.

    Parameters

    ----------
    det_x: int
        The feedhorn number under consideration.
    
    detector_numbers_list: list
        A list, written in (x,f) format, of all the detectors left after the detector cuts.

    power_spectrum_func: function
        The function to be used to compute the power spectrum. Typically this is either Welch's method or spec1D (defined in a separate module in timesoft). 
    
    *kwargs: key word arguments
    The arguments needed for power_spectrum_func to compute power spectra.
    
    Returns

    ----------
    Plots of power spectral densities against frequencies of original time streams and the time streams after polynomial subtraction and pca. 

    """
    X_pca,X_orig, det_x,detector_sublist_det_x,z=xwise_pca(det_x,n_pca,detector_numbers_list)
    difference=np.subtract(X_pca,z)
    
    for detectorno in detector_sublist_det_x: 
           
       # (f, S) = power_spectrum_func(X_pca[:,detector_sublist_det_x.index(detectorno)], *kwargs)  #Thought this was useful but not actually necessary.    
        (f2, S2) = power_spectrum_func(X_orig[:,detector_sublist_det_x.index(detectorno)], *kwargs)
        (f3, S3) = power_spectrum_func(difference[:,detector_sublist_det_x.index(detectorno)], *kwargs)
        plt.figure()
       # plt.loglog(f, S,'k') #Thought this was useful but not actually necessary.    
        plt.loglog(f2, S2,'r--',label='PSD of original timestream')    
        plt.loglog(f3, S3,'g--',label='PSD of timestream after quadratic subtraction and PCA')   
        plt.xlabel('f')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('PSD for (x={}, f={})'.format(det_x, detectorno))
        plt.legend()
        plt.show()
