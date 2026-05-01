import pandas as pd
import numpy as np


def intersection(df, df2):
    """
    This function computes the intersection of Sukhman and Dongwoo's lists (the two inputs written below) and stores it in'good_detectors'

    Parameters

    ----------
    iteration24.csv : csv file
        Sukhman's list of good detectors
    
    20221212_noisepsd_nonPID.txt.csv: csv file
        Dongwoo's jupiter callibrations
   
    Returns

    ----------
    good_detectors: list
        a list containing the intersection described above.
    
    detectors_DC: list
        a list containing all the detectors in Dongwoo's callibration list.
    
    df2: data frame
        a data frame containing all the data in '20221212_noisepsd_nonPID.txt.csv'

    """
    good_detectors_init=[]

    for i in range(df.shape[0]):
        good_detectors_init.append(tuple([int(df.iloc[i]['det_x']), int(df.iloc[i]['det_f'])]))
    

    detectors_DC=[] 
    for j in range(df2.shape[0]):
        detectors_DC.append(tuple([int(df2.iloc[j]['x']), int(df2.iloc[j]['f'])]))


    good_detectors=[] 
    for j in range(df2.shape[0]):
        if tuple([int(df2.iloc[j]['x']), int(df2.iloc[j]['f'])]) in good_detectors_init:
            good_detectors.append(tuple([int(df2.iloc[j]['x']), int(df2.iloc[j]['f'])]))
    return good_detectors, detectors_DC, df2
    

