"""
Overall, this code compares the NEI's of the detectors kept and the detectors discarded after the detector cuts. The role of the individual functions is also specified below

"""
import numpy as np
import matplotlib.pyplot as plt


def discarded(df2, detector_numbers_list):
    """
    This function returns a list, in (x,f) format, of all the detectors that are discarded after the detector cuts.


    Parameters

    ----------
    df2: data frame
        a data frame containing all the data in '20221212_noisepsd_nonPID.txt.csv'
    
    detector_numbers_list: list
        a list of detectors kept after the second cut

    Returns

    ----------
    ldiscarded: list
        a list, in (x,f) format, of all the detectors that are discarded after the data cuts.

    """
    ldiscarded=[] 
    for j in range (len(df2)):
        if tuple([int(df2.iloc[j]['x']), int(df2.iloc[j]['f'])]) not in detector_numbers_list:
            ldiscarded.append(tuple([int(df2.iloc[j]['x']), int(df2.iloc[j]['f'])]))
    return ldiscarded             
  

def NEIdiscarded(ldiscarded, df2):
    """
    This function returns the xf labels, stored in list called 'xd', and NEI's, stored in a list called 'yd', of the detectors that were discarded during the data cuts.  


    Parameters

    ----------
    ldiscarded: list 
        a list, in (x,f) format, of all the detectors that are discarded after the data cuts.
    
    df2: data frame
        a data frame containing all the data in '20221212_noisepsd_nonPID.txt.csv'
    
    Returns

    ----------
    xd: list
        a list, in (x,f) format, of all the detectors that were discarded during the detector cuts.
    
    yd: list
        a list of the NEI's all the detectors that were discarded during the detector cuts


    """
    NEIdiscarded={}
    for j in range (len(ldiscarded)):
        xcoord= ldiscarded[j][0]
        fcoord=ldiscarded[j][1]   
        NEIdiscarded['x{} f{}'.format(xcoord, fcoord)]=list(df2[(df2['x']==xcoord) & (df2['f']==fcoord)]['NEI'])
    xd=list(NEIdiscarded.keys()) #xf labels of the detectors that are discarded.
    yd=np.reshape((list(NEIdiscarded.values())),-1) #NEI's of the detectors that are discarded.
    
    return xd, yd



def NEIkept(detector_numbers_list, df2):
    """
    This function returns the xf labels, stored in list called 'xk', and NEI's, stored in a list called 'yk', of the detectors that were kept during the data cuts.


    Parameters

    ----------
    detector_numbers_list: list
        a list, written in (x,f) format, of all the detectors left after the data cuts.
    
    df2: data frame
        a data frame containing all the data in '20221212_noisepsd_nonPID.txt.csv'
    
    Returns

    ----------
    xk: list
        a list, in (x,f) format, of all the detectors that were kept during the detector cuts.
    
    yk: list
        a list of the NEI's all the detectors that were kept during the detector cuts
    
    """
    NEIkept={}
    for j in range (len(detector_numbers_list)):
        xcoord= detector_numbers_list[j][0]
        fcoord=detector_numbers_list[j][1]
        NEIkept['x{} f{}'.format(xcoord, fcoord)]=list(df2[(df2['x']==xcoord) & (df2['f']==fcoord)]['NEI'])
    xk=list(NEIkept.keys()) #xf labels of the detectors that are kept.
    yk= np.reshape((list(NEIkept.values())),-1) #NEI's of the detectors that are kept.
    return xk, yk
    


def det_plots(xk,xd,yk,yd):
    """
    This function makes plots of the NEI's of the discarded detectors and the kept detectors.

    Parameters

    ----------
    xk: list
        a list, in (x,f) format, of all the detectors that were kept during the detector cuts.
    
    xd: list
        a list, in (x,f) format, of all the detectors that were discarded during the detector cuts.
    
    yk: list
        a list of the NEI's all the detectors that were kept during the detector cuts.

    
    yd: list
        a list of the NEI's all the detectors that were discarded during the detector cuts.
    
    Returns

    ----------
    various plots comparing the (logarithm of) NEI's of the discarded detectors and the kept detectors.

    """
 
    plt.figure()
    plt.plot(np.log(yk), label='kept NEI\'s') 
    plt.plot(np.log(yd), label='Discarded NEI\'s')
    plt.title('Comparison of NEI\'s-1')
    plt.ylabel('Log[NEI]')
    plt.xlabel('arbitrarily assigned detector indices')
    plt.legend()
    plt.show()
    plt.figure() 
    plt.bar(xk, np.log(yk), label='Kept NEI\'s',align='edge') 
    plt.bar(xd, np.log(yd), label='Discarded NEI\'s',align='edge') 
    plt.ylabel('Log[NEI]')
    plt.xlabel('detector labels')
    plt.title('Comparison of NEI\'s-2')
    plt.subplots_adjust(bottom=0.2)
    plt.legend()
    plt.show()
    plt.figure()
    plt.hist(np.log(yk), alpha=0.5, label='Kept NEI\'s',orientation='vertical') 
    plt.hist(np.log(yd), alpha=0.5, label='Discarded NEI\'s',orientation='vertical') 
    plt.ylabel('Number of detectors')
    plt.xlabel('Log[NEI]')
    plt.title('Comparison of NEI\'s-3')
    plt.legend()
    plt.show()
