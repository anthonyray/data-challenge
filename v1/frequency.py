import numpy as np
from scipy import signal, integrate
import scipy
from pyeeg import *

def compute_frequency_features_train_test(X_train,X_test,Fs):
    """Generate frequency features matrix for training set and test set

    Parameters
    ----------
    X_train : array, shape=(n,p) where p is a power of 2
        Training set
    X_test : array, shape=(m,p) where p is a power of 2
        Test set
    Fs : int, Sampling frequency

    Returns
    -------
    XX_train : array, shape=(n, p)
        Features training matrix
    X_test : array, shape=(n,p)
        Features test matrix
    """

    return compute_frequency_features(X_train,Fs),compute_frequency_features(X_test,Fs)

def compute_frequency_features(X,Fs):
    N = X.shape[1]
    XX = np.c_[
                     np.mean(X[:,3*N/8:N/2]**2,axis=1),
                     scipy.stats.skew(X,axis=1),
                     scipy.stats.kurtosis(X,axis=1),
                     np.percentile(X,75,axis=1),
                     np.apply_along_axis(bin_power,1,X,[0,10,20,30,40,50,60,70,80,90],int(Fs))
             ]
    return XX
