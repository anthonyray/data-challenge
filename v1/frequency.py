import numpy as np
from scipy import signal, integrate
import scipy

def compute_frequency_features_train_test(X_train,X_test):
    """Generate frequency features matrix for training set and test set

    Parameters
    ----------
    X_train : array, shape=(n,p) where p is a power of 2
        Training set
    X_test : array, shape=(m,p) where p is a power of 2
        Test set

    Returns
    -------
    XX_train : array, shape=(n, p)
        Features training matrix
    X_test : array, shape=(n,p)
        Features test matrix
    """

    return compute_frequency_features(X_train),compute_frequency_features(X_test)

def compute_frequency_features(X):
    N = X.shape[1]
    XX = np.c_[
                     np.mean(X[:,3*N/8:N/2]**2,axis=1),
                     scipy.stats.skew(X,axis=1),
                     scipy.stats.kurtosis(X,axis=1),
                     np.percentile(X,75,axis=1)
             ]
    return XX
