import numpy as np
from scipy import signal, integrate

def mpf(p,f):
    """
    Parameters
    ----------
    p : 1D-array of power
    f : 1D-array of frequencies

    Returns
    -------

    mpf : float
        Mean power frequency in Hz.

    """
    Area = integrate.cumtrapz(p, f, initial=0)
    Ptotal = Area[-1]
    mpf = integrate.trapz(f * p, f) / Ptotal

    return mpf

def fmax(p,f):
    """
    Returns
    -------

    fmax : float
        Maximum power frequency in Hz.
    """
    return f[np.argmax(p)]

def Ptotal(p,f):
    """
    Returns
    -------

    Ptotal : float
        Total power in `units` squared.

    """
    Area = integrate.cumtrapz(p, f, initial=0)
    Ptotal = Area[-1]
    return Ptotal

def freq_percentile(p,f,percentile):
    Area = integrate.cumtrapz(p, f, initial=0)
    Ptotal = Area[-1]
    inds = [0]
    Area = 100 * Area / Ptotal  # + 10 * np.finfo(np.float).eps
    for i in range(1, 101):
        inds.append(np.argmax(Area[inds[-1]:] >= i) + inds[-1])
    fpcntile = f[inds]
    return fpcntile[percentile]


def compute_energy_features(X,fs=1.0, window='hanning', nperseg=None, noverlap=None, nfft=None,
        detrend='constant'):
    if not nperseg:
        nperseg = X.shape[1] / 2
    f,P = signal.welch(X, fs, window, nperseg, noverlap, nfft, detrend,axis=1)
    X_energy = np.c_[np.apply_along_axis(mpf,1,P,f),
                     np.apply_along_axis(fmax,1,P,f),
                     np.apply_along_axis(Ptotal,1,P,f),
                     np.apply_along_axis(freq_percentile,1,P,f,10),
                     np.apply_along_axis(freq_percentile,1,P,f,90)
    ]
    return X_energy

def compute_energy_features_train_test(X_train,X_test,fs=1.0, window='hanning', nperseg=None, noverlap=None, nfft=None,
        detrend='constant'):
    return compute_energy_features(X_train,fs,window,nperseg,noverlap,nfft,detrend='constant'),compute_energy_features(X_test,fs,window,nperseg,noverlap,nfft,detrend='constant')
