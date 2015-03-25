# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:42:22 2015

@author: anthonyreinette
"""

import pandas as pd
import numpy as np
from scipy import io
import scipy.fftpack
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

dataset = io.loadmat('data_challenge.mat')

X, y, X_final_test = dataset['X_train'], dataset['y_train'], dataset['X_test']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

N = X_train.shape[1]
T = 30.0 / float(N)  # Période d'échantillonage
X_train_freq = scipy.fftpack.fft(X_train,axis=1)
X_train_freq = (2.0 / N) * np.apply_along_axis(np.abs,1,X_train_freq)

X_test_freq = scipy.fftpack.fft(X_test,axis=1)
X_test_freq = (2.0 / N) * np.apply_along_axis(np.abs,1,X_test_freq)

X_freq = scipy.fftpack.fftfreq(N,T)

# Downsampling one time series
np.mean(X_train[0].reshape(-1,50),axis=1)

fig,axes = plt.subplots(nrows=1,ncols=2)
axes[0].plot(X_train[0])
axes[1].plot(np.mean(X_train[0].reshape(-1,50),axis=1))

# Downsampling the whole matrix

def downsample_mean(a,factor):
    """
    Aggregates a time series with a factor. If the original time series is 6000 points, 
    the aggregated time series will have 6000 / factor points. 
    Example : aggregate(X_train,50) will return a matrix of time series with (6000/50) = 120 points
    """
    nrows,ncols = a.shape
    arr = np.mean(a[0].reshape(-1,factor),axis=1)
    for row in a[1:]:    
        arr = np.vstack((arr,np.mean(row.reshape(-1,factor),axis=1)))
    return arr

test = downsample_mean(X_train,50)

fig,axes = plt.subplots(nrows=1,ncols=2)
axes[0].plot(X_train[3567])
axes[1].plot(test[3567])
    