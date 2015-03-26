# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:23:38 2015

@author: anthonyreinette
"""

import pandas as pd
import numpy as np
from scipy import io
import scipy.fftpack
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import scipy.signal

# Loading data

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

# Filtering data
N = 1001
Fc = 10
Fs = 200
# provide them to firwin

h = scipy.signal.firwin( numtaps=N, cutoff=50, nyq=Fs/2, window='hamming')

y = scipy.signal.lfilter( h, 1.0, X_train_freq[0]) # 'x' is the time-series data you are filtering

def low_pass_filter(xz):
    """
    Filter the matrix
    """
    return scipy.signal.lfilter(h, 1.0, xz)
    
X_train_freq = np.apply_along_axis(low_pass_filter,1,X_train_freq)

XX_train = np.c_[
                 np.std(X_train, axis=1), 
                 np.max(X_train,axis=1),                 
                 np.min(X_train,axis=1)/np.max(X_train,axis=1),
                 np.mean(X_train_freq[:,0:N/8]**2,axis=1),
                 np.mean(X_train_freq[:,N/8:2*N/8]**2,axis=1),
                 np.mean(X_train_freq[:,2*N/8:3*N/8]**2,axis=1),
                 np.mean(X_train_freq[:,3*N/8:N/2]**2,axis=1),
                 np.percentile(X_train_freq,70,axis=1),
                 np.percentile(X_train_freq,20,axis=1),
                 np.mean(X_train_freq**2,axis=1),
                 scipy.stats.skew(X_train_freq,axis=1)]

XX_test = np.c_[
                np.std(X_test, axis=1),   
                np.max(X_test,axis=1),              
                np.min(X_test,axis=1)/np.max(X_test,axis=1),                
                np.mean(X_test_freq[:,0:N/8]**2,axis=1),
                np.mean(X_test_freq[:,N/8:2*N/8]**2,axis=1),
                np.mean(X_test_freq[:,2*N/8:3*N/8]**2,axis=1),
                np.mean(X_test_freq[:,3*N/8:N/2]**2,axis=1),
                np.percentile(X_test_freq,70,axis=1),
                np.percentile(X_test_freq,20,axis=1),
                np.mean(X_test_freq**2,axis=1),
                scipy.stats.skew(X_test_freq,axis=1)]

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
#clf = KNeighborsClassifier(n_neighbors=15)

y_pred = clf.fit(XX_train, y_train).predict(XX_test)
score = clf.score(XX_test,y_test)
print score

