import sys
import time
import numpy as np
from scipy import io
import scipy.fftpack
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import math
import climsg

total_start = time.time()
climsg.welcome_message(sys.argv[0])

"""
Loading data
"""
climsg.loading_data()
start_loading_data = time.time()

dataset = io.loadmat('data_challenge.mat')
sleep_labels = {1:'N1',2:'N2',3:'N3',4:'R ',5:'W '}
X, y, X_final_test = dataset['X_train'], dataset['y_train'], dataset['X_test']


climsg.done_loading_data(time.time() - start_loading_data)

"""
Splitting data into training and test sets
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


"""
Features Construction

We have three categories of features :
    - Static Features
    - Frequency domain features
    - Wavelet domain features
"""


"""
Wavelets Features Construction
"""
from wavelets import *

def compute_wavelets_features_train_test(X_train,X_test):
    """Generate features matrix for training set and test set

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

    return compute_wavelets_features(X_train), compute_wavelets_features(X_test)

def compute_wavelets_features(X):
    XX = np.c_[
                     np.apply_along_axis(db_dwt_4,1,X)
              ]
    return XX


"""
Frequency Features Construction
"""

from frequency import *
from energy import *

"""
Static Features Construction
"""

from static import *

"""
Features Construction
"""
climsg.features_building_init()
# Preparing data for wavelet features extraction

# Padding data
X_train_w = np.pad(X_train,((0,0),(0,2192)),mode="constant")
X_test_w = np.pad(X_test,((0,0),(0,2192)),mode="constant")

# Preparing data for frequency features extraction

N = X_train.shape[1]
T = 30.0 / float(N)  # Sampling period
f = 1.0 / T # Sampling freq
f_s = int(f)

X_train_freq = scipy.fftpack.fft(X_train,N,axis=1)
X_train_freq = (1.0 / float(N) ) * np.apply_along_axis(np.abs,1,X_train_freq)
X_train_freq = X_train_freq[:,:N/2]

X_test_freq = scipy.fftpack.fft(X_test,axis=1)
X_test_freq = (1.0 / N) * np.apply_along_axis(np.abs,1,X_test_freq)
X_test_freq = X_test_freq[:,:N/2]

freqs = scipy.fftpack.fftfreq(N,T)
freqs = freqs[:N/2]

# Building features for frequency :
start_freq = time.time()

XX_train_freq,XX_test_freq = compute_frequency_features_train_test(X_train_freq,X_test_freq,f_s)

XX_train_en, XX_test_en = compute_energy_features_train_test(X_train,X_test,f)

nb_freq_features = XX_train_freq.shape[1] + XX_train_en.shape[1]


climsg.freq_features(time.time()-start_freq,nb_freq_features)

# Building features for wavelets
start_wav = time.time()

XX_train_wav, XX_test_wav = compute_wavelets_features_train_test(X_train_w,X_test_w)

nb_wav_features = XX_train_wav.shape[1]

climsg.wav_features(time.time() - start_wav,nb_wav_features)

# Building static features
start_static = time.time()

XX_train_stat,XX_test_stat = compute_static_features_train_test(X_train,X_test)

nb_stat_features = XX_train_stat.shape[1]
end_static = time.time()
climsg.stat_features(end_static-start_static,nb_stat_features)
# Combining features

XX_train = np.c_[XX_train_stat,XX_train_freq,XX_train_wav,XX_train_en]
XX_test = np.c_[XX_test_stat,XX_test_freq,XX_test_wav,XX_test_en]

"""
Training classifier
"""
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

models = [SVC(),KNeighborsClassifier(n_neighbors=20),AdaBoostClassifier(),LogisticRegression(),RandomForestClassifier()]
predictions = list()
for clf in models:
    clf.fit(XX_train,y_train)


"""
Predict classes on test set

"""
climsg.predict()
start_pred = time.time()


for clf in models:
    predictions.append(clf.predict(XX_test))


climsg.done_predicting(time.time() - start_pred)

"""
Report for all classifiers
"""
from sklearn.metrics import classification_report
climsg.report()

for y_pred in predictions:
    print classification_report(y_pred,y_test,
                            target_names=[l for l in sleep_labels.values()])


"""
Visualize features

"""
if len(sys.argv) == 2:
    for i in range(XX_train.shape[1]):
        plt.close('all')
        plt.hist(XX_train[np.where(y_train=='N1')][:,i],bins = 20, alpha=0.5, label='N1')
        plt.hist(XX_train[np.where(y_train=='N2')][:,i],bins = 20, alpha=0.5, label='N2')
        plt.hist(XX_train[np.where(y_train=='N3')][:,i],bins = 20, alpha=0.5, label='N3')
        plt.hist(XX_train[np.where(y_train=='R ')][:,i],bins = 20, alpha=0.5, label='R ')
        plt.hist(XX_train[np.where(y_train=='W ')][:,i],bins = 20, alpha=0.5, label='W ')
        plt.legend(loc='upper right')
        plt.show()

"""
Export prediction
"""

def export(X,y,X_pred):
    # Data Preparation
    X_w = np.pad(X,((0,0),(0,2192)),mode="constant")  # Wavelets
    X_pred_w = np.pad(X_pred,((0,0),(0,2192)),mode="constant")

    N = X.shape[1]  # Frequency
    T = 30.0 / float(N)
    X_f = scipy.fftpack.fft(X,axis=1)
    X_pred_f = scipy.fftpack.fft(X_pred,axis=1)

    XX_freq = compute_frequency_features(X_f)
    XX_pred_freq = compute_frequency_features(X_pred_f)

    XX_wav = compute_wavelets_features(X_w)
    XX_pred_wav = compute_wavelets_features(X_pred_w)

    XX_stat = compute_static_features(X)
    XX_pred_stat = compute_static_features(X_pred)

    #XX_en = compute_energy_features(X)
    #XX_pred_en = compute_energy_features(X_pred)

    XX = np.c_[XX_stat,XX_freq,XX_wav]#,XX_en]
    XX_pred = np.c_[XX_pred_stat,XX_pred_freq,XX_pred_wav]#,XX_pred_en]

    classifier = RandomForestClassifier()
    classifier.fit(XX,y)
    y_pred = classifier.predict(XX_pred)
    np.savetxt('y_pred.txt', y_pred, fmt='%s')

def export_train_test(X_train,y_train,X_pred):
    # Data Preparation
    X, X_test, y, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    X_w = np.pad(X,((0,0),(0,2192)),mode="constant")  # Wavelets
    X_pred_w = np.pad(X_pred,((0,0),(0,2192)),mode="constant")

    N = X.shape[1]  # Frequency
    T = 30.0 / float(N)
    f = 1.0 / T
    f_s = int(f)
    X_f = scipy.fftpack.fft(X,axis=1)
    X_pred_f = scipy.fftpack.fft(X_pred,axis=1)

    XX_freq = compute_frequency_features(X_f,f_s)
    XX_pred_freq = compute_frequency_features(X_pred_f,f_s)

    XX_wav = compute_wavelets_features(X_w)
    XX_pred_wav = compute_wavelets_features(X_pred_w)

    XX_stat = compute_static_features(X)
    XX_pred_stat = compute_static_features(X_pred)

    XX_en = compute_energy_features(X)
    XX_pred_en = compute_energy_features(X_pred)

    XX = np.c_[XX_stat,XX_freq,XX_wav,XX_en]
    XX_pred = np.c_[XX_pred_stat,XX_pred_freq,XX_pred_wav,XX_pred_en]

    classifier = RandomForestClassifier()
    classifier.fit(XX,y)
    y_pred = classifier.predict(XX_pred)
    np.savetxt('y_pred.txt', y_pred, fmt='%s')

if len(sys.argv) == 3:
    climsg.export()
    start_export = time.time()
    export_train_test(X,y,X_final_test)
    end_export = time.time()
    climsg.done_export(end_export - start_export)



"""
Finishing
"""
total_end = time.time()
climsg.goodbye(total_end - total_start)
