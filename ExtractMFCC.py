"""
this file is used to extract the MFCC features from the audio files
"""
import numpy as np
import pandas as pd
from init import PreProcessing
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import warnings

warnings.filterwarnings('ignore')


# %%
def extract_mfcc(filename):
    """
    Args:
        filename:

    Returns:
        the MFCC features of the audio file

    """
    data, sampling_rate = librosa.load(filename, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T, axis=0)
    return mfccsscaled

def getMFCCFeature():
    """
    Returns:
        the MFCC features of the audio files
    """
    pre = PreProcessing()
    fileList = pre.getFileNameList()
    labelList = pre.getLabelList()
    mfccList = [extract_mfcc(file) for file in fileList]
    X = np.expand_dims(mfccList, axis=-1)
    print(X.shape)
    return mfccList, labelList

def plotMFCC(address):
    """
    This function is used to plot the MFCC features of the audio file
    Args:
        address:

    Returns:

    """
    data, sampling_rate = librosa.load(address, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=40)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()