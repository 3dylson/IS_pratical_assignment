# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''
import wfdb
import os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
from scipy import signal as sig
from scipy.signal import medfilt
import scipy.io as sio

import warnings
warnings.filterwarnings('ignore')

def butterworth_high_pass(x, cut_off, order, sampling_freq):
    #
    nyq_freq = sampling_freq / 2
    digital_cutoff = cut_off / nyq_freq
    #
    b, a = sig.butter(order, digital_cutoff, btype='highpass')
    y = sig.lfilter(b, a, x, axis = -1)

    return y

def butterworth_notch(x, cut_off, order, sampling_freq):
    #
    cut_off = np.array(cut_off)
    nyq_freq = sampling_freq / 2
    digital_cutoff = cut_off / nyq_freq
    #
    b, a = sig.butter(order, digital_cutoff, btype='bandstop')
    y = sig.lfilter(b, a, x, axis = -1)
    #
    return y
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, lfilter, lfilter_zi, filtfilt, sosfreqz, resample
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def butter_bandpass_filter(df_data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    ecg_sig = np.zeros([df_data.shape[0],df_data.shape[1]])
    for i in range(df_data.shape[1]):
        data = df_data[:,i]
        y = sosfilt(sos,
                    data)  # Filter data along one dimension using cascaded second-order sections. Using lfilter for each second-order section.
        ecg_sig[:,i] = y
    return ecg_sig

def butter_bandpass_forward_backward_filter(df_data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    ecg_sig = np.zeros([df_data.shape[0],df_data.shape[1]])
    for i in range(df_data.shape[1]):
        data = df_data[:,i]
        y = sosfiltfilt(sos,
                        data)  # Apply a digital filter forward and backward to a signal.This function applies a linear digital filter twice, once forward and once backwards. The combined filter has zero phase and a filter order twice that of the original.
        ecg_sig[:,i] = y
    return ecg_sig

def scaling(X, sigma=0.05):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def verflip(sig):
    '''
    :param sig:
    :return:
    '''
    return sig[::-1, :]

def shift(sig, interval=20):
    '''
    :param sig:
    :return:
    '''
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))/100
        sig[:, col] += offset
    return sig

def transform(sig, train=True):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.3: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)

        if np.random.randn() > 0.3:
            sig = butter_bandpass_filter(sig,0.05,46,256)
    else:
        pass
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig

def transform_beat(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)

    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig

import scipy.io as scio
class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """
    def __init__(self, data_path, data_dir,train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(data_path) #config.train_data
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])
        self.FS = 500
        self.SIGLEN = 500 * 10
        self.train_dir = data_dir#config.train_dir
        self.test_dir = data_dir#config.test_dir

    def __getitem__(self, index):
        # fid = self.data[index]
        fid = self.data[index]

        #method three

        file = fid.split('/')[-1]
        if self.train:
            file_path = os.path.join(self.train_dir, file)
            #df = scio.loadmat(file_path)["val"].T/1000
            sig = scio.loadmat(file_path)["val"]#.T
        else:
            file_path = os.path.join(self.test_dir, file)
            # df = scio.loadmat(file_path)["val"].T/1000
            sig = scio.loadmat(file_path)["val"]#.T

        #print(df.shape)
        x = transform(sig.T, self.train)

        target = np.zeros(config.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return x, target #beat,

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    print(d[0])