# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:58:08 2021

@author: kibong
"""
# In[]
import torch
from torch.utils.data import Dataset
import numpy as np
from random import choice, choices
import cv2

from scipy import signal
from scipy.fft import fftshift
# from scipy.signal import decimate
import matplotlib.pyplot as plt

import librosa

from AAA import Wav2Vec2Tokenizer
# In[]

class custom_dataset(Dataset):
    def __init__(self, data, Moving=True): 
        self.Moving = Moving
        self.X_train = data['radar_data']
        self.X_train = self.X_train.squeeze()
        # self.X_time = data['t_radar']
        self.PRF = data['PRF']
        self.y_train = data['record_data']
        self.y_train = self.y_train.squeeze()
        # self.y_time = data['t_record']
        self.Fs = data['Fs']
        self.re_Fs = 16000
        self.Fs = self.Fs.squeeze()
        self.PRF = self.PRF.squeeze()
        
        # Time 
        self.time_gap = 0.5
        self.time_min = 20
        self.time_max = 20 
        self.time = np.arange(self.time_min, self.time_max + self.time_gap, self.time_gap)

        # Moving Average
        self.MovWind = int(0.05*self.PRF)
        
        # # Spectrogram paramater
        # self.stft_wseg = 0.3
        # self.stft_step = 0.01
        # self.nfft_num = 512
        # self.time_num = 100
        # self.flag_DB = True
        # self.crop_ratio = 0.5
        # self.resize2_dim = [3 * self.time_num, 256]
        
        # # Padding parameter
        # self.radar_time_max = self.time_max * self.time_num
        # self.audio_time_max = self.time_max * self.re_Fs
                
    def moving_average(self, sig) :
        win = signal.windows.hann(self.MovWind)
        filtered = signal.convolve(sig, win, mode='same') / sum(win)
    
        return filtered

    # def form_spectrogram(self, signal_temp):
    #     win_size = int(self.PRF*self.stft_wseg)
    #     step_size = int(self.PRF*self.stft_step)
    #     over_size = int(win_size - step_size)
    #     nfft_stft = self.nfft_num
    #     # form spectrogram
    #     f, t, Sxx = signal.spectrogram(signal_temp, self.PRF, window='hamming', nperseg=win_size, noverlap=over_size, nfft=nfft_stft, 
    #                                     return_onesided=False, mode='magnitude')
    #     Sxx = fftshift(Sxx, axes=0)
    #     f = fftshift(f)
    #     # log scale
    #     if self.flag_DB:
    #         Sxx = np.log10(Sxx)
    #         Sxx = 10* Sxx

    #     # spectrum cut
    #     cut_len = int((self.crop_ratio/2)*Sxx.shape[0])
    #     Sxx = Sxx[cut_len:-cut_len+1,:]
    #     f= f[cut_len:-cut_len+1]
    #     # Reject outliers
    #     Sxx_sort = np.sort(Sxx.flatten())
    #     dat_min = Sxx_sort[int(len(Sxx_sort)*0.005)]
    #     dat_max = Sxx_sort[int(len(Sxx_sort)*0.995)]
    #     Sxx[np.where(Sxx<=dat_min)] = dat_min
    #     Sxx[np.where(Sxx>=dat_max)] = dat_max
    #     # 2D resize
    #     if self.resize2_dim[0]==0:
    #         Sxx_intr = Sxx
    #     else:
    #         Sxx_intr = cv2.resize(Sxx, (self.resize2_dim[0], self.resize2_dim[1]), interpolation=cv2.INTER_LINEAR)
            

    #     # plot
    #     plt.pcolormesh(t, f, Sxx, shading='gouraud')
    #     plt.ylabel('Frequency [Hz]')
    #     plt.xlabel('Time [sec]')
    #     plt.show()

    #     return Sxx_intr

    def down_sample(self, y):
        # y, sr = librosa.load('sample-audio.wav', sr=resample_sr)
        resample = librosa.resample(y, self.Fs, self.re_Fs)
        # print("original wav sr: {}, original wav shape: {}, resample wav sr: {}, resmaple shape: {}".format(self.Fs, y.shape, self.re_Fs, resample.shape))
        
        return resample
    
    def __len__(self):
        return len(self.X_train)
        
    def __getitem__(self,idx):       
        X, Y = self.X_train[idx], self.y_train[idx]
        X = X.squeeze()
        Y = Y.squeeze()
        
        X_time_main = np.arange(0, np.size(X)/self.PRF, 1/self.PRF) 
        Y_time_main = np.arange(0, np.size(Y)/self.Fs, 1/self.Fs)  
        # self.X_time[0,idx], self.y_time[0,idx]
        # X_time_main, Y_time_main = self.X_time, self.y_time
        # X_time_main, Y_time_main = X_time_main.squeeze(), Y_time_main.squeeze()

        t_gap = choice(self.time)
        tmp_ind =  np.argmin(abs(X_time_main - t_gap - 0.1))
        tmp_ind = int( len(X_time_main) - tmp_ind)
        tmp_X = X_time_main[:tmp_ind]
        t_sel = choice(tmp_X)
        
        X_st_ind =  np.argmin(abs(X_time_main - t_sel))
        X_end_ind =  int(self.PRF * t_gap) # np.argmin(abs(X_time_main - (t_sel+t_gap) ))

        X = X[X_st_ind:X_st_ind + X_end_ind]
        # X_time = X_time_main[X_st_ind:X_st_ind + X_end_ind]
        
        Y_st_ind =  np.argmin(abs(Y_time_main - t_sel))
        Y_end_ind = int(self.Fs * t_gap) # np.argmin(abs(Y_time_main - (t_sel+t_gap) ))
        
        Y = Y[Y_st_ind: Y_st_ind + Y_end_ind]
        # Y_time = Y_time_main[Y_st_ind:Y_st_ind + Y_end_ind]       
        
        # Audio Downsampling
        if self.Fs >= self.re_Fs:
            Y = self.down_sample(Y)
            # Y_time_main = np.arange(Y_time_main[0], Y_time_main[-1], 1/self.re_Fs)
            
        # Moving Mean
        if self.Moving:
            filtered = self.moving_average(X)
            X = X - filtered

        # # Spectrogram
        # self.resize2_dim[0] = int(t_gap * self.time_num)
        # X = self.form_spectrogram(X)
    
        # if self.normalize:
        #     X = (X-np.min(X))/(np.max(X)-np.min(X))
        #     Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))
        #     # raw_speech = [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in raw_speech]
            
      
        # X = torch.FloatTensor(X)  
        # X = torch.unsqueeze(X,0)
        # Y = torch.FloatTensor(Y)  
        return X, Y