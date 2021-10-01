#%%
# # -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:59:34 2021

@author: kibong
"""

# import numpy as np

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random 
# from random import choice, choices
import cv2

import scipy.io as sio # MATLAB data load 용
# import mat73
import h5py

from scipy import signal
from scipy.fft import fftshift
# from scipy.signal import decimate
import matplotlib.pyplot as plt
# import pandas as pd

import librosa

from AAA import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Tokenizer

# In[]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# In[]
# import visdom
# vis = visdom.Visdom()
# vis.close(env="main")
# python -m visdom.server

# In[]
class form_spectrogram():
    def __init__(self):
        # self.data = np.array(data)
        self.PRF = 1000
        self.time_num = 100
        
        # self.stft_wseg = 1
        self.stft_step = 0.01
        self.nfft_num = 1024
        self.flag_DB = True
        self.crop_ratio = 1
        self.resize2_dim = [1, 256]

    def get(self, data, stft_wseg):
        for i in np.nonzero(data==0):
            data[i] = (random.random() + random.random()*1j)* 1e-5
        win_size = int(self.PRF*stft_wseg)
        step_size = int(self.PRF*self.stft_step)
        over_size = int(win_size - step_size)
        nfft_stft = self.nfft_num
        data = np.array(data)
        data_ln = np.size(data)
        time_ln = int(data_ln / self.PRF * self.time_num)
        self.resize2_dim[0] = time_ln
        
        # form spectrogram
        f, t, Sxx = signal.spectrogram(data, self.PRF, window='hamming', nperseg=win_size, noverlap=over_size, nfft=nfft_stft, 
                                        return_onesided=False, mode='psd')
        Sxx = fftshift(Sxx, axes=0)
        f = fftshift(f)
        if np.sum(np.isnan(Sxx)) > 0:
            print(' Sxx zero error')
        # log scale
        if self.flag_DB:
            Sxx[np.where(Sxx==0)] = random.random() * 1e-5
            Sxx = np.log10(Sxx)
            Sxx = 10* Sxx

        # spectrum cut
        cut_len = int((self.crop_ratio)*Sxx.shape[0])
        cut_ind = np.arange(0,cut_len) + int((Sxx.shape[0] - cut_len)/2 )
        Sxx = Sxx[cut_ind,:]
        f= f[cut_ind]

        # Reject outliers
        Sxx_sort = np.sort(Sxx.flatten())
        dat_min = Sxx_sort[int(len(Sxx_sort)*0.4)]
        dat_max = Sxx_sort[int(len(Sxx_sort)*0.99)]
        Sxx = np.where(Sxx <= dat_min, dat_min, Sxx)
        Sxx = np.where(Sxx >= dat_max, dat_max, Sxx)
        
        # 2D resize
        if self.resize2_dim[0]==0:
            Sxx_intr = Sxx
        else:
            Sxx_intr = cv2.resize(Sxx, (self.resize2_dim[0], self.resize2_dim[1]), interpolation=cv2.INTER_LINEAR)
            
        # Sxx_intr = cv2.flip(Sxx_intr,0) 
        # Sxx_intr = np.reshape(Sxx_intr,(2,int(self.resize2_dim[1]/2),-1))
        # Sxx_intr[0] = cv2.flip(Sxx_intr[0],0)

        # for i in range(2):
            # print(i)
            # Sxx_intr = (Sxx_intr - np.mean(Sxx_intr)) / np.sqrt(np.var(Sxx_intr) + 1e-5) /
            # Sxx_intr[i] = (Sxx_intr[i] - np.min(Sxx_intr[i])) / (np.max(Sxx_intr[i]) - np.min(Sxx_intr[i]))

        Sxx_intr = (Sxx_intr - np.min(Sxx_intr)) / (np.max(Sxx_intr) - np.min(Sxx_intr))
                    
        # # plot
        # plt.pcolormesh(t, f, Sxx, shading='gouraud')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()
        # vis.image(Sxx_intr)
        del Sxx, Sxx_sort, dat_min, dat_max
        return Sxx_intr

def get_data(data, spectrogram, model, tokenizer):
    t_gap = 8
    t_step = 3
    re_Fs = 16000
    class_num = 32

    Fs = np.array(data['Fs'], dtype='f8').squeeze()
    PRF = np.array(file['PRF'], dtype='f8').squeeze()
    t_radar =np.array(file['t_radar'], dtype='f8')

    radar_ind_len = int(PRF*t_gap)
    wav_ind_len = int(re_Fs*t_gap)

    radar_ind_step = int(PRF*t_step)
    wave_ind_step = int(re_Fs*t_step)

    radar_ind = np.arange(radar_ind_len) 
    wave_ind = np.arange(wav_ind_len) 

    # data_set = []
    

    with torch.no_grad():
        for j, t_max in enumerate(t_radar):
            t_max = t_radar[j]
            itr = int( (t_max-t_gap)/t_step) 
            radar_data = data['radar_data'][j][0]
            raedar_data2 = np.array( data[radar_data]['real'] + 1j*data[radar_data]['imag'], dtype='c8' )
            radar_data2 = raedar_data2.squeeze()
            wave_data = data['record_data'][j][0]
            wave_data2 = np.array(data[wave_data], dtype='f8')
            wave_data2 = wave_data2.squeeze()
            wave_data2 = librosa.resample(wave_data2, Fs, re_Fs)
            data_set2 = []
            for i in range(itr):
                tmp_data = {}
                print( 'data num: {:4d}, data_itr: {:4d}'.format(j, i))
                # get spectrogram
                radar_ind2 = radar_ind + i  * radar_ind_step
                radar_tmp = radar_data2[radar_ind2]
                radar_tmp = tokenizer(radar_tmp, return_tensors="pt", padding="longest").input_values  # Batch size 1
                radar_tmp = radar_tmp.squeeze(0)
                radar_tmp = [spectrogram.get(radar_tmp, wseg) for wseg in stft_wseg]
                radar_tmp = np.array(radar_tmp)
                # radar_tmp = np.reshape(radar_tmp,(6,128,-1))

                tmp_data["X_train"] =  radar_tmp

                # get feature & logits
                wave_ind2 = wave_ind + i * wave_ind_step
                wave_tmp = wave_data2[wave_ind2]
                wave_tmp = tokenizer(wave_tmp, return_tensors="pt", padding="longest").input_values  # Batch size 1
                wave_tmp = wave_tmp.type(torch.FloatTensor).to(device)
                outputs = model(wave_tmp)
                wave_features = outputs.extract_features2.squeeze(0)
                wave_hidden = outputs.hidden_states2.squeeze(0)
                wave_logits = outputs.logits.squeeze(0)

                wave_features = wave_features.to('cpu')
                wave_hidden = wave_hidden.to('cpu')
                wave_logits = wave_logits.to('cpu')

                tmp_data["Y_feat"] = wave_features
                tmp_data["Y_hid"] = wave_hidden
                tmp_data["Y_logits"] = wave_logits

                # data_set.append(tmp_data)
                data_set2.append(tmp_data)
                del tmp_data, radar_tmp, wave_tmp
            pickle.dump(data_set2, a_file)
            del radar_data, radar_data2, wave_data, wave_data2, data_set2

    # return data_set

# In[]
stft_wseg = (0.1, 0.5, 1)

# In[]
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

spectrogram = form_spectrogram()

model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').to(device)

# model2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

# In[]

data_path = '/workspace/datasets/'
data_name_train = 'wav2txt_data_train'
data_name_test = 'wav2txt_data_test'
data_full_train = data_path + data_name_train
data_full_test = data_path + data_name_test

# In[]

# a_file = open(data_full_train + ".pkl", "wb")
# with h5py.File(data_full_train +'.mat', 'r') as file:
#     get_data(file, spectrogram, model, tokenizer)
# a_file.close()

a_file = open(data_full_test + ".pkl", "wb")
with h5py.File(data_full_test +'.mat', 'r') as file:
    get_data(file, spectrogram, model, tokenizer)
a_file.close()

