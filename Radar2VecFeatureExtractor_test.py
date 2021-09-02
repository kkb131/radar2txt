# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:59:34 2021

@author: kibong
"""

# import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import random
# from random import choice, choices
import cv2

import scipy.io as sio # MATLAB data load 용
from scipy import signal
from scipy.fft import fftshift
# from scipy.signal import decimate
import matplotlib.pyplot as plt
# import pandas as pd

# import librosa

from AAA import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Tokenizer
from AAA import Radar2VecModel
from AAA.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config

from custom_dataset_lip import custom_dataset
# In[]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)
    
# In[2]
# parameters
learning_rata = 1e-3
training_epochs = 1
bat_size = 2
drop_prob = 0.4

# In[]
model_args = ()
kwargs = {}
config = kwargs.pop("config", None)
state_dict = kwargs.pop("state_dict", None)
cache_dir = kwargs.pop("cache_dir", None)
from_tf = kwargs.pop("from_tf", False)
from_flax = kwargs.pop("from_flax", False)
ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
force_download = kwargs.pop("force_download", False)
resume_download = kwargs.pop("resume_download", False)
proxies = kwargs.pop("proxies", None)
output_loading_info = kwargs.pop("output_loading_info", False)
local_files_only = kwargs.pop("local_files_only", False)
use_auth_token = kwargs.pop("use_auth_token", None)
revision = kwargs.pop("revision", None)
mirror = kwargs.pop("mirror", None)
from_pipeline = kwargs.pop("_from_pipeline", None)
from_auto_class = kwargs.pop("_from_auto", False)
_fast_init = kwargs.pop("_fast_init", True)
torch_dtype = kwargs.pop("torch_dtype", None)

config_path = 'facebook/wav2vec2-base-960h'
config, model_kwargs = Wav2Vec2Config.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )

# In[]
# a simple custom collate function, just to show the idea
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]
    
class form_spectrogram():
    def __init__(self):
        # self.data = np.array(data)
        self.PRF = 200
        self.time_num = 100
        
        self.stft_wseg = 0.3
        self.stft_step = 0.01
        self.nfft_num = 512
        self.flag_DB = True
        self.crop_ratio = 0.5
        self.resize2_dim = [1, 256]

    def get(self, data):
        for i in np.nonzero(data==0):
            data[i] = (random.random() + random.random()*1j)* 1e-5
        win_size = int(self.PRF*self.stft_wseg)
        step_size = int(self.PRF*self.stft_step)
        over_size = int(win_size - step_size)
        nfft_stft = self.nfft_num
        data = np.array(data)
        data_ln = np.size(data)
        time_ln = int(data_ln / self.PRF * self.time_num)
        self.resize2_dim[0] = time_ln
        
        # form spectrogram
        f, t, Sxx = signal.spectrogram(data, self.PRF, window='hamming', nperseg=win_size, noverlap=over_size, nfft=nfft_stft, 
                                        return_onesided=False, mode='magnitude')
        Sxx = fftshift(Sxx, axes=0)
        f = fftshift(f)
        # log scale
        if self.flag_DB:
            Sxx = np.log10(Sxx)
            Sxx = 10* Sxx

        # spectrum cut
        cut_len = int((self.crop_ratio/2)*Sxx.shape[0])
        Sxx = Sxx[cut_len:-cut_len+1,:]
        f= f[cut_len:-cut_len+1]
        # Reject outliers
        Sxx_sort = np.sort(Sxx.flatten())
        dat_min = Sxx_sort[int(len(Sxx_sort)*0.005)]
        dat_max = Sxx_sort[int(len(Sxx_sort)*0.995)]
        Sxx[np.where(Sxx<=dat_min)] = dat_min
        Sxx[np.where(Sxx>=dat_max)] = dat_max
        # 2D resize
        if self.resize2_dim[0]==0:
            Sxx_intr = Sxx
        else:
            Sxx_intr = cv2.resize(Sxx, (self.resize2_dim[0], self.resize2_dim[1]), interpolation=cv2.INTER_LINEAR)
            
        Sxx_intr = (Sxx_intr - np.mean(Sxx_intr)) / np.sqrt(np.var(Sxx_intr) + 1e-5) 
                    
        # # plot
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        return Sxx_intr

# In[]
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

spectrogram = form_spectrogram()

model1 = Radar2VecModel(config).to(device)

model2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device)

model3 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# In[]
print(model1)
# In[]
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model1.parameters(), lr=learning_rata)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay=5e-4)
# lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# In[]

# =============================================================================
# # 모델의 state_dict 출력
# print("Model's state_dict:")
# for param_tensor in model1.state_dict():
#     print(param_tensor, "\t", model1.state_dict()[param_tensor].size())
# 
# # 옵티마이저의 state_dict 출력
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])
# =============================================================================
    
    
    
# In[]
# =============================================================================
# rate1 = 100
# rate2 = 16000
# time = 5.7
# t1_len = round( rate1 * time )
# t2_len = round( rate2 * time )
# 
# a=torch.rand(bat_size*2,1,256,t1_len).to(device)
# out1 = model1(a)
# print(out1.shape)
# 
# b=torch.rand(bat_size*2,t2_len).to(device)
# out2 = model2(b).extract_features
# print(out2.shape)
# =============================================================================

# In[]

data_path = 'G:/data/'
data_name = 'rainbow_4'
data_full = data_path + data_name

mat_data = sio.loadmat(data_full)

X_time = mat_data['t_radar']
X_time = X_time.squeeze()
num_class = np.size(X_time)

class_weigth_all = np.zeros(num_class)
for i, data in enumerate(X_time):
    class_weigth_all[i] = np.max(data)
    
class_weigth_all = class_weigth_all / sum(class_weigth_all)
# class_weigth_all = [0, 1, 0, 0, 0, 0, 0, 0]
# class_weigth_all = torch.FloatTensor(class_weigth_all)
# In[]
trainset = custom_dataset(mat_data, normalize=True, Moving=True, Padding=True)
# testset = custom_dataset(X_test, y_test, normalize=False)

weighted_sampler = WeightedRandomSampler(
    weights=class_weigth_all,
    num_samples=len(class_weigth_all),
    replacement=True
)

train_loader = DataLoader(trainset, batch_size = bat_size, shuffle = False, collate_fn=my_collate, sampler=weighted_sampler)
# test_loader = DataLoader(testset, batch_size = len(y_test), shuffle = False)

# In[]
# train_total_batch = len(train_loader)
# test_total_batch = len(test_loader)
# print(train_total_batch)
# print(test_total_batch)

# epochs = 150

for epoch in range(training_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    # lr_sche.step()
    # model.train() # set the model to train mode (dropout=True)
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels_input = data
        
        # Tokenize
        inputs = tokenizer(inputs, return_tensors="pt", padding="longest").input_values  # Batch size 1
        labels_input = tokenizer(labels_input, return_tensors="pt", padding="longest").input_values  # Batch size 1
        
        # Get spectrogram
        inputs = [spectrogram.get(item) for item in inputs]
        inputs = torch.FloatTensor(inputs).to(device)
        inputs = torch.unsqueeze(inputs, 0)
        inputs = inputs.transpose(0,1)
        labels_input = labels_input.to(device)
        
        # Get encoding features from wav2vec2model
        with torch.no_grad():
            labels = model2(labels_input).extract_features
            labels = labels.type(torch.LongTensor).to(device)
            labels = labels.transpose(1,2)
            # # retrieve logits
            # logits = model3(labels_input).logits
            
            # # take argmax and decode
            # predicted_ids = torch.argmax(logits, dim=-1)
            # transcription = tokenizer.batch_decode(predicted_ids)
            # print(transcription)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model1(inputs)
        outputs = outputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        loss = torch.sqrt(criterion(outputs, labels))
        print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
# =============================================================================
#         if i % 9 == 8:    # print every 30 mini-batches
#             value_tracker(loss_plt, torch.Tensor([running_loss/10]), torch.Tensor([i + epoch*len(train_loader) ]))
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 9))
#             running_loss = 0.0
# =============================================================================
    
    #Check Accuracy
    # model.eval()
    # acc, losses = acc_check(model, criterion, train_loader, test_loader, train_total_batch, test_total_batch, epoch, save=1)
    # value_tracker(loss_plt, torch.Tensor([losses]), torch.Tensor([epoch]))
    # value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))
    
    
    