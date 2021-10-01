#%%
# # -*- coding: utf-8 -*-
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
from AAA import Radar2VecModel, Radar2Vec
from AAA.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config

from custom_dataset_lip import custom_dataset
# In[]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.manual_seed(777)
# if device =='cuda':
#     torch.cuda.manual_seed_all(777)
    
# In[2]
# parameters
learning_rata = 1e-3
training_epochs = 20000
bat_size = 8
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
import visdom
vis = visdom.Visdom()
vis.close(env="main")
# python -m visdom.server

def value_tracker(value_plot, value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
             Y=value,
             win = value_plot,
             update='append'
             )

loss_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
acc_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='Accuracy', legend=['Acc'], showlegend=True))

# In[]
# a simple custom collate function, just to show the idea
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]
    

class form_spectrogram():
    def __init__(self):
        # self.data = np.array(data)
        self.PRF = 1000
        self.time_num = 100
        
        self.stft_wseg = 0.1
        self.stft_step = 0.01
        self.nfft_num = 512
        self.flag_DB = False
        self.crop_ratio = 1
        self.resize2_dim = [1, 512]

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
        # plt.pcolormesh(t, f, Sxx, shading='gouraud')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()
        # vis.image(Sxx_intr)

        return Sxx_intr

# In[]
def acc_check(model1, model2, model3, model4, criterion, train_set, test_set, train_total_batch, test_total_batch, epoch, save=1):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    correct_train = 0
    total_train = 0
    avg_loss_train = 0
    
    correct_test = 0
    total_test = 0
    avg_loss_test = 0

    with torch.no_grad():
        for data in train_set:
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
            labels_logits = model2(labels_input).extract_features2
            labels_logits = labels_logits.type(torch.FloatTensor).to(device)

            # forward + backward + optimize
            outputs = model1(inputs).extract_features
            outputs = outputs.type(torch.FloatTensor).to(device)
            # outputs = model3(outputs).logits
            loss = torch.sqrt(criterion(outputs, labels_logits))

            # predicted = torch.argmax(outputs, dim=-1)
            # labels = torch.argmax(labels_logits, dim=-1)

            # transcription = tokenizer.batch_decode(labels)
            # print(transcription)

            total_train += labels_logits.size(0)
            # correct_train += (predicted == labels).sum().item()
            avg_loss_train += loss.item()

        for data in test_set:
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
            labels_logits = model2(labels_input).extract_features2
            labels_logits = labels_logits.type(torch.FloatTensor).to(device)
            
            # forward + backward + optimize
            outputs = model1(inputs).extract_features
            outputs = outputs.type(torch.FloatTensor).to(device)
            # outputs = model3(outputs).logits
            loss = torch.sqrt(criterion(outputs, labels_logits))

            # predicted = torch.argmax(outputs, dim=-1)
            # labels = torch.argmax(labels_logits, dim=-1)

            # transcription = tokenizer.batch_decode(predicted)
            # print(transcription)

            total_test += labels_logits.size(0)
            # correct_test += (predicted == labels).sum().item()
            avg_loss_test += loss.item()
       
    if np.isnan(avg_loss_train) or np.isnan(avg_loss_test) :
        print('nan error, epcoh {:4d}'.format(epoch+1))
        avg_loss_train = 100
        avg_loss_test = 100
            
    avg_loss_train, avg_loss_test = avg_loss_train / train_total_batch, avg_loss_test /  test_total_batch
    acc_test = 0# ( correct_test / total_test / 499) * 100
    acc_train = 0#( correct_train / total_train / 499) * 100
    print('############### Epoch {:4d} ##############'.format(epoch + 1)) 
    print('Train => Cost: {:.4f}, Accuarcy: {:.3f}'.format( avg_loss_train, acc_train ) )
    print('Test  => Cost: {:.4f}, Accuarcy: {:.3f}'.format(avg_loss_test, acc_test ) )
    print('##########################################')
    print()
    
    if save:
        torch.save(model1.state_dict(), "/media/code/wav2vec2-huggingface-demo-main/model/model_epoch_{}_loss_{}.pth".format(epoch, int(avg_loss_test)))
    return acc_test, avg_loss_test

# In[]
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

spectrogram = form_spectrogram()

model1 = Radar2VecModel(config).to(device)
# checkpoint  = torch.load( "/media/code/wav2vec2-huggingface-demo-main/model/model_epoch_9_acc_2.pth")
# model1.load_state_dict(checkpoint)
# model1.train()

model2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device)

model3 = Radar2Vec.from_pretrained("facebook/wav2vec2-base-960h").to(device)

model4 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
# In[]
print(model1)
# In[]
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model1.parameters(), lr=learning_rata)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay=5e-4)
# lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# In[]
# 모델의 state_dict 출력
# print("Model's state_dict:")
# for param_tensor in model1.state_dict():
#     print(param_tensor, "\t", model1.state_dict()[param_tensor].size())

# # 옵티마이저의 state_dict 출력
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

# In[]

data_path = '/media/datasets/'
data_name_train = 'wav2txt_data_train'
data_name_test = 'wav2txt_data_test'
data_full_train = data_path + data_name_train
data_full_test = data_path + data_name_test

train_data = sio.loadmat(data_full_train)
test_data = sio.loadmat(data_full_test)

X_time = train_data['t_radar']
X_time = X_time.squeeze()
num_class = np.size(X_time)
    
class_weigth_all = X_time / sum(X_time)
# class_weigth_all = [0, 1, 0, 0, 0, 0, 0, 0]
# class_weigth_all = torch.FloatTensor(class_weigth_all)
# In[]
trainset = custom_dataset(train_data, Moving=False)
testset = custom_dataset(test_data, Moving=False)

weighted_sampler = WeightedRandomSampler(
    weights=class_weigth_all,
    num_samples=len(class_weigth_all),
    replacement=True
)

train_loader = DataLoader(trainset, batch_size = bat_size, shuffle = False, collate_fn=my_collate, sampler=weighted_sampler)
test_loader = DataLoader(testset, batch_size = np.size(test_data['t_radar'],1), shuffle = False, collate_fn=my_collate)

# In[]
train_total_batch = len(train_loader)
test_total_batch = len(test_loader)
print(train_total_batch)
print(test_total_batch)

# epochs = 150
model1.train()
model2.eval()
model3.eval()
model4.eval()
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
            labels = model2(labels_input).extract_features2
            labels = labels.type(torch.FloatTensor).to(device)
            if np.size(labels_input, axis=1) != 160000:
                print('label input error')
            elif np.size(labels, axis=1) != 499:
                print('lables error')
            # # retrieve logits
            # logits = model4(labels_input).logits
            
            # # take argmax and decode
            # predicted_ids = torch.argmax(logits, dim=-1)
            # transcription = tokenizer.batch_decode(predicted_ids)
            # print(transcription)

            # import sounddevice as sd
            # sd.play(labels_input[1], 16000)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model1(inputs).extract_features
        outputs = outputs.type(torch.FloatTensor).to(device)
        loss = torch.sqrt(criterion(outputs, labels))
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        
    if epoch % 10 == 9:    # print every 30 mini-batches
        # value_tracker(loss_plt, torch.Tensor([running_loss/10]), torch.Tensor([ epoch + 1 ]))
        # print('[%d] loss: %.3f' %
        #         (epoch + 1, running_loss / 10))
        # running_loss = 0.0

        #Check Accuracy
        acc, losses = acc_check(model1, model2, model3, model4, criterion, train_loader, test_loader, train_total_batch, test_total_batch, epoch, save=1)
        value_tracker(loss_plt, torch.Tensor([losses]), torch.Tensor([epoch]))
        value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))
        model1.train()

    
    #Check Accuracy
    # model.eval()
    # acc, losses = acc_check(model1, model3, model4, criterion, train_loader, test_loader, train_total_batch, test_total_batch, epoch, save=1)
    # value_tracker(loss_plt, torch.Tensor([losses]), torch.Tensor([epoch]))
    # value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))
    
    #
    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model1.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss,
    #         }, "/media/code/wav2vec2-huggingface-demo-main/model/model_epoch_{}_loss_{}.pth".format(epoch, int(avg_loss_test))
    