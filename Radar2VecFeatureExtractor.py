# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:45:55 2021

@author: kibong
"""
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import scipy.io as sio # MATLAB data load 용

from torch.utils.data import Dataset, TensorDataset, DataLoader

import itertools # for plot confusion matrix
from sklearn.metrics import confusion_matrix # for making confusion matrix
from sklearn.metrics import classification_report
from AAA import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model


# In[]
# import visdom

# vis = visdom.Visdom()
# vis.close(env="main")

# # In[]
# def value_tracker(value_plot, value, num):
#     '''num, loss_value, are Tensor'''
#     vis.line(X=num,
#              Y=value,
#              win = value_plot,
#              update='append'
#              )
    
# In[]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)
    
# In[2]
# parameters
learning_rata = 1e-3
training_epochs = 100
bat_size = 32
drop_prob = 0.4

# In[]
class custom_dataset(Dataset):
    def __init__(self, data,  label, normalize=True):
        self.normalize = normalize
        
        self.data = data
        self.label = label
                
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):       
        X, Y = self.data[idx,:,:,:], self.label[idx]

        if self.normalize:
            X = (X-np.min(X))/(np.max(X)-np.min(X))
        
        if np.max(np.isnan(X)):
            X = np.ones(X.shape)
            print('error')
            
        
        X = torch.FloatTensor(X)  
        return X, Y

# confusion matrix plot
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Description:
    This function is a modified version of the code made in scikit-learn 
    which is available at 
    "https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html"
    Changes from the code are marked with JHChoi, SHJin
    Description by inventor:
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm.T) # modified by JHChoi, SHJin

    plt.imshow(cm.T, interpolation='nearest', cmap=cmap) # modified by JHChoi, SHJin
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.0f' if normalize else 'd' # modified by JHChoi, SHJin
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # modified by JHChoi, SHJin
        plt.text(i, j, format(cm[i, j]*100, fmt), # modified by JHChoi, SHJin
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('True label') # modified by JHChoi, SHJin
    plt.ylabel('Predicted label') # modified by JHChoi, SHJin
    plt.tight_layout() 

# In[]
# import resnet
import torchvision.models.resnet as resnet
# In[]
conv1x1=resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock= resnet.BasicBlock
# In[]
class Radar2VecGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # self.activation = ACT2FN[config.feat_extract_activation]

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        # hidden_states = self.activation(hidden_states)
        return hidden_states
    
    
# In[]
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x.shape =[1, 16, 32,32]
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        #x.shape =[1, 128, 32,32]
        x = self.layer2(x)
        #x.shape =[1, 256, 32,32]
        x = self.layer3(x)
        #x.shape =[1, 512, 16,16]
        x = self.layer4(x)
        # x.shape =[1, 1024, 8,8]
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
# In[]

data_path = 'G:/data/'
data_name = 'ch3_p3_vowel_data'
data_full = data_path + data_name

mat_data = sio.loadmat(data_full)

X_train = mat_data['X_train']
X_test = mat_data['X_test']
y_train = mat_data['Y_train']
y_test = mat_data['Y_test']
y_train = np.squeeze(y_train, 1)
y_test = np.squeeze(y_test, 1)
y_train = y_train-1
y_test = y_test-1

# =============================================================================
# ####################################################
# a = np.where(np.isnan(X_train))[0]
# for i in range(int(len(a)/(3*88*88))):
#     idx = a[i*3*88*88]
#     # print(np.sum(np.isnan(X_train[idx,:,:,:])))
#     X_train[idx,:,:,:] = X_train[0,:,:,:].copy()
#     y_train[idx,:] = y_train[0,:].copy()
#     
# # y_train = np.squeeze(y_train)
# # y_test = np.squeeze(y_test)
# ###################################################    
# =============================================================================

trainset = custom_dataset(X_train, y_train, normalize=False)
testset = custom_dataset(X_test, y_test, normalize=False)

# In[]
train_loader = DataLoader(trainset, batch_size = bat_size, shuffle = True)
test_loader = DataLoader(testset, batch_size = len(y_test), shuffle = False)

# =============================================================================
# for itr, data in enumerate(train_loader):
#     X, Y = data
#     X.shape
#     Y.shape
# =============================================================================

# In[]
r_conv_dim=(512, 512, 512, 512, 512, 512, 512)
r_conv_stride=(5, 2, 2, 2, 2, 2, 2)
r_conv_kernel=(10, 3, 3, 3, 3, 2, 2)



# In[]
def resnet50(pretrained=False):
    model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 6, True).to(device) 
    #1(conv1) + 9(layer1) + 12(layer2) + 18(layer3) + 9(layer4) +1(fc)= ResNet50
    return model
def resnet18(pretrained=False):
    model= ResNet(resnet.BasicBlock, [2, 2, 2, 2], 6, True).to(device) 
    return model

model = resnet18().to(device)

# In[]
print(model)

# In[]
a=torch.Tensor(1,3,88,88).to(device)
out = model(a)
print(' ')
print(' ')
print("output")
print(out)

# In[]




    
    
    