# In[]
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import jiwer

import numpy as np

from AAA import Radar2VecModel, Radar2Vec, Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from AAA.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config

from Radar2txtCustomDatasets import custom_dataset

# In[]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# In[]
# import visdom
# vis = visdom.Visdom()
# vis.close(env="main")
# # python -m visdom.server -port 8097

# def value_tracker(value_plot, value, num):
#     '''num, loss_value, are Tensor'''
#     vis.line(X=num,
#              Y=value,
#              win = value_plot,
#              update='append'
#              )

# loss_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
# acc_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='Accuracy', legend=['Acc'], showlegend=True))
# wer_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='WER', legend=['WER'], showlegend=True))

# In[]
def ConfigLoad():
    model_args = ()
    kwargs = {}
    config = kwargs.pop("config", None)
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)

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
    return config

# In[]
def acc_check( criterion, train_set, test_set, train_total_batch, test_total_batch, epoch, save=1):
    model.eval()
    model2.eval()

    class_size = 32

    correct_train = np.zeros(class_size)
    cnt_train = np.zeros(class_size)
    total_train = 0
    avg_loss_train = 0
    wer_train = 0
    
    correct_test = np.zeros(class_size)
    cnt_test = np.zeros(class_size)
    total_test = 0
    avg_loss_test = 0
    wer_test = 0

    with torch.no_grad():
        for data in train_set:
            # get the inputs
            inputs, labels, logits = data["X_train"], data["Y_feat"], data["Y_logits"] #Y_hid Y_feat
            inputs = inputs.to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            logits = logits.type(torch.FloatTensor).to(device)

            # forward + backward + optimize
            outputs = model(inputs).extract_features
            # outputs = model(inputs).hidden_states
            outputs = outputs.type(torch.FloatTensor).to(device)
            
            # Predict sentence
            predicted = model2(outputs).logits
            predicted_gt = torch.argmax(logits, dim=-1)
            # ind = np.where(predicted_gt.to('cpu') !=0)
            loss = criterion(outputs, labels)*1 + criterion(predicted, logits) * weigth_loss
            # loss = criterion(outputs, labels)*1 
            predicted = torch.argmax(predicted, dim=-1)

            for i in range(class_size):
                ind = np.where(predicted_gt.to('cpu') == i) 
                len = np.size(ind,1)
                if len != 0:
                    correct_train[i] += (predicted[ind] == i).sum().item() 
                    cnt_train[i] += len

            transcription_gt = tokenizer.batch_decode(predicted_gt)
            hypothesis  = tokenizer.batch_decode(predicted)
            wer_train += jiwer.wer(transcription_gt, hypothesis)
            total_train += labels.size(0)
            avg_loss_train += loss.item()

            # print(' ')
            # print(transcription_gt[0])
            # print(hypothesis[0])
            # print(' ')

            del outputs, loss, labels, inputs, logits
            torch.cuda.empty_cache()

        for data in test_set:
            # get the inputs
            inputs, labels, logits = data["X_train"], data["Y_feat"], data["Y_logits"] # Y_hid Y_feat
            inputs = inputs.to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            logits = logits.type(torch.FloatTensor).to(device)

            # forward + backward + optimize
            outputs = model(inputs).extract_features
            # outputs = model(inputs).hidden_states
            outputs = outputs.type(torch.FloatTensor).to(device)
            
            # Predict sentence
            predicted = model2(outputs).logits
            predicted_gt = torch.argmax(logits, dim=-1)
            # ind = np.where(predicted_gt.to('cpu') !=0)
            loss = criterion(outputs, labels)*1 + criterion(predicted, logits) * weigth_loss
            # loss = criterion(outputs, labels)*1 
            predicted = torch.argmax(predicted, dim=-1)

            for i in range(class_size):
                ind = np.where(predicted_gt.to('cpu') == i) 
                len = np.size(ind,1)
                if len != 0:
                    correct_test[i] += (predicted[ind] == i).sum().item() 
                    cnt_test[i] += len

            transcription_gt = tokenizer.batch_decode(predicted_gt)
            hypothesis  = tokenizer.batch_decode(predicted)
            wer_test += jiwer.wer(transcription_gt, hypothesis)
            total_test += labels.size(0)
            avg_loss_test += loss.item()
            
            # print(' ')
            # print(transcription_gt[0])
            # print(hypothesis[0])
            # print(' ')

            del outputs, loss, labels, inputs, logits
            torch.cuda.empty_cache()

    if np.isnan(avg_loss_train) or np.isnan(avg_loss_test) :
        print('nan error, epcoh {:4d}'.format(epoch+1))
        avg_loss_train = 100
        avg_loss_test = 100
    
    ind_train = np.where(cnt_train != 0)
    correct_train = correct_train[ind_train]
    cnt_train = cnt_train[ind_train]
    correct_train = correct_train/cnt_train
    correct_train = ( np.sum(correct_train) / np.size(ind_train,1) ) * 100

    ind_test = np.where(cnt_test != 0)
    correct_test = correct_test[ind_test]
    cnt_test = cnt_test[ind_test]
    correct_test = correct_test/cnt_test
    correct_test = ( np.sum(correct_test) / np.size(ind_test,1) ) * 100

    avg_loss_train, avg_loss_test = avg_loss_train / train_total_batch, avg_loss_test /  test_total_batch
    wer_test = ( wer_test / test_total_batch  ) * 100
    wer_train = ( wer_train /  train_total_batch) * 100
    print('#################### Epoch {:4d} ####################'.format(epoch)) 
    print('Train => Cost: {:.4f}, Accuarcy: {:.3f}, WER: {:.3f}'.format( avg_loss_train, correct_train, wer_train ) )
    print('Test  => Cost: {:.4f}, Accuarcy: {:.3f}, WER: {:.3f}'.format(avg_loss_test, correct_test, wer_test ) )
    print('####################################################')
    print()
    
    if save:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': avg_loss_train,
            'loss_test': avg_loss_test,
            }, "/NAS/LGbuilding/KKB/code/wav2vec2-huggingface-demo-main copy/model/model_LossOne_epoch_{}_acc_{}_wer_{}.pth".format(epoch, int(correct_test),int(wer_test)))
    return correct_test, avg_loss_test, wer_test

# In[]
# parameters
learning_rata = 1e-4
training_epochs = 1000
bat_size = 20
test_bat_size = bat_size * 2
drop_prob = 0.1
weigth_loss = 0.01
# In[]
# Data load
data_path = '/workspace/datasets/'
data_name_train = 'wav2txt_data_test'
data_name_test = 'wav2txt_data_test'
data_full_train = data_path + data_name_train
data_full_test = data_path + data_name_test

#To load from pickle file
output_train = []
with open(data_full_train + ".pkl", 'rb') as fr:
    try:
        while True:
            output_train += pickle.load(fr)
    except EOFError:
        pass

output_test = []
with open(data_full_test + ".pkl", 'rb') as fr:
    try:
        while True:
            output_test += pickle.load(fr)
    except EOFError:
        pass

trainset = custom_dataset(output_train)
testset = custom_dataset(output_test)

train_loader = DataLoader(trainset, batch_size = bat_size, shuffle = True)
test_loader = DataLoader(testset, batch_size = bat_size, shuffle = False)

train_loader2 = DataLoader(trainset, batch_size = test_bat_size, shuffle = False)
test_loader2 = DataLoader(testset, batch_size = test_bat_size, shuffle = False)

# In[]
config = ConfigLoad()
# model = Davenet();
model = Radar2VecModel(config).to(device)
# model.load_state_dict(checkpoint)
print(model)
model2 = Radar2Vec.to(device)
print(model2)

model2 = Radar2Vec.from_pretrained("facebook/wav2vec2-base-960h").to(device)
print(model2)

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

# In[]
criterion = nn.L1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rata)

# In[]
# epoch_num = 98
# acc_num = 5
# wer_num = 133  
# checkpoint  = torch.load( f"/NAS/LGbuilding/KKB/code/wav2vec2-huggingface-demo-main copy/model/model_LossOne_epoch_{epoch_num}_acc_{acc_num}_wer_{wer_num}.pth") 
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss_train']
# In[]
# 모델의 state_dict 출력
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# # 옵티마이저의 state_dict 출력
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])
summary(model, input_size=(bat_size, 3, 256, 800))
# summary(model2, input_size=(bat_size, 499, 512))

# In[]
train_total_batch = len(train_loader2)
test_total_batch = len(test_loader2)
print(train_total_batch)
print(test_total_batch)

for epoch in range(training_epochs):  # loop over the dataset multiple times
    # lr_sche.step()

    #Check Accuracy
    # if (epoch + 1) % 2 == 0:
    #     save_num = 0
    # else:
    #     save_num = 1

    model.train() # set the model to train mode (dropout=True)
    model2.eval()
    for data in train_loader:
        # get the inputs
        inputs, labels, logits = data["X_train"], data["Y_feat"], data["Y_logits"] # Y_hid Y_feat
        inputs = inputs.to(device)
        labels = labels.type(torch.FloatTensor).to(device)
        logits = logits.type(torch.FloatTensor).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs).extract_features
        outputs = outputs.type(torch.FloatTensor).to(device)
        
        # Predict sentence
        # predicted = model2(outputs).logits
        # loss = criterion(outputs, labels) + criterion(predicted, logits) * weigth_loss 
        loss =  criterion(outputs, labels) 

        loss.backward()
        optimizer.step()

        del outputs, loss, labels, inputs
    # if (epoch + 1) % 2 == 0:
    torch.cuda.empty_cache()
    acc, losses, wer = acc_check( criterion, train_loader2, test_loader2, train_total_batch, test_total_batch, epoch, save=1)
    torch.cuda.empty_cache()
    # value_tracker(loss_plt, torch.Tensor([losses]), torch.Tensor([epoch]))
    # value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))
    # value_tracker(wer_plt, torch.Tensor([wer]), torch.Tensor([epoch]))

    
    



# %%
