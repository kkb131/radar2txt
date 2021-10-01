# In[]
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
# python -m visdom.server -port 9002

# def value_tracker(value_plot, value, num):
#     '''num, loss_value, are Tensor'''
#     vis.line(X=num,
#              Y=value,
#              win = value_plot,
#              update='append'
#              )

# loss_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
# acc_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='Accuracy', legend=['Acc'], showlegend=True))

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
def acc_check( criterion, train_set, test_set, train_total_batch, test_total_batch, epoch, save=0):
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

    class_size = 32
    matrix_test = np.zeros((class_size, class_size))
    matrix_train = np.zeros((class_size, class_size))

    with torch.no_grad():
        for data in train_set:
            # get the inputs
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

            # print(' ')
            # transcription = tokenizer.batch_decode(predicted_gt)
            # print(transcription[0])
            # transcription = tokenizer.batch_decode(predicted)
            # print(transcription[0])
            # print(' ')
            
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

            # print(' ')
            # transcription = tokenizer.batch_decode(predicted_gt)
            # print(transcription)
            # transcription = tokenizer.batch_decode(predicted)
            # print(transcription)
            # print(' ')
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
    print('#################### Epoch {:4d} ###################'.format(epoch + 1)) 
    print('Train => Cost: {:.4f}, Accuarcy: {:.3f}, WER: {:.3f}'.format( avg_loss_train, correct_train, wer_train ) )
    print('Test  => Cost: {:.4f}, Accuarcy: {:.3f}, WER: {:.3f}'.format(avg_loss_test, correct_test, wer_test ) )
    print('####################################################')
    print()
    

# In[]
# parameters
training_epochs = 1
bat_size = 50
# epoch_num = 193
# acc_num = 66
weigth_loss = 0.005

# In[]
# Data load
data_path = '/workspace/datasets/'
data_name_train = 'wav2txt_data_train'
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

train_loader = DataLoader(trainset, batch_size = bat_size, shuffle = False)
test_loader = DataLoader(testset, batch_size = bat_size, shuffle = False)

# In[]
config = ConfigLoad()
# model = Davenet();
model = Radar2VecModel(config).to(device)
epoch_num = 423
acc_num = 5
wer_num = 140   
checkpoint  = torch.load( f"/NAS/LGbuilding/KKB/code/wav2vec2-huggingface-demo-main copy/model/model_Loss_one_epoch_{epoch_num}_acc_{acc_num}_wer_{wer_num}.pth") 
model.load_state_dict(checkpoint)
print(model)

model2 = Radar2Vec.from_pretrained("facebook/wav2vec2-base-960h").to(device)

model3 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

# In[]
criterion = nn.MSELoss().to(device)

# In[]
train_total_batch = len(train_loader)
test_total_batch = len(test_loader)
print(train_total_batch)
print(test_total_batch)

# lr_sche.step()
#Check Accuracy
save_num = 0
acc, losses = acc_check( criterion, train_loader, test_loader, train_total_batch, test_total_batch, training_epochs, save=save_num)
# value_tracker(loss_plt, torch.Tensor([losses]), torch.Tensor([epoch]))
# value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))

