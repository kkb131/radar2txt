# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 23:06:00 2021

@author: kibong
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:49:57 2021

@author: kibong
"""

# from IPython.display import Audio
from scipy.io import wavfile
import numpy as np

import torch
import torchaudio
# import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

import audio_preprocessing as ap
import sounddevice as sd
# import requests
# import tarfile
# import boto3


# In[]

file_name = 'sample-audio.wav'
# Audio(file_name)

data = wavfile.read(file_name)
framerate = data[0]
sounddata = data[1]
time = np.arange(0,len(sounddata))/framerate
print('Sample rate:',framerate,'Hz')
print('Total time:',len(sounddata)/framerate,'s')

# In[]
# import soundfile as sf
# import librosa
import torch
from AAA import Wav2Vec2Processor, Wav2Vec2ForCTC
from AAA import Radar2VecModel

# In[]

tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# In[]

dataset2 = torchaudio.datasets.LIBRISPEECH("G:/data/libri/", 
                                           url = "test-clean",
                                           folder_in_archive = "LibriSpeech",
                                           download=False,)
# In[]

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128
sample_rate = 16000

mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

# In[]

for i in [1, 3, 5]:
    waveform, sample_rate, script, sp_id, book_id, data_num  = dataset2[i]
    
    melspec = mel_spectrogram(waveform)
    ap.plot_spectrogram(
        melspec[0], title=f"MelSpectrogram - Sample: {i}, speak id: {sp_id}", ylabel='mel freq')
    ap.play_audio2(waveform, sample_rate)
    
    input_values = tokenizer(waveform, return_tensors="pt").input_values
    input_values = input_values.squeeze(1)
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    
    print(' ')
    print(i)
    print(transcription)
    print(script)
    
    sd.stop()
    

# In[]
print(transcription)

# In[]

# In[]

