# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:49:57 2021

@author: kibong
"""

from IPython.display import Audio
from scipy.io import wavfile
import numpy as np

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
import soundfile as sf
import librosa
import torch
from AAA import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model

# In[]

tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# In[]

input_audio, _ = librosa.load(file_name, 
                              sr=16000)

# In[]

input_values = tokenizer(input_audio, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]

# In[]
print(transcription)

# In[]

# In[]

