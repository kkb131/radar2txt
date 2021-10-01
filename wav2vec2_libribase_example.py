# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:26:32 2021

@author: kibong
"""

# In[]

from AAA import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import sounddevice as sd
import torch
import numpy as np

# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# define function to read in sound file
def map_to_array(batch):
    speech, fs = sf.read(batch["file"])
    batch["speech"] = speech
    batch["fs"] = fs
    return batch

# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

waveform = np.array(ds["speech"][0])
fs = np.array(ds["fs"][0])
sd.play(waveform, fs)
print('Sample rate:',fs,'Hz')
print('Total time:',len(waveform)/fs,'s')

# tokenize
input_values = tokenizer(ds["speech"][:2], return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)


# In[]

# from datasets import load_dataset
# from AAA import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
# import soundfile as sf
# import torch
# from jiwer import wer


# librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")

# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
# tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

# def map_to_array(batch):
#     speech, _ = sf.read(batch["file"])
#     batch["speech"] = speech
#     return batch

# librispeech_eval = librispeech_eval.map(map_to_array)

# def map_to_pred(batch):
#     input_values = tokenizer(batch["speech"], return_tensors="pt", padding="longest").input_values
#     with torch.no_grad():
#         logits = model(input_values.to("cuda")).logits

#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = tokenizer.batch_decode(predicted_ids)
#     batch["transcription"] = transcription
#     return batch

# result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])

# print("WER:", wer(result["text"], result["transcription"]))