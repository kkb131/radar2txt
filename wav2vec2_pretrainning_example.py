# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:11:06 2021

@author: kibong
"""

import torch
from AAA import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining
from AAA.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from datasets import load_dataset
import soundfile as sf
import sounddevice as sd
import numpy as np

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wav2vec2-base")
model = Wav2Vec2ForPreTraining.from_pretrained("patrickvonplaten/wav2vec2-base")


def map_to_array(batch): 
    speech, fs = sf.read(batch["file"])
    batch["speech"] = speech
    batch["fs"] = fs
    return batch


ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

waveform = np.array(ds["speech"][0])
fs = np.array(ds["fs"][0])
sd.play(waveform, fs)

input_values = feature_extractor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1

# compute masked indices
batch_size, raw_sequence_length = input_values.shape
sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2, device=model.device)

with torch.no_grad():
    outputs = model(input_values, mask_time_indices=mask_time_indices)

# compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
cosine_sim = torch.cosine_similarity(
outputs.projected_states, outputs.projected_quantized_states, dim=-1 )

# show that cosine similarity is much higher than random
assert cosine_sim[mask_time_indices].mean() > 0.5

# for contrastive loss training model should be put into train mode
model.train()
loss = model(input_values, mask_time_indices=mask_time_indices).loss