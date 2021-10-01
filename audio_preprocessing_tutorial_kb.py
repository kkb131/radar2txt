# When running this tutorial in Google Colab, install the required packages
# with the following.
# !pip install torchaudio librosa boto3

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

import audio_preprocessing as ap
import requests
import tarfile
import boto3

# In[]
metadata = torchaudio.info(ap.SAMPLE_WAV_PATH)
print(metadata)

# metadata = torchaudio.info(SAMPLE_MP3_PATH)
# print(metadata)

# metadata = torchaudio.info(SAMPLE_GSM_URL)
# print(metadata)

# In[]

# print("Source:", ap.SAMPLE_WAV_URL)
# with requests.get(ap.SAMPLE_WAV_URL, stream=True) as response:
#   metadata = torchaudio.info(response.raw)
# print(metadata)

# print("Source:", ap.SAMPLE_MP3_URL)
# with requests.get(ap.SAMPLE_MP3_URL, stream=True) as response:
#     metadata = torchaudio.info(response.raw, format="mp3")

#     print(f"Fetched {response.raw.tell()} bytes.")
# print(metadata)


# In[]
waveform, sample_rate = torchaudio.load(ap.SAMPLE_WAV_SPEECH_PATH)

ap.print_stats(waveform, sample_rate=sample_rate)
ap.plot_waveform(waveform, sample_rate)
ap.plot_specgram(waveform, sample_rate)
ap.play_audio2(waveform, sample_rate)


# In[]

# Load audio from tar file
with tarfile.open(ap.SAMPLE_TAR_PATH, mode='r') as tarfile_:
  fileobj = tarfile_.extractfile(ap.SAMPLE_TAR_ITEM)
  waveform, sample_rate = torchaudio.load(fileobj)
ap.plot_specgram(waveform, sample_rate, title="TAR file")

# # Load audio from S3
# client = boto3.client('s3', config=ap.Config(signature_version=ap.UNSIGNED))
# response = client.get_object(Bucket=ap.S3_BUCKET, Key=ap.S3_KEY)
# waveform, sample_rate = torchaudio.load(response['Body'])
# ap.plot_specgram(waveform, sample_rate, title="From S3")


# In[]
ap.YESNO_DOWNLOAD_PROCESS.join()

dataset = torchaudio.datasets.YESNO(ap.YESNO_DATASET_PATH, download=True)


# In[]

for i in [1, 3, 5]:
    waveform, sample_rate, label = dataset[i]
    ap.plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
    ap.play_audio(waveform, sample_rate)
  
# In[]

dataset2 = torchaudio.datasets.LIBRISPEECH("G:/data/libri/", 
                                           url = "test-clean",
                                           folder_in_archive = "LibriSpeech",
                                           download=False,)

for i in [1, 3, 5]:
    waveform, sample_rate, label = dataset[i]
    ap.plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
    ap.play_audio2(waveform, sample_rate)
    
# In[]
# waveform, sample_rate = ap.get_speech_sample()
 
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

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

for i in [1, 3, 5]:
    waveform, sample_rate, script, sp_id, book_id, data_num  = dataset2[i]
    
    melspec = mel_spectrogram(waveform)
    ap.plot_spectrogram(
        melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')
