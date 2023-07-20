import pyaudio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from spectrogram import get_spectrogram,get_mel_spectrogram
from WavDataset import WavDataSet
from model import ASR
import torch
from utils import decode_result
from transforms import RandomOffset
from datetime import datetime
import librosa
from transforms import RandomOffset
import python_speech_features as pf

CHUNK = 32000 # number of data points to read at a time
RATE = 16000 # time resolution of the recording device (Hz)

p=pyaudio.PyAudio() # start the PyAudio class
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK) #uses default input device

# create a numpy array holding a single read of audio data
model = ASR()
model.cuda()
model.load_state_dict(torch.load("ASR"))

for i in range(100000): #to it a few times just to see
    data = np.fromstring(stream.read(CHUNK,exception_on_overflow = False),dtype=np.int16)
    data = data.astype(np.float64)
    print(data)

    n_fft = int(16000 * 0.025)
    hop = n_fft // 2
    spectrogram = torch.tensor(pf.mfcc(data, RATE))
    sequence = torch.split(spectrogram, 1)

    if sequence[-1].size()[0] != 1:
        sequence = sequence[:-1]
    sequence = torch.stack(sequence).cuda()

    sequence = torch.squeeze(sequence)

    sequence = torch.unsqueeze(sequence, dim=1).cuda()

    sequence = sequence.type(torch.float32)
    # 237100
    result = model(sequence)
    print("Тест:")
    decode_result(torch.exp(result))

stream.stop_stream()
stream.close()
p.terminate()