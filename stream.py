import pyaudio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from spectrogram import get_spectrogram,get_mel_spectrogram
from WavDataset import WavDataSet
from SpeechRecognition import SpeechRecognition
import torch
from utils import decode_result,get_features
from transforms import RandomOffset
from datetime import datetime
import librosa
from transforms import RandomOffset


CHUNK = int(2*22000) # number of data points to read at a time
RATE = 22000 # time resolution of the recording device (Hz)

p=pyaudio.PyAudio() # start the PyAudio class
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK) #uses default input device

# create a numpy array holding a single read of audio data
model = SpeechRecognition()
model.load(False)
with torch.no_grad():
    for i in range(10000): #to it a few times just to see
        data = np.fromstring(stream.read(CHUNK,exception_on_overflow = False),dtype=np.int16)
        data = (data).astype(np.float32)
        wavfile.write("karplus.wav", RATE, data)

        print(data)
        #data = nr.reduce_noise(data, RATE)

        sequence = get_features(data,RATE).unsqueeze(dim=1).to(torch.float32).cuda()      # 237100
        result = model(sequence)
        print("Тест:")
        print(decode_result(result))

stream.stop_stream()
stream.close()
p.terminate()
