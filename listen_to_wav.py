import pyaudio
import wave
import numpy as np
import scipy.signal

import transforms
import io
from scipy.io import wavfile
import sounddevice as sd
import torch
from torch_audiomentations  import *
import os
import soundfile as sf
import librosa
import python_speech_features_cuda as pf

fs = 16000  # Sample rate
seconds = 3.5  # Duration of recording

print("recording...")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)[:,0]
sd.wait()  # Wait until recording is finished
print("end recording...")
wavfile.write('karplus.wav', 16000, myrecording)  # Save as WAV file

transforms = [PitchShift(-2,2,p=0.2,sample_rate=fs),
              AddBackgroundNoise("augmentation/",15,21,sample_rate=fs,p=0.2),
              Gain(-10,10,p=0.2),
              PolarityInversion(p=0.2)]

samp, freq = sf.read("G:/SpeechDataset/train\public_youtube1120_hq/1/0c/3cae2b8ebb60.opus", dtype='float32')


reverbed = myrecording

impact, freq = sf.read("S1R1_sweep4000.wav", dtype='float32')
impact = impact[:10000:2]

reverbed = scipy.signal.convolve(reverbed,impact,mode="same")
wavfile.write('karplusReverb.wav', 16000, reverbed*0.03 + myrecording*0.97)  # Save as WAV file
