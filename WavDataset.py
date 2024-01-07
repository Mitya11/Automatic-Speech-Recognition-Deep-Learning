import torch
import json
from torch.utils.data import Dataset, DataLoader
from spectrogram import get_spectrogram , get_mel_spectrogram
from scipy.io import wavfile
from utils import alphabet , get_transcript
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import python_speech_features as pf
from utils import shuffle_packets
import csv
from tqdm import tqdm
import base64
import io
import soundfile as sf

class WavDataSet(Dataset):
    def __init__(self, hard_path, labels_file="manifest.jsonl", transform=None,delay = 0,count = 600000,type="train"):
        self.train_data = []
        self.transform = transform

        if type == "train":
            pass
        else:
            delay = delay+count
            count = 6000
        with open(hard_path, newline='') as file:
            i= 0
            for line in tqdm(file):
                if i < delay:
                    i += 1
                    continue
                a = line.strip()
                data = eval(a)
                o = data["transcript"]
                if data["duration"] > 6:
                    continue
                self.train_data.append(data)
                i += 1
                if i +delay >= count:
                    break

        # noise generating
        for i in range(int(count * 0.1)):
            self.train_data.append({"text": "noise", "audio":None, "duration": random.uniform(1,5)})
        self.train_data.sort(key = lambda x:float(x["duration"]))
        self.train_data = self.train_data[:len(self.train_data) - len(self.train_data)%32]
        self.train_data = shuffle_packets(self.train_data,32)

        #self.train_data = self.train_data[:512]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):

        split_size = 1
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx == 4051:
            k = 0
        data = self.train_data[idx]

        text = data["text"]
        if text == "noise":
            samp, freq = np.random.normal(0,random.uniform(0,50), int(22000*float(data["duration"]))).astype(np.float32), 22000
            text = ""
            transcript = [178]
        else:
            audio = io.BytesIO(base64.b64decode(data["audio"]))
            duration = data["duration"]
            transcript = data["transcript"]

            samp, freq = sf.read(audio, dtype='int16')
            samp = (samp).astype(np.float32)

        if self.transform:
            for transform in self.transform:
                samp = transform(samp)
        samp = np.array(samp,dtype=np.float32)
        n_fft = 512
        hop = 160
        mfcc = pf.mfcc(samp, freq, nfilt=40,nfft=350,winlen=0.015,winstep=0.01)
        delta_mfcc = pf.delta(mfcc,2)
        a_mfcc = pf.delta(delta_mfcc,2)
        features = torch.tensor(np.concatenate([mfcc,delta_mfcc,a_mfcc],axis=1))




        #spectrogram = torch.tensor(get_mel_spectrogram(samp, freq))
        """plt.imshow(spectrogram.transpose(0,1), origin = "lower")
        plt.show()"""
        sequence = torch.split(features, split_size)

        if sequence[-1].size()[0] != split_size:
            sequence = sequence[:-1]

        sequence =torch.stack(sequence)
        #standarize
        #l = sequence.min()[0]
        #m = sequence.max()[0]
        #sequence -=l
        #sequence = sequence / m *2
        #sequence = torch.squeeze(sequence)
        assert sequence.isnan().any().item() == 0
        target = torch.tensor(transcript)

        return sequence[:1300], target[:180]
