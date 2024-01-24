import librosa
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
import torchaudio
class WavDataSet(Dataset):
    def __init__(self, hard_path, labels_file="manifest.jsonl", transform=None,delay = 0,count = 600000,type="train"):
        self.train_data = []
        self.transform = transform

        if type == "train":
            pass
        else:
            delay = delay+count
            count = 25000
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
        for i in range(int(count * 0.00)):
            self.train_data.append({"text": "noise", "audio":None, "duration": random.uniform(1,5)})
        self.train_data.sort(key = lambda x:len(x["text"]))
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
            samp = torch.tensor([[np.squeeze(samp)]]).to(torch.float32)
        else:
            audio = io.BytesIO(base64.b64decode(data["audio"]))
            duration = data["duration"]
            transcript = data["transcript"]

            samp, freq = sf.read(audio, dtype='float32')
            assert freq ==16000
            samp = torch.tensor([[np.squeeze(samp)]]).to(torch.float32)


        samp = torch.tensor(samp[0][0])

        assert samp.isnan().any().item() == 0
        target = torch.tensor(transcript)

        return samp[:freq*10], target[:280]
