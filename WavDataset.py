import torch
import json
from torch.utils.data import Dataset, DataLoader
from spectrogram import get_spectrogram , get_mel_spectrogram
from scipy.io import wavfile
from utils import alphabet
import matplotlib.pyplot as plt
import os
import random
import librosa
import numpy as np
import python_speech_features as pf

class WavDataSet(Dataset):
    def __init__(self, folder, labels_file="manifest.jsonl", transform=None,count = 330024):
        self.train_data = []
        self.transform = transform
        self.folder = folder
        with open(folder+labels_file) as file:
            i= 0
            for line in file:
                json_line = json.loads(line)
                if not os.path.isfile(folder+json_line["audio_filepath"]):
                    continue
                if float(json_line["duration"]) > 6 or float(json_line["duration"]) < 1:
                    continue
                self.train_data.append(json_line)
                i+=1
                if i >=count:
                    break
        #self.train_data = self.train_data[-2500:-2]
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):

        split_size = 1
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx == 4051:
            k = 0
        file = self.train_data[idx]["audio_filepath"]

        freq, samp = wavfile.read(self.folder + file, "r")

        if self.transform:
            for transform in self.transform:
                samp = transform(samp)
        samp = np.array(samp,dtype=np.float64)
        n_fft = 512
        hop = 160
        spectrogram = torch.tensor(pf.mfcc(samp,freq))




        #spectrogram = torch.tensor(get_mel_spectrogram(samp, freq))
        """plt.imshow(spectrogram.transpose(0,1), origin = "lower")
        plt.show()"""
        sequence = torch.split(spectrogram, split_size)

        if sequence[-1].size()[0] != split_size:
            sequence = sequence[:-1]

        sequence =torch.stack(sequence)
        #standarize
        l = sequence.min()
        #sequence -=sequence.min()
        #sequence /= 20
        sequence = torch.squeeze(sequence)
        assert sequence.isnan().any().item() == 0
        target = torch.tensor([alphabet[i] for i in self.train_data[idx]["text"]])

        return sequence, target
