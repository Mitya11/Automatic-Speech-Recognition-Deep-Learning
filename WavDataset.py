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
from utils import shuffle_packets
import csv
from tqdm import tqdm
class WavDataSet(Dataset):
    def __init__(self, folder, labels_file="manifest.jsonl", transform=None,delay = 170000,count = 70000,type="train"):
        self.train_data = []
        self.transform = transform
        self.folder = folder
        with open(folder+labels_file) as file:
            i= 0
            for line in file:
                if i <delay/4:
                    i += 1
                    continue
                json_line = json.loads(line)
                if not os.path.isfile(folder+json_line["audio_filepath"]):
                    continue
                if float(json_line["duration"]) > 5.5 or float(json_line["duration"]) < 1:
                    continue
                json_line["audio_filepath"] = folder+json_line["audio_filepath"]
                self.train_data.append(json_line)
                i+=1
                if i >=(count+delay)/4:
                    break
        self.train_data.sort(key= lambda x:x["duration"],reverse=False)
        self.train_data = self.train_data[:len(self.train_data) - len(self.train_data) % 32]
        if type == "train":
            hard_path = 'F:/SpeechDataset/train/manifest.csv'
        else:
            delay = 0
            hard_path = 'F:/SpeechDataset/train/manifest_test.csv'

        with open(hard_path, newline='') as csvfile:
            i= 0
            opus_files = []
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(spamreader):
                if i <delay*3:
                    i += 1
                    continue
                path , text = row
                opus_files.append({"audio_filepath":path,"text":text,"size":os.path.getsize(path)})
                i += 1
                if i >= (count+delay)*3:
                    break
            opus_files.sort(key= lambda x: x["size"])
        opus_files = opus_files[:len(opus_files) - len(opus_files) % 32]
        self.train_data.extend(opus_files)
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
        file = self.train_data[idx]["audio_filepath"]

        if file[-4:] == ".wav":
            freq, samp = wavfile.read(file, "r")
        else:
            samp, freq = librosa.load(file,
                                      res_type='scipy')
            samp = (samp * 32767).astype(np.float64)

        if self.transform:
            for transform in self.transform:
                samp = transform(samp)
        samp = np.array(samp,dtype=np.float64)
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
        target = []
        for i in self.train_data[idx]["text"]:
            if i in alphabet:
                target.append(alphabet[i])
        target = torch.tensor(target[:80]+[35])

        return sequence[:600], target
