import torch
import json
from torch.utils.data import Dataset, DataLoader
from spectrogram import get_spectrogram
from scipy.io import wavfile
from utils import alphabet


class WavDataSet(Dataset):
    def __init__(self, folder, labels_file="manifest.jsonl", transform=None):
        self.train_data = []
        self.folder = folder
        with open(folder+labels_file) as file:
            i= 0
            for line in file:
                self.train_data.append(json.loads(line))
                i+=1
                if i >=128:
                    break
        #self.train_data = self.train_data[1:2]
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):

        split_size = 6
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.train_data[idx]["audio_filepath"]

        freq, samp = wavfile.read(self.folder + file, "r")

        spectrogram = torch.tensor(get_spectrogram(samp, freq))

        sequence = torch.split(spectrogram, split_size)

        if sequence[-1].size()[0] != split_size:
            sequence = sequence[:-1]

        sequence =torch.stack(sequence)
        sequence = torch.nn.functional.normalize(sequence,dim=0)
        target = torch.tensor([alphabet[i] for i in self.train_data[idx]["text"]])

        return sequence, target
