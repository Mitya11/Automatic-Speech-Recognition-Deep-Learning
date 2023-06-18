from scipy.io import wavfile
import matplotlib.pyplot as plt
from spectrogram import get_spectrogram
import numpy as np
from WavDataset import WavDataSet
from model import ASR
import torch
from utils import decode_result
def custom_collate(batch):
    inputs = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[0],batch)))
    target = torch.transpose(torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[1],batch))),0,1)
    target = torch.cat([target, torch.zeros((target.size()[0],max(inputs.size()[0]-target.size()[1],0)))],dim=1)
    inputs = torch.unsqueeze(inputs,dim=2)
    inputs = inputs.type(torch.float32)
    return inputs,target
freq, samp = wavfile.read("H://WavTrain/farfield/files/26093ef7a8c5ec6a3c586a6a929c1bd3.wav","r")#Салют вызов Светлане Васильевне

print(freq)
spectrogram =get_spectrogram(samp,freq)
print(len(spectrogram))
#plt.imshow(spectrogram.transpose(), origin = "lower")
#plt.show()

dataset = WavDataSet(folder="H://WavTrain/farfield/")

data = torch.utils.data.DataLoader(dataset,batch_size=32,collate_fn=custom_collate,shuffle=True)

a = iter(data)

inp, tar = next(a)

model = ASR()
model.load_state_dict(torch.load("ASR"))

model.train(10,data)

torch.save(model.state_dict(), "ASR")




spectrogram = torch.tensor(get_spectrogram(samp, freq),dtype=torch.float32)

sequence = torch.split(spectrogram, 6)

if sequence[-1].size()[0] != 6:
    sequence = sequence[:-1]

sequence =torch.unsqueeze(torch.unsqueeze(torch.stack(sequence),dim=1),dim=1)



result = model(sequence)

decode_result(torch.exp(result))

