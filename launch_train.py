from scipy.io import wavfile
import matplotlib.pyplot as plt
from spectrogram import get_spectrogram,get_mel_spectrogram
import numpy as np
from WavDataset import WavDataSet
from model import ASR
import torch
from utils import decode_result
from transforms import RandomOffset

#torch.set_num_threads(8)
def custom_collate(batch):
    input_lengths = torch.tensor(list(map(lambda x:x[0].size(dim=0),batch)))
    target_lengths = torch.tensor(list(map(lambda x:x[1].size(dim=0),batch)))
    inputs = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[0],batch)))
    target = torch.transpose(torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[1],batch))),0,1)
    #target = torch.cat([target, torch.zeros((target.size()[0],max(inputs.size()[0]-target.size()[1],0)))],dim=1)
    target = torch.where(target ==0,33,target)
    """if target.size()[1] >= inputs.size()[0]:
        target = target[:,:inputs.size()[0]-2]"""
    target_lengths = torch.where(target_lengths>=input_lengths,input_lengths-2,target_lengths)
    inputs = inputs.type(torch.float32)
    inputs = torch.nn.BatchNorm1d(inputs.size(dim=1))(inputs)
    return inputs,target , input_lengths, target_lengths
freq, samp = wavfile.read("WavTrain/crowd/files/1286be06c90555003cd7fd5dff85ac4d.wav","r")#Салют вызов Светлане Васильевне

print(freq)
spectrogram =get_spectrogram(samp,freq)
print(len(spectrogram))
#plt.imshow(spectrogram.transpose(), origin = "lower")
#plt.show()

dataset = WavDataSet(folder="WavTrain/train/",transform=RandomOffset())

train = torch.utils.data.Subset(dataset, range(int(len(dataset)*0.85)))
val = torch.utils.data.Subset(dataset, range(int(len(dataset)*0.85),len(dataset)))
val.transforms = None
train_data = torch.utils.data.DataLoader(train,batch_size=48,collate_fn=custom_collate,shuffle=True)
val_data = torch.utils.data.DataLoader(val,batch_size=32,collate_fn=custom_collate,shuffle=False)


model = ASR()
model.load_state_dict(torch.load("ASR"))
model.cuda()
model.train(1, train_data,val_data)

try:
    1
except:
    pass

torch.save(model.state_dict(), "ASR")




spectrogram = torch.tensor(get_mel_spectrogram(samp, freq),dtype=torch.float32)

sequence = torch.split(spectrogram, 3)

if sequence[-1].size()[0] != 3:
    sequence = sequence[:-1]
sequence = torch.stack(sequence).cuda()

sequence = sequence.reshape(-1, 29 * 3)

sequence =torch.unsqueeze(sequence,dim=1).cpu()
sequence = torch.nn.BatchNorm1d(1)(sequence)
result = model(sequence)
print("Тест:")
decode_result(torch.exp(result))

