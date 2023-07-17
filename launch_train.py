from scipy.io import wavfile
import matplotlib.pyplot as plt
from spectrogram import get_spectrogram,get_mel_spectrogram
import numpy as np
from WavDataset import WavDataSet
from model import ASR
import torch
from utils import decode_result
from transforms import RandomOffset
from datetime import datetime

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
    return inputs,target , input_lengths, target_lengths
freq, samp = wavfile.read("WavTrain/crowd/files/794915d4730d7bd2870506ed5174d3ad.wav","r")#Салют вызов Светлане Васильевне

print(freq)
spectrogram =get_spectrogram(samp,freq,512)
print(len(spectrogram))
#plt.imshow(spectrogram.transpose(), origin = "lower")
#plt.show()

dataset = WavDataSet(folder="WavTrain/train/",transform=RandomOffset())

train = torch.utils.data.Subset(dataset, range(int(len(dataset)-533)))
val = WavDataSet(folder="WavTrain/crowd/",count=533)
train_data = torch.utils.data.DataLoader(train,batch_size=48,collate_fn=custom_collate,shuffle=True)
val_data = torch.utils.data.DataLoader(val,batch_size=32,collate_fn=custom_collate,shuffle=False)


model = ASR()
model.load_state_dict(torch.load("ASR"))
model.cuda()
start_time = datetime.now()
#model.train(1, train_data,val_data)
model.validate_epoch(val_data)
try:
    1
except:
    pass

torch.save(model.state_dict(), "ASR")

print("Total Time: {}".format(datetime.now()-start_time))



"""spectrogram = torch.tensor(get_mel_spectrogram(samp, freq),dtype=torch.float32)

sequence = torch.split(spectrogram, 2)

if sequence[-1].size()[0] != 2:
    sequence = sequence[:-1]
sequence = torch.stack(sequence).cuda()

sequence = sequence.reshape(-1, 28 * 2)

sequence =torch.unsqueeze(sequence,dim=1).cpu()
sequence = torch.nn.BatchNorm1d(1)(sequence)

#237100
result = model(sequence)
print("Тест:")
decode_result(torch.exp(result))
"""
