from scipy.io import wavfile
import matplotlib.pyplot as plt
from spectrogram import get_spectrogram,get_mel_spectrogram
import numpy as np
from WavDataset import WavDataSet
from model import ASR
import torch
from utils import decode_result
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
freq, samp = wavfile.read("WavTrain/crowd/files/736e91b6835b198a0e22249055f06572.wav","r")#Салют вызов Светлане Васильевне

print(freq)
spectrogram =get_spectrogram(samp,freq)
print(len(spectrogram))
#plt.imshow(spectrogram.transpose(), origin = "lower")
#plt.show()

dataset = WavDataSet(folder="WavTrain/crowd/")

data = torch.utils.data.DataLoader(dataset,batch_size=32,collate_fn=custom_collate,shuffle=True)

a = iter(data)


model = ASR()
model.load_state_dict(torch.load("ASR"))
model.cuda()
try:
    model.train(5,data)
except:
    pass

torch.save(model.state_dict(), "ASR")




spectrogram = torch.tensor(get_mel_spectrogram(samp, freq),dtype=torch.float32)

sequence = torch.split(spectrogram, 9)

if sequence[-1].size()[0] != 9:
    sequence = sequence[:-1]
sequence = torch.stack(sequence).cuda()

sequence -=sequence.min()
sequence /= sequence.max()
sequence = sequence.reshape(-1, 18 * 9)

sequence =torch.unsqueeze(sequence,dim=1)

result = model(sequence)
print("Тест:")
decode_result(torch.exp(result))

