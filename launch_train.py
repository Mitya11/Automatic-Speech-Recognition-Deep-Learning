from scipy.io import wavfile
import matplotlib.pyplot as plt
from spectrogram import get_spectrogram,get_mel_spectrogram
import numpy as np
from WavDataset import WavDataSet
from model import ASR
import torch
from utils import decode_result
from transforms import RandomOffset,RandomNoise
from datetime import datetime
from SpeechRecognition import SpeechRecognition
#torch.set_num_threads(8)
def custom_collate(batch):
    input_lengths = torch.tensor(list(map(lambda x:x[0].size(dim=0),batch)))
    target_lengths = torch.tensor(list(map(lambda x:x[1].size(dim=0),batch)))
    inputs = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[0],batch)))
    target = torch.transpose(torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[1],batch))),0,1)
    #target = torch.cat([target, torch.zeros((target.size()[0],max(inputs.size()[0]-target.size()[1],0)))],dim=1)
    target = torch.where(target ==0,0,target)
    """if target.size()[1] >= inputs.size()[0]:
        target = target[:,:inputs.size()[0]-2]"""
    target_lengths = torch.where(target_lengths>=input_lengths,input_lengths-2,target_lengths)
    inputs = inputs.type(torch.float32)
    return inputs,target , input_lengths, target_lengths


dataset = WavDataSet(folder="H://WavTrain/crowd/")#,transform=[RandomOffset()])

train = dataset
#val = WavDataSet(folder="WavTrain/crowd/",count=2000)
train_data = torch.utils.data.DataLoader(train,batch_size=48,collate_fn=custom_collate,shuffle=True)
#val_data = torch.utils.data.DataLoader(val,batch_size=32,collate_fn=custom_collate,shuffle=False)

torch.set_printoptions(precision=3)
model = SpeechRecognition()
#model.load_state_dict(torch.load("ASR"))
start_time = datetime.now()
model.train(20, train_data)
#model.validate_epoch(val_data)
try:
    1
except:
    pass

#torch.save(model.state_dict(), "ASR")

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
