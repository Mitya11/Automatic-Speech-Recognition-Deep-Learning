from scipy.io import wavfile
import matplotlib.pyplot as plt
from spectrogram import get_spectrogram,get_mel_spectrogram
import numpy as np
from WavDataset import WavDataSet
import torch
from utils import decode_result
from transforms import RandomOffset,RandomNoise
from datetime import datetime
from SpeechRecognition import SpeechRecognition
import time
import gc
#torch.set_num_threads(8)
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
def custom_collate(batch):
    input_lengths = torch.tensor(list(map(lambda x:x[0].size(dim=0),batch)))
    target_lengths = torch.tensor(list(map(lambda x:x[1].size(dim=0),batch)))
    inputs = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[0],batch)))
    target = torch.transpose(torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[1],batch)),padding_value=179),0,1)
    #target = torch.cat([target, torch.zeros((target.size()[0],max(inputs.size()[0]-target.size()[1],0)))],dim=1)
    target = torch.where(target ==0,0,target)
    """if target.size()[1] >= inputs.size()[0]:
        target = target[:,:inputs.size()[0]-2]"""
    target_lengths = torch.where(target_lengths>=input_lengths,input_lengths-2,target_lengths)
    inputs = inputs.type(torch.float32)
    return inputs,target , input_lengths, target_lengths

start_time = datetime.now()
torch.set_printoptions(precision=3)

COUNT_DATA = 6
data_files = ["G:/SpeechDataset/train/split_files/output0.jsonl","G:/SpeechDataset/train/split_files/output1.jsonl","G:/SpeechDataset/train/split_files/output2.jsonl",
"G:/SpeechDataset/train/split_files/output3.jsonl","G:/SpeechDataset/train/split_files/output4.jsonl",
"G:/SpeechDataset/train/split_files/output5.jsonl","G:/SpeechDataset/train/split_files/output6.jsonl",
            "G:/SpeechDataset/train/split_files/output7.jsonl","G:/SpeechDataset/train/split_files/output8.jsonl",
              "G:/SpeechDataset/train/split_files/output9.jsonl","G:/SpeechDataset/train/split_files/output10.jsonl"]

model = SpeechRecognition()

for i in range(1,COUNT_DATA):

    train = WavDataSet(hard_path=data_files[i],type = "train")#,transform=[RandomOffset()])
    train_data = torch.utils.data.DataLoader(train,batch_size=32,collate_fn=custom_collate,shuffle=False)

    if i == COUNT_DATA-1:
        val = WavDataSet(hard_path="G:/SpeechDataset/train/split_files/valdata11.jsonl",type = "valid",count=2000)#,transform=[RandomOffset()])
        val_data = torch.utils.data.DataLoader(val,batch_size=32,collate_fn=custom_collate,shuffle=False)
    else:
        val_data = None

    model.train(1, train_data,val_data)
    del train
    del train_data
    gc.collect()
    time.sleep(5)
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
