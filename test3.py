from  scipy.io import wavfile
from SpeechRecognition import SpeechRecognition
import torch
from utils import decode_result,get_features
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
samp, freq = sf.read("G:/SpeechDataset/train\public_youtube1120_hq/1/0c/3cae2b8ebb60.opus", dtype='float32')

samp = torch.tensor(samp,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#samp = (samp * 32767).astype(np.float32)
#wavfile.write("karplus.wav", 22000, samp)

sampling_rate, data = wavfile.read("karplus.wav")
data = torch.tensor(data,dtype=torch.float32).unsqueeze(0)
model = SpeechRecognition()
model.load(False)
#data = nr.reduce_noise(data, sampling_rate,prop_decrease=0.3)

features = get_features(data.transpose(0,1),sampling_rate)[0].unsqueeze(dim=1).to(torch.float32).cuda()

k = features.cpu().numpy()
result = model(features)
print("Тест:")
print(decode_result(result))
