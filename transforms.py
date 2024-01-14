import random
import numpy as np
import torchaudio
import torch
class RandomOffset:
    def __call__(self,wave):
        try:
            left_pad, right_pad = random.randint(0,16000*4 - wave.size)//2, random.randint(0,16000*4 - wave.size)//2
            wave_with_offset = np.concatenate((np.random.normal(0,40,left_pad)*1,wave,np.random.normal(0,40,right_pad)*1)).astype(int)
            return wave_with_offset
        except:
            return wave
class RandomNoise:
    def __call__(self, wave):
        period = random.randint(3,10)
        multypliers = np.random.rand(period)
        noise = np.random.normal(0,50,wave.size)
        noise *= np.interp(range(wave.size),np.linspace(0,wave.size,period),multypliers)
        return wave+noise

class RoomReverb:
    def __call__(self,wave,power = 1):
        from torchaudio.utils import download_asset

        rir_raw, sample_rate = torchaudio.load("augmentation/S1R2_sweep4000.wav")
        rir = rir_raw[:, int(sample_rate * 0.01): int(sample_rate * 1.3)][0]
        rir = rir / torch.linalg.vector_norm(rir, ord=2)
        o =torchaudio.functional.fftconvolve(wave,rir,mode="full")
        return torchaudio.functional.fftconvolve(wave,rir,mode="full")