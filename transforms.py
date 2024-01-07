import random
import numpy as np

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
