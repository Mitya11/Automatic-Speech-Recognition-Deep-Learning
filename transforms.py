import random
import numpy as np

class RandomOffset:
    def __call__(self,wave):
        try:
            left_pad, right_pad = random.randint(0,16000*4 - wave.size)//2, random.randint(0,16000*4 - wave.size)//2
            wave_with_offset = np.concatenate((np.random.rand(left_pad)*1,wave,np.random.rand(right_pad)*1)).astype(int)
            return wave_with_offset
        except:
            return wave
