import json
from scipy.io import wavfile

file_name = "H://WavTrain/farfield/manifest.jsonl"

x = []

with open(file_name) as f:
    for line in f:
        x.append(json.loads(line))

print(min(list(map(lambda y: len(wavfile.read("H://WavTrain/farfield/" + y["audio_filepath"])[1]), x))))

print(sum(list(map(lambda y: y["duration"], x)))/len(x))
