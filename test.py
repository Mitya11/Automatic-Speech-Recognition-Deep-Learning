import os
import shutil
from tqdm import tqdm
from glob import iglob
import librosa
import io
import json
import base64

source_folder = "G:/SpeechDataset/train"
destination_folder = "G:/SpeechDataset/flatted"
with open("G:/SpeechDataset/train/SpeechData2.jsonl", 'w') as result:
    for fn in tqdm(iglob('G:/SpeechDataset/train/*/*/*/*.txt')):
        file = fn.split('.')[0]
        with open(file+".txt","r",encoding="utf-8") as txt:
            text = txt.read().strip()
        with open(file+".opus","rb") as opus:
            bytes = base64.b64encode(opus.read())
            content = librosa.load(io.BytesIO(base64.b64decode(bytes)))
        duration = librosa.get_duration(path=file+".opus")
        json_line = {"text":text , "audio":bytes, "duration":duration}
        #json_line = json.dumps(str(json_line))

        result.write(str(json_line).replace("'",'"')+"\n")

#data = librosa.load(bytes,res_type='scipy')
with open("G:/SpeechDataset/train/SpeechData.jsonl", 'r') as result:
    for line in tqdm(result):
        a = line.strip()
        data = eval(a)
        text = data["text"]
        audio = io.BytesIO(base64.b64decode(data["audio"]))
        opus = librosa.load(audio)
        duration = data["duration"]
print("content")