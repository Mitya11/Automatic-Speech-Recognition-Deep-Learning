import torch
from model import EncoderNN, DecoderNN
import numpy as np
import python_speech_features as pf
import noisereduce as nr


class SpeechRecognition:
    def __init__(self):
        self.encoder = EncoderNN()
        self.decoder = DecoderNN()

    def __call__(self, samples, freq):
        samples = samples.astype(np.float64)
        samples = nr.reduce_noise(samples, freq)
        mfcc = torch.tensor(pf.mfcc(samples, freq, numcep=18))
        mfcc = torch.unsqueeze(mfcc, dim=1).type(torch.float32)

        encoder_output, prev_hidden = self.encoder(mfcc)

        result = []
        output, hidden = self.decoder(encoder_output, prev_hidden, torch.zeros((1), dtype=torch.int32))
        for i in range(encoder_output.size()[0]):
            output, hidden = self.decoder(encoder_output, hidden, torch.argmax(output, dim=1))

        return encoder_output

    def train(self, epochesCount ,train_data,val_data= None):
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.02)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.02)
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(epochesCount):
            print("Epoch:", epoch + 1)
            it = iter(train_data)
            NLL = torch.nn.NLLLoss()
            sr = 0
            for i in range(len(train_data)):
                input ,target, _,_ = next(it)
                loss = torch.tensor(0,dtype=torch.float64)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_output, prev_hidden = self.encoder(input)

                output = torch.zeros((1),dtype=torch.long)
                hidden = (prev_hidden,prev_hidden)

                result = []
                for j in range(target.size()[1]):
                    #teacher forcing
                    output, hidden = self.decoder(encoder_output, hidden, output)
                    loss += NLL(output,target[:,j])
                    result.append(torch.argmax(output, dim=1).item())
                    output = target[:,j]


                from utils import alphabet
                import itertools
                text = "".join(list(map(lambda x: alphabet[x], result)))
                result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
                print(result, len(text))


                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                sr += float(loss)
            print("LOSS:",sr)

