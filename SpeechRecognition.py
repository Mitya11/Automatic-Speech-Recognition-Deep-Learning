import torch
from model import EncoderNN, DecoderNN
import numpy as np
import python_speech_features as pf
import noisereduce as nr


class SpeechRecognition:
    def __init__(self):
        self.encoder = EncoderNN()
        self.decoder = DecoderNN()
        self.device = torch.device("cpu")

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()
            self.device = torch.device("cuda")

    def __call__(self, samples, freq):
        samples = samples.astype(np.float64)
        samples = nr.reduce_noise(samples, freq)
        mfcc = torch.tensor(pf.mfcc(samples, freq, numcep=18),device = self.device)
        mfcc = torch.unsqueeze(mfcc, dim=1).type(torch.float32)

        encoder_output, prev_hidden = self.encoder(mfcc)

        result = []
        output, hidden = self.decoder(encoder_output, prev_hidden, torch.zeros((1), dtype=torch.int32))
        for i in range(encoder_output.size()[0]):
            output, hidden = self.decoder(encoder_output, hidden, torch.argmax(output, dim=1))

        return encoder_output

    def train(self, epochesCount ,train_data,val_data= None):
        self.load()
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.02)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.02)
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(epochesCount):
            print("Epoch:", epoch + 1)
            it = iter(train_data)
            NLL = torch.nn.NLLLoss(ignore_index=0)
            sr = 0
            for i in range(len(train_data)):
                inputs ,target, _,_ = next(it)
                inputs = inputs.to(self.device)
                target = target.to(self.device)

                loss = torch.tensor(0,dtype=torch.float64,device = self.device)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_output, prev_hidden = self.encoder(inputs)

                output = torch.zeros((encoder_output.shape[1]),dtype=torch.long,device = self.device)
                hidden = (prev_hidden.repeat(2,1,1),prev_hidden.repeat(2,1,1))

                result = []
                for j in range(target.size()[1]):
                    #teacher forcing
                    output, hidden = self.decoder(encoder_output, hidden, output)
                    loss += NLL(output,target[:,j]).nan_to_num(0)
                    result.append(torch.argmax(output[0:1], dim=1).item())
                    output = target[:,j]


                from utils import alphabet
                import itertools
                text = "".join(list(map(lambda x: alphabet[x], result)))
                result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
                print(result, len(text),"Completed:", round(i/len(train_data)*100,4),"%")


                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                sr += float(loss)
            print("LOSS:",sr)

            if val_data:
                it = iter(val_data)
                sr = 0
                NLL = torch.nn.NLLLoss(ignore_index=0)

                with torch.no_grad():
                    for i in range(len(val_data)):
                        inputs, target, _, _ = next(it)
                        inputs = inputs.to(self.device)
                        target = target.to(self.device)

                        encoder_output, prev_hidden = self.encoder(inputs)

                        output = torch.zeros((encoder_output.shape[1]), dtype=torch.long, device=self.device)
                        hidden = (prev_hidden.repeat(2, 1, 1), prev_hidden.repeat(2, 1, 1))

                        result = []
                        for j in range(target.size()[1]):
                            # teacher forcing
                            output, hidden = self.decoder(encoder_output, hidden, output)
                            sr += NLL(output, target[:, j]).nan_to_num(0)

                            result.append(torch.argmax(output[0:1], dim=1).item())
                            output = torch.argmax(output, dim=1)

                        from utils import alphabet
                        import itertools
                        text = "".join(list(map(lambda x: alphabet[x], result)))
                        result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
                        print(result, len(text))
                        print("                -----", "".join([alphabet[i.item()] for i in target[0]]))
                    print("VALIDATE Loss:", sr, "")
        self.save()


    def load(self):
        self.encoder.load_state_dict(torch.load("trained_param/encoder_params"))
        self.decoder.load_state_dict(torch.load("trained_param/decoder_params"))
    def save(self):
        torch.save(self.encoder.state_dict(), "trained_param/encoder_params")
        torch.save(self.decoder.state_dict(), "trained_param/decoder_params")
