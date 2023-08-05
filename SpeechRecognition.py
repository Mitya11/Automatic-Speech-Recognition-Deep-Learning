import torch
from model.CTC import CTCdecoder
from model.decoder import DecoderNN
from model.encoder import EncoderNN
import numpy as np
import python_speech_features as pf
import noisereduce as nr
import random


class SpeechRecognition:
    def __init__(self):
        self.encoder = EncoderNN()
        self.decoder = DecoderNN()
        self.device = torch.device("cpu")
        self.ctc_classifier = None
        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()
            self.device = torch.device("cuda")

    def __call__(self, samples, freq):
        samples = samples.astype(np.float64)
        samples = nr.reduce_noise(samples, freq)
        mfcc = torch.tensor(pf.mfcc(samples, freq, numcep=18), device=self.device)
        mfcc = torch.unsqueeze(mfcc, dim=1).type(torch.float32)

        encoder_output, prev_hidden = self.encoder(mfcc)

        result = []
        output, hidden = self.decoder(encoder_output, prev_hidden, torch.zeros((1), dtype=torch.int32))
        for i in range(encoder_output.size()[0]):
            output, hidden = self.decoder(encoder_output, hidden, torch.argmax(output, dim=1))

        return encoder_output

    def train(self, epochesCount, train_data, val_data=None):

        self.ctc_classifier = CTCdecoder()

        self.load(ctc_load=True)
        optimizer = torch.optim.Adam(list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.ctc_classifier.parameters()),lr=0.001)
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(epochesCount):
            print("Epoch:", epoch + 1)
            it = iter(train_data)
            criterion_cross = torch.nn.CrossEntropyLoss(ignore_index=0)
            criterion_ctc = torch.nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
            sr = 0
            for i in range(len(train_data)):
                inputs, target, input_lengths, target_lengths = next(it)
                inputs = inputs.to(self.device)
                target = target.to(self.device)

                loss = torch.tensor(0, dtype=torch.float64, device=self.device)
                optimizer.zero_grad()

                encoder_output, prev_hidden = self.encoder(inputs)
                # CTC-based model
                ctc_output = self.ctc_classifier(encoder_output)
                input_lengths = input_lengths // 8
                loss = criterion_ctc(ctc_output.transpose(0, 1), target, input_lengths, target_lengths)

                # Attention-based model
                output = torch.nn.functional.one_hot(
                    torch.full([encoder_output.shape[0]], 34, dtype=torch.long, device=self.device), num_classes=36).to(
                    dtype=torch.float32)
                rnn_input = torch.cat([output.unsqueeze(dim=1), encoder_output[:, 0:1, :]], dim=-1)

                hidden = None
                result = []
                for j in range(target.size()[1]):
                    # teacher forcing
                    output, hidden, context = self.decoder(encoder_output, rnn_input, hidden)
                    loss += criterion_cross(output, target[:, j]).nan_to_num(0)
                    result.append(torch.argmax(output[0:1], dim=1).item())
                    if random.randint(1, 100) < 60:
                        output = target[:, j]
                        output = torch.nn.functional.one_hot(
                            output, num_classes=36).to(
                            dtype=torch.float32)

                    rnn_input = torch.cat([output.unsqueeze(dim=1), context.unsqueeze(1)], dim=-1)

                from utils import alphabet
                import itertools
                text = "".join(list(map(lambda x: alphabet[x], result)))
                result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
                print(result, len(text), "Completed:", round(i / len(train_data) * 100, 4), "%")

                loss.backward()
                # print(self.encoder.lstm1.all_weights[0][0].grad)
                optimizer.step()

                sr += float(loss)
            print("LOSS:", sr)

            if val_data:
                it = iter(val_data)
                sr = 0
                NLL = torch.nn.CrossEntropyLoss(ignore_index=0)

                with torch.no_grad():
                    for i in range(len(val_data)):
                        inputs, target, _, _ = next(it)
                        inputs = inputs.to(self.device)
                        target = target.to(self.device)

                        encoder_output, prev_hidden = self.encoder(inputs)

                        output = torch.nn.functional.one_hot(
                            torch.full([encoder_output.shape[0]], 34, dtype=torch.long, device=self.device),
                            num_classes=36).to(dtype=torch.float32)
                        rnn_input = torch.cat([output.unsqueeze(dim=1), encoder_output[:, 0:1, :]], dim=-1)

                        hidden = None
                        result = []
                        for j in range(target.size()[1]):
                            # teacher forcing
                            output, hidden, context = self.decoder(encoder_output, rnn_input, hidden)
                            sr += NLL(output, target[:, j]).nan_to_num(0)
                            result.append(torch.argmax(output[0:1], dim=1).item())
                            if random.randint(1, 100) < 60:
                                output = target[:, j]
                                output = torch.nn.functional.one_hot(
                                    output, num_classes=36).to(
                                    dtype=torch.float32)

                            rnn_input = torch.cat([output.unsqueeze(dim=1), context.unsqueeze(1)], dim=-1)

                        from utils import alphabet
                        import itertools
                        text = "".join(list(map(lambda x: alphabet[x], result)))
                        result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
                        print(result, len(text))
                        print("                -----", "".join([alphabet[i.item()] for i in target[0]]))
                    print("VALIDATE Loss:", sr, "")
        self.save(ctc_load=True)

    def load(self, ctc_load):
        self.encoder.load_state_dict(torch.load("trained_param/encoder_params"))
        self.decoder.load_state_dict(torch.load("trained_param/decoder_params"))

        if ctc_load:
            self.ctc_classifier.load_state_dict(torch.load("trained_param/CTCdecoder_params"))

    def save(self, ctc_load):
        torch.save(self.encoder.state_dict(), "trained_param/encoder_params")
        torch.save(self.decoder.state_dict(), "trained_param/decoder_params")
        if ctc_load:
            torch.save(self.ctc_classifier.state_dict(), "trained_param/CTCdecoder_params")
