import torch
from model.CTC import CTCdecoder
from model.decoder import DecoderNN
from model.encoder import EncoderNN
import numpy as np
import soundfile as sf
import scipy
import random
from matplotlib import pyplot as plt
from utils import beam_search,get_features
from torch_audiomentations  import *

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

    def __call__(self, sequence):

        encoder_output, prev_hidden = self.encoder(sequence)

        result = []
        prev_output = torch.zeros((encoder_output.shape[0])).to(self.device,torch.long)
        # rnn_input = torch.cat([output.unsqueeze(dim=1), encoder_output[:, 0:1, :]], dim=-1)

        hidden = [torch.zeros((1, encoder_output.shape[0], 512), device=self.device)] * 2
        result = []
        attention_matrix = []
        while prev_output != 33 and len(result) < 80:
            # teacher forcing
            output, hidden, context,attention_score = self.decoder(encoder_output, prev_output, hidden)
            output = beam_search(self, output[0:1], [encoder_output[0:1], prev_output[0:1], hidden],
                                       3, 3)

            result.append(output.item()+1)

            prev_output = output.unsqueeze(0)
            attention_matrix.append(attention_score[0][0].cpu().detach().numpy())
        plt.imshow(attention_matrix)
        plt.show()
        return result

    def train(self, epochesCount, train_data, val_data=None):
        self.ctc_classifier = CTCdecoder().to(self.device)
        self.load(ctc_load=True)
        optimizer = torch.optim.Adam(list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.ctc_classifier.parameters()),lr=0.00007)
        torch.autograd.set_detect_anomaly(True)

        transforms = [PitchShift(-2, 2, p=0.85, sample_rate=16000,mode="per_example",p_mode="per_example"),
                      AddBackgroundNoise(
                          "C:/Users/mitya/PycharmProjects/Automatic-Speech-Recognition-Deep-Learning/augmentation/", 13,
                          21, sample_rate=16000, p=0.85,mode="per_example",p_mode="per_example"),
                      #Gain(-10, 10, p=0.8,mode="per_example",p_mode="per_example"),
                      PolarityInversion(p=0.85,mode="per_example",p_mode="per_example")]
        impact, freq = sf.read("C:/Users/mitya/PycharmProjects/Automatic-Speech-Recognition-Deep-Learning/S1R1_sweep4000.wav", dtype='float32')

        for epoch in range(epochesCount):
            print("Epoch:", epoch + 1)
            it = iter(train_data)
            criterion_cross = torch.nn.CrossEntropyLoss(ignore_index=34,label_smoothing=0.06)
            criterion_ctc = torch.nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
            sr = 0
            for i in range(len(train_data)):
                inputs, target, input_lengths, target_lengths = next(it)

                for j in range(inputs.shape[1]):
                    if random.randint(1, 100) < 100:
                        inputs[:,j] = inputs[:,j]*0.97 + torch.tensor(scipy.signal.convolve(inputs[:,j],impact,mode="same")*0.03)

                inputs = inputs.to(self.device).unsqueeze(1).transpose(0, 2)

                try:
                    if transforms:
                        for transform in transforms:
                            inputs = transform(inputs)
                except:
                    pass
                sample_rate = random.randint(16000, 17200)

                inputs = inputs.cpu().squeeze(1).transpose(0,1)
                inputs,input_lengths = get_features(inputs,sample_rate)
                inputs = inputs.to(self.device).squeeze(dim=2).transpose(0,1)
                print("seq len: ",inputs.shape)
                target = target.to(self.device)

                loss = torch.tensor(0, dtype=torch.float64, device=self.device)
                optimizer.zero_grad()

                encoder_output, prev_hidden = self.encoder(inputs)
                p = encoder_output.cpu().detach().numpy()
                # CTC-based model
                ctc_output = self.ctc_classifier(encoder_output)
                input_lengths = input_lengths // 8
                loss = criterion_ctc(ctc_output.transpose(0, 1), target, input_lengths, target_lengths) * 0.005

                # Attention-based model
                prev_output = torch.zeros((encoder_output.shape[0])).to(self.device,torch.long)
                #rnn_input = torch.cat([output.unsqueeze(dim=1), encoder_output[:, 0:1, :]], dim=-1)

                hidden = [torch.zeros((1,encoder_output.shape[0],512), device=self.device)]*2
                result = []
                attention_matrix = []
                target = target-1
                for j in range(target.size()[1]):
                    # teacher forcing
                    output, hidden, context , attention_score = self.decoder(encoder_output, prev_output, hidden)
                    loss += criterion_cross(output, target[:, j]).nan_to_num(0)
                    result.append(torch.argmax(output[0:1], dim=1).item()+1)
                    if random.randint(1, 100) < 40:
                        output = target[:, j]
                    else:
                        output = output.max(dim=1)[1]

                    prev_output = output
                    attention_matrix.append(attention_score[0][0])

                #plt.imshow(torch.stack(attention_matrix).cpu().detach())
                #plt.show()
                from utils import transcript,alphabet
                import itertools
                text = "".join(list(map(lambda x: alphabet[x], result)))
                result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
                print(result, len(text), "Completed:", round(i / len(train_data) * 100, 4), "%")
                print("           -----", "".join([alphabet[i.item()+1] for i in target[0]]))

                try:
                    loss.backward()
                except:
                    print("ERROR! Lose is :",loss)
                    print("ERROR! Tens is :", inputs)
                    continue
                #print(torch.nn.utils.clip_grad_norm_(list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.ctc_classifier.parameters()), 1))
                # print(self.encoder.lstm1.all_weights[0][0].grad)
                optimizer.step()

                sr += float(loss)
            print("LOSS:", sr / len(train_data))
            final_loss = sr/len(train_data)
            if val_data:
                it = iter(val_data)
                sr = 0
                cer_loss = 0
                NLL = torch.nn.CrossEntropyLoss(ignore_index=0)
                with torch.no_grad():
                    for i in range(len(val_data)):
                        inputs, target, _, _ = next(it)
                        inputs, input_lengths = get_features(inputs, 16000)
                        inputs = inputs.to(self.device).squeeze(dim=2).transpose(0, 1)
                        target = target.to(self.device)-1

                        encoder_output, prev_hidden = self.encoder(inputs)

                        prev_output = torch.zeros((encoder_output.shape[0])).to(self.device,torch.long)
                        # rnn_input = torch.cat([output.unsqueeze(dim=1), encoder_output[:, 0:1, :]], dim=-1)

                        hidden = [torch.zeros((1, encoder_output.shape[0], 512), device=self.device)] * 2
                        result = []
                        for j in range(target.size()[1]):
                            # teacher forcing
                            output, hidden, context,_ = self.decoder(encoder_output, prev_output, hidden)
                            sr += criterion_cross(output, target[:, j]).nan_to_num(0)
                            output_first = beam_search(self, output[0:1], [encoder_output[0:1], prev_output[0:1], hidden],
                                                 3, 3)
                            result.append(output_first.item()+1)

                            output = output.max(dim=1)[1]
                            prev_output = output

                        from utils import alphabet
                        import itertools
                        text = "".join(list(map(lambda x: alphabet[x], result)))
                        result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
                        true_pred = "".join([alphabet[i.item()+1] for i in target[0]])

                        print(result, len(text))
                        print("                -----", true_pred)

                        import torchmetrics
                        cer = torchmetrics.CharErrorRate()

                        cer_loss += cer(result, true_pred)
                    print("VALIDATE Loss:", sr / len(val_data), "")
                    print("VALIDATE CER:", cer_loss / len(val_data), "")
        self.save(ctc_load=True)
        return final_loss
    def load(self, ctc_load):
        self.encoder.load_state_dict(torch.load("C:/Users/mitya/PycharmProjects/Automatic-Speech-Recognition-Deep-Learning/trained_param/encoder_params"))
        self.decoder.load_state_dict(torch.load("C:/Users/mitya/PycharmProjects/Automatic-Speech-Recognition-Deep-Learning/trained_param/decoder_params"))

        if ctc_load:
            self.ctc_classifier.load_state_dict(torch.load("C:/Users/mitya/PycharmProjects/Automatic-Speech-Recognition-Deep-Learning/trained_param/CTCdecoder_params"))

    def save(self, ctc_load):
        torch.save(self.encoder.state_dict(), "C:/Users/mitya/PycharmProjects/Automatic-Speech-Recognition-Deep-Learning/trained_param/encoder_params")
        torch.save(self.decoder.state_dict(), "C:/Users/mitya/PycharmProjects/Automatic-Speech-Recognition-Deep-Learning/trained_param/decoder_params")
        if ctc_load:
            torch.save(self.ctc_classifier.state_dict(), "C:/Users/mitya/PycharmProjects/Automatic-Speech-Recognition-Deep-Learning/trained_param/CTCdecoder_params")
