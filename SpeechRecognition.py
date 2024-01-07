import torch
from model.CTC import CTCdecoder
from model.decoder import DecoderNN
from model.encoder import EncoderNN
import numpy as np

import random
from matplotlib import pyplot as plt
from utils import beam_search

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
        while prev_output != 178 and len(result) < 80:
            # teacher forcing
            output, hidden, context,attention_score = self.decoder(encoder_output, prev_output, hidden)
            output = beam_search(self, output[0:1], [encoder_output[0:1], prev_output[0:1], hidden],
                                       3, 3)

            result.append(output.item())

            prev_output = output.unsqueeze(0)
            attention_matrix.append(attention_score[0][0].cpu().detach().numpy())
        plt.imshow(attention_matrix)
        plt.show()
        return result

    def train(self, epochesCount, train_data, val_data=None):
        self.ctc_classifier = CTCdecoder().to(self.device)
        self.load(ctc_load=True)
        optimizer = torch.optim.Adam(list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.ctc_classifier.parameters()),lr=0.00045)
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(epochesCount):
            print("Epoch:", epoch + 1)
            it = iter(train_data)
            criterion_cross = torch.nn.CrossEntropyLoss(ignore_index=179)
            criterion_ctc = torch.nn.CTCLoss(blank=179, reduction='sum', zero_infinity=True)
            sr = 0
            for i in range(len(train_data)):
                inputs, target, input_lengths, target_lengths = next(it)
                inputs = inputs.to(self.device).squeeze(dim=2)
                print("seq len: ",inputs.shape)
                target = target.to(self.device)

                loss = torch.tensor(0, dtype=torch.float64, device=self.device)
                optimizer.zero_grad()

                encoder_output, prev_hidden = self.encoder(inputs)
                # CTC-based model
                ctc_output = self.ctc_classifier(encoder_output)
                input_lengths = input_lengths // 8
                loss = criterion_ctc(ctc_output.transpose(0, 1), target, input_lengths, target_lengths) * 0.2

                # Attention-based model
                prev_output = torch.zeros((encoder_output.shape[0])).to(self.device,torch.long)
                #rnn_input = torch.cat([output.unsqueeze(dim=1), encoder_output[:, 0:1, :]], dim=-1)

                hidden = [torch.zeros((1,encoder_output.shape[0],512), device=self.device)]*2
                result = []
                attention_matrix = []
                for j in range(target.size()[1]):
                    # teacher forcing
                    output, hidden, context , attention_score = self.decoder(encoder_output, prev_output, hidden)
                    loss += criterion_cross(output, target[:, j]).nan_to_num(0)
                    result.append(torch.argmax(output[0:1], dim=1).item())
                    if random.randint(1, 100) < 80:
                        output = target[:, j]
                    else:
                        output = output.max(dim=1)[1]

                    prev_output = output
                    attention_matrix.append(attention_score[0][0])

                #plt.imshow(torch.stack(attention_matrix).cpu().detach())
                #plt.show()
                from utils import transcript
                import itertools
                text = "".join(list(map(lambda x: transcript[x], result)))
                result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
                print(result, len(text), "Completed:", round(i / len(train_data) * 100, 4), "%")
                print("           -----", "".join([transcript[i.item()] for i in target[0]]))

                loss.backward()
                print(torch.nn.utils.clip_grad_norm_(list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.ctc_classifier.parameters()), 1))
                # print(self.encoder.lstm1.all_weights[0][0].grad)
                optimizer.step()

                sr += float(loss)
            print("LOSS:", sr/len(train_data))
            if val_data:
                it = iter(val_data)
                sr = 0
                NLL = torch.nn.CrossEntropyLoss(ignore_index=0)

                with torch.no_grad():
                    for i in range(len(val_data)):
                        inputs, target, _, _ = next(it)
                        inputs = inputs.to(self.device).squeeze(dim=2)
                        target = target.to(self.device)

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
                            result.append(output_first.item())

                            output = output.max(dim=1)[1]
                            prev_output = output

                        from utils import transcript
                        import itertools
                        text = "".join(list(map(lambda x: transcript[x], result)))
                        result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
                        print(result, len(text))
                        print("                -----", "".join([transcript[i.item()] for i in target[0]]))
                    print("VALIDATE Loss:", sr/len(val_data), "")
        self.save(ctc_load=True)

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
