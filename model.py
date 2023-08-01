import torch


class ASR(torch.nn.Module):
    def __init__(self):
        super(ASR, self).__init__()

        self.rnn = torch.nn.LSTM(1 * 13, 256, 3, dropout=0.2, bidirectional=True)
        self.linear = torch.nn.Linear(512, 34)

    def forward(self, x):
        x = x
        x, _ = self.rnn(x)
        output = torch.zeros((x.size()[0], x.size()[1], 34))
        for i in range(x.size()[0]):
            output[i] = torch.nn.functional.log_softmax(self.linear(x[i, :, :]), dim=-1)
        return output

    def train(self, epochesCount, dataset, validate=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0035)
        torch.autograd.set_detect_anomaly(True)
        for i in range(epochesCount):
            print("Epoch:", i + 1)
            if 0.0002 > self.epoch(dataset, optimizer):
                break
            if validate:
                self.validate_epoch(validate)

    def epoch(self, dataset, optimizer):
        it = iter(dataset)
        sr = 0
        CTC = torch.nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        for i in range(len(dataset)):
            data, target, input_lengths, target_lengths = next(it)
            target = target.cuda()
            output = self(data)

            ch = torch.exp(torch.squeeze(output))
            from utils import decode_result
            bib = decode_result(output[:, 0:1])

            optimizer.zero_grad()
            loss = CTC(output, target, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            optimizer.step()
            del data
            del target
            sr += float(loss)
            del loss

        print("Loss:", sr, "")
        return sr / len(dataset)

    def validate_epoch(self, val_data):
        it = iter(val_data)
        sr = 0
        with torch.no_grad():
            CTC = torch.nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
            for i in range(len(val_data)):
                data, target, input_lengths, target_lengths = next(it)
                target = target.cuda()
                output = self(data)

                loss = CTC(output, target, input_lengths, target_lengths)
                sr += float(loss)
                from utils import decode_result, alphabet
                bib = decode_result(output[:, 0:1])
                print("                -----", "".join([alphabet[i.item()] for i in target[0]]))
            print("VALIDATE Loss:", sr, "")


class EncoderNN(torch.nn.Module):
    def __init__(self):
        super(EncoderNN, self).__init__()
        # pyromidal LSTM
        self.lstm1 = torch.nn.LSTM(1 * 39, 256, 1, dropout=0.0, bidirectional=True, batch_first=True)
        self.lstm2 = torch.nn.LSTM(1024, 256, 1, dropout=0.0, bidirectional=True, batch_first=True)
        self.lstm3 = torch.nn.LSTM(1024, 256, 1, dropout=0.0, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(1024,512)

    def forward(self, x):
        x = x.transpose(0, 1)

        x, (h, _) = self.lstm1(x)
        x = x[:, :x.shape[1] - x.shape[1] % 2, :].reshape(-1, x.shape[1] // 2, x.shape[2] * 2) #pyramid transform

        x, _ = self.lstm2(x)
        x = x[:, :x.shape[1] - x.shape[1] % 2, :].reshape(-1, x.shape[1] // 2, x.shape[2] * 2)

        x, (h, _) = self.lstm3(x)
        x = x[:, :x.shape[1] - x.shape[1] % 2, :].reshape(-1, x.shape[1] // 2, x.shape[2] * 2)
        x = self.linear(x)
        h = h.reshape(h.shape[0] // 2,-1, h.shape[2] * 2)
        return x, h


class DecoderNN(torch.nn.Module):
    def __init__(self, hidden_size=512,rnn_layer_count = 1):
        super(DecoderNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer_count = rnn_layer_count
        # Attention
        self.W_decoder = torch.nn.Linear(rnn_layer_count * hidden_size, hidden_size)
        self.W_encoder = torch.nn.Linear(hidden_size, hidden_size)
        self.W_align = torch.nn.Parameter(torch.rand((hidden_size, 1)))

        self.embedding = torch.nn.Embedding(34, hidden_size)
        self.lstm = torch.nn.LSTM(34, hidden_size, rnn_layer_count, dropout=0.2)
        self.linear = torch.nn.Linear(2*hidden_size, 34)

    def forward(self, encoder_outputs, hidden_state, prev_output):
        batch_size = encoder_outputs.shape[0]

        output, hidden = self.lstm(prev_output.unsqueeze(dim=0),hidden_state)

        #Attention
        scores = torch.bmm(encoder_outputs,output.permute(1,2,0))  # seq*hid
        #scores = torch.bmm(scores, self.W_align.unsqueeze(0).repeat(batch_size, 1, 1))
        attent_weights = torch.nn.functional.softmax(scores, dim=1)  # B*S*1
        context_vector = torch.bmm(attent_weights.transpose(1, 2), encoder_outputs) # B*1*H

        features_concat = torch.cat([output.transpose(0,1),context_vector],dim=2)
        result = (self.linear(torch.squeeze(features_concat, dim=1)))

        return result, hidden
