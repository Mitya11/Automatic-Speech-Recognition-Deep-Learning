import torch


class ASR(torch.nn.Module):
    def __init__(self):
        super(ASR, self).__init__()

        self.rnn = torch.nn.LSTM(1*13, 256, 3, dropout=0.2, bidirectional=True)
        self.linear = torch.nn.Linear(512, 34)

    def forward(self, x):
        x = x
        x, _ = self.rnn(x)
        output = torch.zeros((x.size()[0], x.size()[1], 34))
        for i in range(x.size()[0]):
            output[i] = torch.nn.functional.log_softmax(self.linear(x[i, :, :]), dim=-1)
        return output

    def train(self, epochesCount, dataset,validate = None):
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
        CTC = torch.nn.CTCLoss(blank=0,reduction = 'sum',zero_infinity=True)
        for i in range(len(dataset)):
            data, target, input_lengths, target_lengths = next(it)
            target = target.cuda()
            output = self(data)

            ch = torch.exp(torch.squeeze(output))
            from utils import decode_result
            bib = decode_result(output[:,0:1])

            optimizer.zero_grad()
            loss = CTC(output,target, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            optimizer.step()
            del data
            del target
            sr += float(loss)
            del loss

        print("Loss:", sr,"")
        return sr / len(dataset)
    def validate_epoch(self,val_data):
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
                from utils import decode_result,alphabet
                bib = decode_result(output[:, 0:1])
                print("                -----","".join([alphabet[i.item()] for i in target[0]]))
            print("VALIDATE Loss:", sr, "")



class EncoderNN(torch.nn.Module):
    def __init__(self):
        super(EncoderNN, self).__init__()
        #pyromidal LSTM
        self.lstm1 = torch.nn.LSTM(1 * 18, 128, 1, dropout=0.2, bidirectional=True)
        self.linear1 = torch.nn.Linear(512,128)

        self.lstm2 = torch.nn.LSTM(128, 128, 1, dropout=0.2, bidirectional=True)
        self.linear2 = torch.nn.Linear(512,128)

        self.lstm3 = torch.nn.LSTM(128, 128, 1, dropout=0.2, bidirectional=True)
        self.linear3 = torch.nn.Linear(512,256)

    def forward(self, x):

        def pyromidal_transform(inputs):
            splitted = torch.split(inputs, 2)
            if splitted[-1].size()[0] != 2:
                splitted = splitted[:-1]
            output = []
            for i in range(len(splitted)):
                output.append(torch.cat(torch.split(splitted[i],1), dim=2)[0])

            return torch.stack(output)

        x = x
        x, _ = self.lstm1(x)
        x = torch.tanh(self.linear1(pyromidal_transform(x)))

        x, _ = self.lstm2(x)
        x = torch.tanh(self.linear2(pyromidal_transform(x)))

        x, (h,_) = self.lstm3(x)
        x = torch.tanh(self.linear3(pyromidal_transform(x)))
        h = pyromidal_transform(h)
        return x, h

class DecoderNN(torch.nn.Module):
    def __init__(self, hidden_size=256):
        super(DecoderNN, self).__init__()

        #Attention
        self.W_decoder = torch.nn.Linear(hidden_size, hidden_size)
        self.W_encoder = torch.nn.Linear(hidden_size, hidden_size)
        self.W_align = torch.nn.Parameter(torch.rand((hidden_size,1)))

        self.embedding = torch.nn.Embedding(34,hidden_size)
        self.lstm = torch.nn.LSTM(2*hidden_size,hidden_size)
        self.linear = torch.nn.Linear(hidden_size,34)
    def forward(self,encoder_outputs,hidden_state,prev_output):
        encoder_outputs = encoder_outputs.transpose(0,1)

        scores = torch.tanh(self.W_encoder(encoder_outputs) + self.W_decoder(hidden_state[0].transpose(0,1))) #seq*hid
        scores = torch.bmm(scores,self.W_align.unsqueeze(0).repeat(encoder_outputs.shape[0],1,1))

        attent_weights = torch.nn.functional.softmax(scores,dim=1) #B*S*1
        context_vector = torch.bmm(attent_weights.transpose(1,2),encoder_outputs) # B*1*H

        embedded = self.embedding(prev_output)
        decoder_input = torch.cat([torch.squeeze(context_vector,dim=1),embedded],dim=1).unsqueeze(dim=0) # 1*B*2H

        output,hidden = self.lstm(decoder_input,hidden_state) #1*B*H

        result = torch.nn.functional.softmax(self.linear(torch.squeeze(output,dim=0)),dim=1)

        return result, hidden
