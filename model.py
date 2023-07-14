import torch


class ASR(torch.nn.Module):
    def __init__(self):
        super(ASR, self).__init__()

        self.rnn = torch.nn.LSTM(2 * 28, 58, 2, dropout=0.2, bidirectional=True)
        self.linear = torch.nn.Linear(116, 34)

    def forward(self, x):
        x = x.cuda()
        x, _ = self.rnn(x)
        output = torch.zeros((x.size()[0], x.size()[1], 34)).cuda()
        for i in range(x.size()[0]):
            output[i] = torch.nn.functional.log_softmax(self.linear(x[i, :, :]), dim=-1)
        return output

    def train(self, epochesCount, dataset,validate = None):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0006)
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
