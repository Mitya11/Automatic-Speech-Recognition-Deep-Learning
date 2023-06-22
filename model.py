import torch


class ASR(torch.nn.Module):
    def __init__(self):
        super(ASR, self).__init__()

        self.rnn = torch.nn.LSTM(9*18, 1024, 3, dropout=0.2,bidirectional=True)
        self.linear = torch.nn.Linear(2048, 34)

    def forward(self, x):
        x = x.cuda()
        x, _ = self.rnn(x)
        output = torch.zeros((x.size()[0], x.size()[1], 34)).cuda()
        for i in range(x.size()[0]):
            output[i] = torch.nn.functional.log_softmax(self.linear(x[i, :, :]), dim=-1)
        return output

    def train(self, epochesCount, dataset):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0007)
        torch.autograd.set_detect_anomaly(True)
        for i in range(epochesCount):
            print("Epoch:", i + 1)
            if 0.0002 > self.epoch(dataset, optimizer):
                break

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
            bib = decode_result(output[:,-2:-1])

            optimizer.zero_grad()
            loss = CTC(output,target, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            optimizer.step()
            del data
            del target
            sr += float(loss)
            del loss

        print("Loss:", sr)
        return sr / len(dataset)