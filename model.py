import torch


class ASR(torch.nn.Module):
    def __init__(self):
        super(ASR, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (3, 8), padding=(1, 4)),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d((1, 2)),

            torch.nn.Conv2d(32, 64, (3, 8), padding=(1, 4)),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(64, 128, (3, 3), padding=(1, 0)),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d((1, 2)),

        )
        self.lstm = torch.nn.LSTM(5760, 128, 3, dropout=0.2)
        self.linear = torch.nn.Linear(128, 34)

    def forward(self, x):
        conv_output = torch.zeros((x.size()[0], x.size()[1], 5760))
        for i in range(x.size()[0]):
            out = self.conv(x[i])
            conv_output[i] = out.reshape((x.size()[1], 5760))
        x, _ = self.lstm(conv_output)
        output = torch.zeros((x.size()[0], x.size()[1], 34))
        for i in range(x.size()[0]):
            output[i] = torch.nn.functional.log_softmax(self.linear(x[i, :, :]), dim=-1)
        return output

    def train(self, epochesCount, dataset):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        torch.autograd.set_detect_anomaly(True)
        for i in range(epochesCount):
            print("Epoch:", i + 1)
            if 0.0002 > self.epoch(dataset, optimizer):
                break

    def epoch(self, dataset, optimizer):
        it = iter(dataset)
        sr = 0
        CTC = torch.nn.CTCLoss(reduction = 'sum',zero_infinity=True)
        for i in range(len(dataset)):
            data, target = next(it)
            output = self(data)

            ch = torch.exp(torch.squeeze(output))
            bib = (target.size(dim=1),) * output.size(dim=1)
            optimizer.zero_grad()
            loss = CTC(output,target, (output.size(dim=0),) * output.size(dim=1),
                                          (output.size(dim=0),) * output.size(dim=1))
            loss.backward()
            optimizer.step()
            sr += float(loss)
        print("Loss:", sr / len(data))
        return sr / len(dataset)