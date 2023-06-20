import torch


class ASR(torch.nn.Module):
    def __init__(self):
        super(ASR, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (4, 8), padding=(1, 4)),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d((1, 2)),

            torch.nn.Conv2d(32, 64, (5, 8), padding=(0, 4)),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d((1, 2)),

            torch.nn.Conv2d(64, 128, (3, 3), padding=(1, 0)),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d((1, 2)),

        )
        self.lstm = torch.nn.RNN(2688, 512, 3, dropout=0.2,nonlinearity="relu")
        self.linear = torch.nn.Linear(512, 34)

    def forward(self, x):
        x = x.cuda()
        conv_output = torch.zeros((x.size()[0], x.size()[1], 2688)).cuda()
        for i in range(x.size()[0]):
            out = self.conv(x[i])
            conv_output[i] = out.reshape((x.size()[1], 2688))
        x, _ = self.lstm(conv_output)
        output = torch.zeros((x.size()[0], x.size()[1], 34)).cuda()
        for i in range(x.size()[0]):
            output[i] = torch.nn.functional.log_softmax(self.linear(x[i, :, :]), dim=-1)
        return output

    def train(self, epochesCount, dataset):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00002)
        #torch.autograd.set_detect_anomaly(True)
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
            target = target.cuda()
            output = self(data)

            ch = torch.exp(torch.squeeze(output))
            from utils import decode_result
            bib = decode_result(output[:,5])

            optimizer.zero_grad()
            loss = CTC(output,target, (output.size(dim=0)-1,) * output.size(dim=1),
                                          (target.size(dim=1),) * output.size(dim=1))
            loss.backward()
            optimizer.step()
            del data
            del target
            sr += float(loss)
            del loss

        print("Loss:", sr / 4)
        return sr / len(dataset)