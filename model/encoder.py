import torch


class pBLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = torch.nn.LSTM(input_dim * 2, hidden_dim, 1, bidirectional=True,
                                   dropout=0.0, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        length = x.size(1)
        hidden_dim = x.size(2)
        x = x[:, :length - length % 2, :]
        # Reduce time resolution
        x = x.contiguous().view(batch_size, int(length / 2), hidden_dim * 2)
        # Bidirectional RNN
        output, hidden = self.BLSTM(x)
        return output, hidden


class EncoderNN(torch.nn.Module):
    def __init__(self):
        super(EncoderNN, self).__init__()
        # pyromidal LSTM
        self.BLSTM1 = pBLSTM(1 * 90, 512)
        self.BLSTM2 = pBLSTM(1024, 512)
        self.BLSTM3 = pBLSTM(1024, 512)

    def forward(self, x):
        x = x.contiguous().transpose(0, 1)


        x, h = self.BLSTM1(x)
        x, _ = self.BLSTM2(x)
        x, h = self.BLSTM3(x)
        p = x[0].cpu().detach().numpy()

        return x, h
