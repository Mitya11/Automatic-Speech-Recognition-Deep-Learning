import torch


class pBLSTMLayer(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_dim):
        super(pBLSTMLayer, self).__init__()

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = torch.nn.LSTM(input_feature_dim * 2, hidden_dim, 1, bidirectional=True,
                                   dropout=0.0, batch_first=True)

    def forward(self, input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        input_x = input_x[:,:timestep - timestep%2,:]
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size, int(timestep / 2), feature_dim * 2)
        # Bidirectional RNN
        output, hidden = self.BLSTM(input_x)
        return output, hidden
class EncoderNN(torch.nn.Module):
    def __init__(self):
        super(EncoderNN, self).__init__()
        # pyromidal LSTM
        self.lstm1 = pBLSTMLayer(1 * 39, 256)
        self.lstm2 = pBLSTMLayer(512, 256)
        self.lstm3 = pBLSTMLayer(512, 256)

    def forward(self, x):
        x = x.contiguous().transpose(0, 1)

        x, h = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, h = self.lstm3(x)
        p = x[0].cpu().detach().numpy()

        return x,h
class Attention(torch.nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.phi = torch.nn.Linear(512, 512)
        self.psi = torch.nn.Linear(512, 512)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, decoder_state, listener_feature):
        comp_decoder_state = self.phi(decoder_state)
        comp_listener_feature = self.psi(listener_feature)

        energy = torch.bmm(comp_decoder_state, comp_listener_feature.transpose(1, 2)).squeeze(dim=1)
        attention_score = [self.softmax(energy)]
        o = attention_score[0].unsqueeze(2).repeat(1, 1, listener_feature.size(2))
        context = torch.sum(listener_feature * attention_score[0].unsqueeze(2).repeat(1, 1, listener_feature.size(2)),
                            dim=1)
        return attention_score, context
class DecoderNN(torch.nn.Module):
    def __init__(self, hidden_size=512,rnn_layer_count = 1):
        super(DecoderNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer_count = rnn_layer_count
        # Attention
        self.attention = Attention()
        self.rnn_layer = torch.nn.LSTM(548,512,1,batch_first=True)
        self.character_distribution = torch.nn.Linear(1024,36)

    def forward(self, listener_feature,rnn_input,hidden_state):

        rnn_output, hidden_state = self.rnn_layer(rnn_input, hidden_state)
        attention_score, context = self.attention(rnn_output, listener_feature)
        concat_feature = torch.cat([rnn_output.squeeze(dim=1), context], dim=-1)
        raw_pred = self.character_distribution(concat_feature)
        return raw_pred, hidden_state,context
