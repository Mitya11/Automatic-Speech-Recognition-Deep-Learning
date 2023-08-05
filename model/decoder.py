import torch

class Attention(torch.nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.encode = torch.nn.Linear(512, 512)
        self.decode = torch.nn.Linear(512, 512)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, decoder_state, listener_feature):
        decoder_state = self.decode(decoder_state)
        listener_feature = self.encode(listener_feature)

        energy = torch.bmm(decoder_state, listener_feature.transpose(1, 2)).squeeze(dim=1)
        attention_score = [self.softmax(energy)]
        context = torch.sum(listener_feature * attention_score[0].unsqueeze(2).repeat(1, 1, listener_feature.size(2)),
                            dim=1)
        return attention_score, context


class DecoderNN(torch.nn.Module):
    def __init__(self, hidden_size=512, rnn_layer_count=1):
        super(DecoderNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer_count = rnn_layer_count
        # Attention
        self.attention = Attention()
        self.rnn_layer = torch.nn.LSTM(548, 512, 1, batch_first=True)
        self.character_distribution = torch.nn.Linear(1024, 36)

    def forward(self, listener_feature, rnn_input, hidden_state):
        rnn_output, hidden_state = self.rnn_layer(rnn_input, hidden_state)
        attention_score, context = self.attention(rnn_output, listener_feature)
        concat_feature = torch.cat([rnn_output.squeeze(dim=1), context], dim=-1)
        raw_pred = self.character_distribution(concat_feature)
        return raw_pred, hidden_state, context
