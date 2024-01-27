import torch

class Attention(torch.nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.encode = torch.nn.Linear(1024, 512)
        self.decode = torch.nn.Linear(512, 512,bias=True)
        self.attent_weights = torch.nn.Linear(512,1,bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, decoder_state, listener_feature):
        con_decoder_state = self.decode(decoder_state[0].transpose(0,1))
        con_listener_feature = self.encode(listener_feature)


        energy = self.attent_weights(torch.nn.functional.tanh(con_decoder_state+con_listener_feature)).squeeze(-1)
        attention_score = [self.softmax(energy)]
        l = attention_score[0][0]
        context = torch.sum(listener_feature * attention_score[0].unsqueeze(2).repeat(1, 1, listener_feature.size(2)),
                            dim=1)
        return attention_score, context


class DecoderNN(torch.nn.Module):
    def __init__(self, hidden_size=512, rnn_layer_count=1):
        super(DecoderNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer_count = rnn_layer_count
        self.character_emb = torch.nn.Embedding(35,128)

        # Attention
        self.attention = Attention()
        self.rnn_layer = torch.nn.LSTM(1024+128, 512, 1, batch_first=True)
        self.character_distribution = torch.nn.Sequential(torch.nn.Linear(1536,  768),
                                                          torch.nn.Dropout(0.0),
                                                          torch.nn.ReLU(),
                                                          torch.nn.Linear(768,34))

    def forward(self, listener_feature, prev_output, hidden_state):
        prev_output = torch.nn.functional.dropout(self.character_emb(prev_output),0.0) # delete dropout after
        attention_score, context = self.attention(hidden_state, listener_feature)
        rnn_input = torch.cat([prev_output, context], dim=-1).unsqueeze(dim=1)


        rnn_output, hidden_state = self.rnn_layer(rnn_input,hidden_state)
        #p = rnn_output[0].cpu().detach().numpy()

        concat_feature = torch.cat([hidden_state[0].squeeze(dim=0), context], dim=-1)
        raw_pred = self.character_distribution(concat_feature)
        return raw_pred, hidden_state, context,attention_score
