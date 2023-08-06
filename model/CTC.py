import torch


class CTCdecoder(torch.nn.Module):
    def __init__(self):
        super(CTCdecoder, self).__init__()

        self.linear = torch.nn.Linear(512,36)

    def forward(self,x):

        sequence_pred = torch.nn.functional.log_softmax(self.linear(x),dim=-1)

        return sequence_pred