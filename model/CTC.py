import torch


class CTCdecoder(torch.nn.Module):
    def __init__(self):
        super(CTCdecoder, self).__init__()

        self.linear = torch.nn.Sequential(torch.nn.Linear(1024,1024),
                                          torch.nn.Dropout(0.0),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(1024,180)
                                          )

    def forward(self,x):

        sequence_pred = torch.nn.functional.log_softmax(self.linear(x),dim=-1)

        return sequence_pred