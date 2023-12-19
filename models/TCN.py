from torch import nn
from layers.tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, layers, kernel_size, pre_len, dropout=0):
        super(TCN, self).__init__()
        self.pre_len = pre_len
        self.output_size = output_size
        self.tcn = TemporalConvNet(input_size, hidden_size, layers, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size*pre_len)
        self.out_linear = nn.Linear(pre_len, pre_len)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # seq_last = x[:, -1:, :].detach()
        # x = x - seq_last
        x = x.transpose(1, 2)
        y1 = self.tcn(x)
        # y1 = self.out_linear(y1)
        # output = y1.permute(0,2,1)[:, -self.pre_len:, :]

        y1 = y1[:, :, -1]
        output = self.linear(y1)  # B, T*C, N, 1
        output = output.view(*output.shape[:-1], self.pre_len, self.output_size)
        return output