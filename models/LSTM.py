from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, layers, kernel_size, pre_len, dropout=0):
        super(LSTM, self).__init__()
        self.pre_len = pre_len
        self.output_size = output_size
        self.gru = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size*pre_len)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.gru(x)[1][0]
        x = x[0, :, :]
        output = self.linear(x)  # B, T*C, N, 1
        output = output.view(*output.shape[:-1], self.pre_len, self.output_size)
        return output