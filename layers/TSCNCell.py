import torch
from torch import nn

class TSCNCell(nn.Module):
    def __init__(self, hidden_channels, k, dropout=0.3):
        super(TSCNCell, self).__init__()

        self.hidden_channels = hidden_channels
        self.dropout = torch.nn.Dropout(p=dropout)
        self.in1 = torch.nn.Linear(2*hidden_channels, 2*hidden_channels)
        self.convList = nn.ModuleList()
        self.k = k
        for i in range(k):
            self.convList.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=k,stride=k))
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, x):
        res = x
        outElements = []
        subSequenceLength = x.shape[1]
        x = x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3])
        for conv in self.convList:
            outElements.append(conv(x.permute(0,2,1)).permute(0,2,1))
        outElements = torch.stack(outElements, dim=1)
        out = outElements.reshape(outElements.shape[0]//subSequenceLength,-1, outElements.shape[2],outElements.shape[3])
        out = self.dropout(out)
        out = out.relu()
        res = res.reshape(out.shape[0],out.shape[2], out.shape[1],out.shape[3])
        out = (out.permute(0,2,1,3)+res).permute(0,2,1,3)
        # out = self.act(out)
        return out

class TSCNCellV2(nn.Module):
    def __init__(self, hidden_channels, k, dropout=0.3):
        super(TSCNCellV2, self).__init__()

        self.hidden_channels = hidden_channels
        self.dropout = torch.nn.Dropout(p=dropout)
        self.in1 = torch.nn.Linear(2*hidden_channels, 2*hidden_channels)
        self.convList = nn.ModuleList()
        self.k = k
        self.conv = nn.Conv1d(hidden_channels, hidden_channels*k, kernel_size=k,stride=k)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, x):
        res = x
        subSequenceLength = x.shape[1]
        x = x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3])
        out= self.conv(x)
        out= out.reshape(out.shape[0]//subSequenceLength, subSequenceLength*self.k, out.shape[1]//self.k,out.shape[2])
        out = self.dropout(out)
        out = out.relu()
        res = res.reshape(out.shape[0],out.shape[2], out.shape[3],out.shape[1]).permute(0,3,1,2)
        out = out+res
        # out = self.act(out)
        return out

class TSCNCellOrigin(nn.Module):
    def __init__(self, hidden_channels,k, dropout=0.3):
        super(TSCNCellOrigin, self).__init__()
        self.k = k
        self.hidden_channels = hidden_channels
        self.dropout = torch.nn.Dropout(p=dropout)
        self.conv = torch.nn.Linear(self.k*hidden_channels, self.k*hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout)


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, x):

        ## We stack elements in the hidden dimension and use fully connected layers to implement convolution operations ##
        res = x     #[batch, subsequenceNums, subsequenceLength, hid_dim]
        stackX = []
        for i in range(self.k-1, -1, -1):
            stackX.append(x[:, i::self.k, :])
        out = torch.cat(stackX, dim=-1)
        out = self.conv(out)
        ####
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2]*self.k, out.shape[3]//self.k)
        out = self.dropout(out)
        out = out.relu()
        res = res.reshape(res.shape[0], res.shape[1]//self.k, res.shape[2]*self.k, res.shape[3])
        out = out+res
        return out

