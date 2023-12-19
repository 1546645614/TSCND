import math
import torch
from torch import nn
from layers.TSCNCell import TSCNCell, TSCNCellV2, TSCNCellOrigin

# TSCND implementation based on conv1d provided by pytorch
class TSCND(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, in_len, pre_len, k, Ad, dropout=0):
        super(TSCND, self).__init__()
        self.pre_len = pre_len
        self.layers = int(math.log2(in_len))+1 if math.log2(in_len)-int(math.log2(in_len)) > 0 else int(math.log2(in_len))
        self.input_len = 2**self.layers
        self.output_size = output_size
        self.Ad = Ad
        self.embedding = nn.Linear(input_size, hidden_size)
        self.TSCNlayers = nn.ModuleList()
        for i in range(self.layers):
            self.TSCNlayers.append(TSCNCellV2(hidden_size, k, dropout=dropout))
        self.DimDecoding = nn.Linear(hidden_size, output_size)
        self.TemporalDecoding = nn.Linear(self.input_len, pre_len)

        if self.Ad == 'RevIN':
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_size))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_size))

    def forward(self, x):
        ### Methods for combating distribution shift ###
        if self.Ad == 'RevIN':
            means = x.mean(1, keepdim=True).detach()
            # mean
            x = x - means
            # var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            x = x * self.affine_weight + self.affine_bias
        elif self.Ad == 'DCM':
            seq_last = x[:, -1:, :].detach()
            x = torch.diff(x, prepend=x[:, 0:1, :], n=1, dim=1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = x - seq_last

        x1 = self.embedding(x) #Embedding

        padding = torch.zeros(x1.shape[0], self.input_len-x1.shape[1], x1.shape[2]).cuda()
        x1 = torch.cat((padding, x1), dim=1) #Padding
        x1 = x1.unsqueeze(dim=1)
        x1 = x1.permute(0,1,3,2)
        ## mutli-layer SCN ##
        for TSCNlayer in self.TSCNlayers:
            x1 = TSCNlayer(x1)
        x1 = x1.squeeze(3)
        output1 = self.DimDecoding(x1) #Decoding variable dimensions
        output1 = self.TemporalDecoding(output1.permute(0, 2, 1)) #Sequence dimension decoding
        output = output1.permute(0, 2, 1)
        if self.Ad == 'RevIN':
            output = output - self.affine_bias
            output = output / (self.affine_weight + 1e-10)
            output = output * stdev
            output = output + means
        else:
            output = output+seq_last
        return output

class TSCNDOrigin(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, in_len, pre_len, k, Ad, dropout=0):
        super(TSCNDOrigin, self).__init__()
        self.pre_len = pre_len
        self.layers = int(math.log2(in_len))+1 if math.log2(in_len)-int(math.log2(in_len)) > 0 else int(math.log2(in_len))
        self.input_len = 2**self.layers
        self.output_size = output_size
        self.Ad = Ad
        self.embedding = nn.Linear(input_size, hidden_size)
        self.TSCNlayers = nn.ModuleList()
        for i in range(self.layers):
            self.TSCNlayers.append(TSCNCellOrigin(hidden_size, k, dropout=dropout))
        self.DimDecoding = nn.Linear(hidden_size, output_size)
        self.TemporalDecoding = nn.Linear(self.input_len, pre_len)

        if self.Ad == 'RevIN':
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_size))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_size))

    def forward(self, x):
        ### Methods for combating distribution shift ###
        if self.Ad == 'RevIN':
            means = x.mean(1, keepdim=True).detach()
            # mean
            x = x - means
            # var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            x = x * self.affine_weight + self.affine_bias
        elif self.Ad == 'DCM':
            seq_last = x[:, -1:, :].detach()
            x = torch.diff(x, prepend=x[:, 0:1, :], n=1, dim=1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = x - seq_last

        x1 = self.embedding(x) #Embedding

        padding = torch.zeros(x1.shape[0], self.input_len-x1.shape[1], x1.shape[2]).cuda()
        x1 = torch.cat((padding, x1), dim=1) #Padding
        x1 = x1.unsqueeze(dim=2)
        ## mutli-layer SCN ##
        for TSCNlayer in self.TSCNlayers:
            x1 = TSCNlayer(x1)
        x1 = x1.squeeze(1)
        output1 = self.DimDecoding(x1) #Decoding variable dimensions
        output1 = self.TemporalDecoding(output1.permute(0, 2, 1)) #Sequence dimension decoding
        output = output1.permute(0, 2, 1)
        if self.Ad == 'RevIN':
            output = output - self.affine_bias
            output = output / (self.affine_weight + 1e-10)
            output = output * stdev
            output = output + means
        else:
            output = output+seq_last
        return output





