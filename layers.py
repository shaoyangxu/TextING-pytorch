import torch
import torch.nn as nn
from inits import glorot,xavier

class gru_unit(nn.Module):
    def __init__(self, output_dim, act, dropout_p):
        super(gru_unit,self).__init__()
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.act = act
        self.z0_weight = glorot([self.output_dim, self.output_dim]) # nn.Parameter(torch.randn(self.output_dim, self.output_dim))
        self.z1_weight = glorot([self.output_dim, self.output_dim])
        self.r0_weight = glorot([self.output_dim, self.output_dim])
        self.r1_weight = glorot([self.output_dim, self.output_dim])
        self.h0_weight = glorot([self.output_dim, self.output_dim])
        self.h1_weight = glorot([self.output_dim, self.output_dim])
        self.z0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.z1_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.r0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.r1_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.h0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.h1_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self,support, x, mask):
        support = self.dropout(support)
        a = torch.matmul(support, x)
        # updata gate
        z0 = torch.matmul(a, self.z0_weight) + self.z0_bias
        z1 = torch.matmul(x, self.z1_weight) + self.z1_bias
        z = torch.sigmoid(z0+z1)
        # reset gate
        r0 = torch.matmul(a, self.r0_weight) + self.r0_bias
        r1 = torch.matmul(x, self.r1_weight) + self.r1_bias
        r = torch.sigmoid(r0+r1)
        # update embeddings
        h0 = torch.matmul(a, self.h0_weight) + self.h0_bias
        h1 = torch.matmul(r*x, self.h1_weight) + self.h1_bias
        h = self.act(mask * (h0 + h1))
        return h*z + x*(1-z)


class GraphLayer(nn.Module):
    """Graph layer."""
    def __init__(self, args,
                      input_dim,
                      output_dim,
                      act=nn.Tanh(),
                      dropout_p = 0.,
                      gru_step = 2):
        super(GraphLayer, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.gru_step = gru_step
        self.gru_unit = gru_unit(output_dim = self.output_dim,
                                 act = self.act,
                                 dropout_p = self.dropout_p)
        # self.dropout
        self.encode_weight = glorot([self.input_dim, self.output_dim])
        self.encode_bias = nn.Parameter(torch.zeros(self.output_dim))


    def forward(self, feature, support, mask):
        feature = self.dropout(feature)
        # encode inputs
        encoded_feature = torch.matmul(feature, self.encode_weight) + self.encode_bias
        output = mask * self.act(encoded_feature)
        # convolve
        for _ in range(self.gru_step):
            output = self.gru_unit(support, output, mask)
        return output

class ReadoutLayer(nn.Module):
    """Graph Readout Layer."""
    def __init__(self, args,
                 input_dim,
                 output_dim,
                 act=nn.ReLU(),
                 dropout_p=0.):
        super(ReadoutLayer, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.att_weight = glorot([self.input_dim, 1])
        self.emb_weight = glorot([self.input_dim, self.input_dim])
        self.mlp_weight = glorot([self.input_dim, self.output_dim])
        self.att_bias = nn.Parameter(torch.zeros(1))
        self.emb_bias = nn.Parameter(torch.zeros(self.input_dim))
        self.mlp_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self,x,_,mask):  # _ not used
        # soft attention
        att = torch.sigmoid(torch.matmul(x, self.att_weight)+self.att_bias)
        emb = self.act(torch.matmul(x, self.emb_weight)+self.emb_bias)
        N = torch.sum(mask, dim=1)
        M = (mask - 1) * 1e9
        # graph summation
        g = mask * att * emb
        g = torch.sum(g, dim=1)/N + torch.max(g+M,dim=1)[0]
        g = self.dropout(g)
        # classification
        output = torch.matmul(g,self.mlp_weight)+self.mlp_bias
        return output
