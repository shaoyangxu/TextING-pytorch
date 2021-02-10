import torch
import torch.nn as nn
from layers import *
class GNN(nn.Module):
    def __init__(self, args, input_dim, output_dim, hidden_dim, gru_step, dropout_p):
        super(GNN,self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.gru_step = gru_step
        self.GraphLayer = GraphLayer(
            args = args,
            input_dim = self.input_dim,
            output_dim = self.hidden_dim,
            act = torch.nn.Tanh(),
            dropout_p = self.dropout_p,
            gru_step = self.gru_step
        )
        self.ReadoutLayer = ReadoutLayer(
            args=args,
            input_dim = self.hidden_dim,
            output_dim = self.output_dim,
            act = torch.nn.Tanh(),
            dropout_p = self.dropout_p
        )
        self.layers = [self.GraphLayer, self.ReadoutLayer]


    def forward(self, feature, support, mask):
        activations = [feature]
        for layer in self.layers:
            hidden = layer(activations[-1], support, mask)
            activations.append(hidden)
        embeddings = activations[-2]
        outputs = activations[-1]
        return outputs,embeddings

