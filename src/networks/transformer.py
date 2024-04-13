import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.modules):
    def __init__(self, dim_input, dim_output, dim_k, dim_model, n_heads, n_layers, seq_len=100, dropout=0.0):
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.n_layers = n_layers

        # Define all the layers, note that there is no bias term in any layers.
        self.input = nn.Linear(dim_input, dim_model, bias=False)
        self.attentions = nn.ModuleList([nn.MultiheadAttention(dim_model, n_heads, kdim=dim_k, vdim=dim_model,
                                                                dropout=dropout, bias=False, batch_first=True) for _ in range(n_layers)])
        self.ff = nn.Module([nn.Linear(dim_model, dim_model, bias=False) for _ in range(2*n_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim_model, elementwise_affine=True, bias=False) for _ in range(2*n_layers)])
        self.out = nn.Linear(dim_model, dim_output, bias=False)
        self.activation = F.relu

    def position_encoding(self, x):
        '''
        Add position embedding to the features out of the input layer.
        Parameters: 
        x: Signal with shape (batch, seq_len, dim_model)
        Return:
        x+pe: Features with position embedding
        '''
        pe = torch.zeros(self.seq_len, self.dim_model)
        for pos in range(self.seq_len):
            for i in range(int(self.dim_model)):
                if i%2==0:
                    pe[pos, i] = math.sin(pos/10000**(2*i/self.dim_model))
                else:
                    pe[pos, i] = math.cos(pos/10000**(2*i/self.dim_model))

        return x + pe
    
    def forward(self, x):
        x = self.input(x)
        x = self.position_encoding(x)
        for i in range(self.n_layers):
            # Attention
            residual = x
            x = self.attentions[i](x)
            x = x + residual
            x = self.layer_norms[2*i](x)
            # Feedforward
            residual = x
            x = self.activation(self.ff[2*i](x))
            x = self.ff[2*i+1](x)
            x = x + residual
            x = self.layer_norms[2*i+1](x)
        return self.out(x)



class Trasnformer_Decoder(nn.modules):
        pass