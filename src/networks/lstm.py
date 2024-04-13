import torch.nn as nn
import torch
from base.base_net import BaseNet

# class LSTM_Net(BaseNet):
#     def __init__(self, input_size, rep_dim, num_layers):
#         super().__init__()
#         self.rep_dim = rep_dim
#         self.lin = nn.Linear(input_size,rep_dim, bias=False)
#         self.lstm = nn.LSTM(rep_dim, rep_dim, num_layers=num_layers, bias=False, batch_first=True)

#     def forward(self, x):
#         x = nn.functional.leaky_relu(self.lin(x))
#         out, _ = self.lstm(x)
#         return out
    

# class LSTM_Decoder(BaseNet):
#     def __init__(self, input_size, rep_dim, num_layers):
#         super().__init__()
#         self.rep_dim = rep_dim
#         self.lstm = nn.LSTM(rep_dim, rep_dim, num_layers=num_layers, bias=False, batch_first=True)
#         self.lin = nn.Linear(rep_dim, input_size, bias=False)

#     def forward(self, x):
#         # print(type(x))
#         x,_ = self.lstm(x)
#         return nn.functional.leaky_relu(self.lin(x))
    

# class LSTM_Autoencoder(BaseNet):
#     def __init__(self, input_size, rep_dim, num_layers):
#         super().__init__()
#         self.rep_dim = rep_dim
#         self.encoder = LSTM_Net(input_size, rep_dim, num_layers)
#         self.decoder = LSTM_Decoder(input_size, rep_dim, num_layers)
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


# if __name__=='__main__':
#     input_size = 10
#     rep_dim = 8
#     num_layer = 3
#     net = LSTM_Autoencoder(input_size, rep_dim, num_layer)
#     data = torch.rand([50, 40, input_size])
#     x = net(data)
# import logging
# class BaseNet(nn.Module):
#     """Base class for all neural networks."""

#     def __init__(self):
#         super().__init__()
#         self.logger = logging.getLogger(self.__class__.__name__)
#         self.rep_dim = None  # representation dimensionality, i.e. dim of the code layer or last layer

#     def forward(self, *input):
#         """
#         Forward pass logic
#         :return: Network output
#         """
#         raise NotImplementedError

#     def summary(self):
#         """Network summary."""
#         net_parameters = filter(lambda p: p.requires_grad, self.parameters())
#         params = sum([np.prod(p.size()) for p in net_parameters])
#         self.logger.info('Trainable parameters: {}'.format(params))
#         self.logger.info(self)

class LSTM_Net(BaseNet):
    def __init__(self, input_size, rep_dim, num_layers):
        super().__init__()
        self.rep_dim = rep_dim
        self.num_layers = num_layers
        # self.lin = nn.Linear(input_size,rep_dim, bias=False)
        self.lstm = nn.LSTM(input_size, rep_dim, num_layers=num_layers, bias=False, batch_first=True)

    def forward(self, x):
        # x = nn.functional.leaky_relu(self.lin(x))
        # print('input shape', x.shape)
        batch_size = x.shape[0]
        out, (h_n, c_n) = self.lstm(x)
        h_n = h_n.view(batch_size, self.num_layers*self.rep_dim)
        # print(self.num_layers)
        # print(self.rep_dim)
        # print('after lstm', out.shape)
        # print('hidden shape',h_n.shape)
        return h_n
    

class LSTM_Decoder(BaseNet):
    def __init__(self, input_size, rep_dim, num_layers, seq_len):
        super().__init__()
        self.rep_dim = rep_dim
        self.seq_len = seq_len
        self.lstm = nn.LSTM(rep_dim*num_layers, rep_dim, num_layers=num_layers, bias=False, batch_first=True)
        self.lin = nn.Linear(rep_dim, input_size, bias=False)

    def forward(self, x):
        # print(type(x))
        # print('decoder in', x.shape)
        x = x.unsqueeze(1)
        x = x.repeat(1, self.seq_len, 1)
        # print('decoder processed hidden', x.shape)
        x,(hidden, cell) = self.lstm(x)
        # print('decoder after lstm',x.shape)
        x = self.lin(x)
        # print('decoder after linear',x.shape)
        return x
    

class LSTM_Autoencoder(BaseNet):
    def __init__(self, input_size, rep_dim, num_layers, seq_len):
        super().__init__()
        self.rep_dim = rep_dim
        self.encoder = LSTM_Net(input_size, rep_dim, num_layers)
        self.decoder = LSTM_Decoder(input_size, rep_dim, num_layers, seq_len)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__=='__main__':
    input_size = 10
    rep_dim = 64
    num_layers = 2
    seq_len = 40
    net = LSTM_Autoencoder(input_size=8, rep_dim=64, num_layers=2, seq_len=100)
    data = torch.rand([50, seq_len, input_size])
    # print('raw shape', data.shape)
    x = net(data)
