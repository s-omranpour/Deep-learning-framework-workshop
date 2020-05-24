import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_layers, latent_dim):
        super(Encoder, self).__init__()
        
        layers = []
        for i in range(len(hidden_layers)):
            in_s = input_size if i==0 else hidden_layers[i-1]
            layers += [nn.Linear(in_s, hidden_layers[i])]
            layers += [nn.BatchNorm1d(hidden_layers[i])]
            layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_layers[-1], latent_dim)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_layers, latent_dim):
        super(Decoder, self).__init__()
        
        layers = []
        for i in range(len(hidden_layers)):
            in_s = latent_dim if i==0 else hidden_layers[i-1]
            layers += [nn.Linear(in_s, hidden_layers[i])]
            layers += [nn.BatchNorm1d(hidden_layers[i])]
            layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_layers[-1], output_size)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    

class AE(nn.Module):
    def __init__(self, input_size, hidden_layers, latent_dim):
        super(AE, self).__init__()
        
        self.encoder = Encoder(input_size, hidden_layers, latent_dim)
        self.decoder = Decoder(input_size, hidden_layers, latent_dim)
        

    def encode(self,x):
        return self.encoder(x)
    
    def decode(self,x):
        return self.decoder(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))