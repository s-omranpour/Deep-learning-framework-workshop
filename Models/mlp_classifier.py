import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers):
        super(Classifier, self).__init__()
        
        layers = []
        for i in range(len(hidden_layers)):
            in_s = input_size if i==0 else hidden_layers[i-1]
            layers += [nn.Linear(in_s, hidden_layers[i])]
            layers += [nn.BatchNorm1d(hidden_layers[i])]
            layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_layers[-1], num_classes), nn.BatchNorm1d(num_classes), nn.Softmax()]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)