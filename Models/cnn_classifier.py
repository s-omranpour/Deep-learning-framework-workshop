import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes, in_channel, out_channels, kernels, strides, dropouts):
        super(Classifier, self).__init__()

        layers = []
        for i in range(len(out_channels)):
            in_c = in_channel if i ==0 else out_channels[i-1]
            out_c = out_channels[i]
            k = kernels[i]
            s = strides[i]
            d = dropouts[i]
            input_size = input_size // s
            layers += [nn.Conv2d(in_c, out_c, k, s, padding=(k-s+1)//2), 
                       nn.BatchNorm2d(out_c),
                       nn.ReLU(),
                       nn.Dropout(d)]
        layers += [nn.Flatten(), nn.Linear(out_c*out_c*input_size, num_classes), nn.BatchNorm2d(num_classes), nn.Softmax()]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)