import torch
import torch.nn as nn

def get_same_pad(k,s):
    return (k-s+1)//2


class Resblock(nn.Module):

    def __init__(self, in_c, out_c, kernel, stride, dropout):
        
        super(Resblock, self).__init__()
        conv1 = nn.Conv2d(in_c, out_c, kernel, stride, padding=get_same_pad(kernel, stride))
        bn1 = nn.BatchNorm1d(out_c)
        relu1 = nn.ReLU()
        do1 = nn.Dropout(p=dropout)
        
        conv2 = nn.Conv1d(out_c, out_c, kernel, padding=get_same_pad(kernel, 1))
        bn2 = nn.BatchNorm1d(out_c)
        
        
        self.downsample = None
        if in_c != out_c:
            self.downsample = nn.Conv1d(in_c, out_c, kernel, stride, padding=get_same_pad(kernel, stride))
        self.block = nn.Sequential(conv1, bn1, relu1, do1,
                                  conv2, bn2)
        self.relu = torch.nn.ReLU()
        self.do = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.block(x)
        if self.downsample:
            h += self.downsample(x)
        h = self.relu(h)
        return self.do(h)


class ResModel(nn.Module):
    def __init__(self, input_size, in_channels, num_classes, out_channels, kernels, strides, dropouts):
        
        super(ResModel, self).__init__()
        self.history = {'train':[], 'val':[]}
        
        self.blocks = nn.ModuleList()
        n = len(kernels)
        for i in range(n):
            in_c = in_channels if i == 0 else out_channels[i-1]
            out_c = out_channels[i]
            kernel = kernels[i]
            stride = strides[i]
            dropout = dropouts[i]
            block = Resblock(in_c, out_c, kernel, stride, dropout)
            self.blocks.append(block)
            input_size = input_size // stride
        
        self.seq = nn.Sequential(nn.Flatten(), nn.Linear(input_size, num_classes), nn.Softmax())
    
    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return self.seq(h)
