import numpy as np
import torch
import torch.nn as nn


def get_model_layout(model_type):
    if "resnet18" in model_type:
        return Block, [2, 2, 2, 2]
    elif "resnet34" in model_type:
        return Block, [2, 4, 6, 3]
    elif "resnet50" in model_type:
        return Bottleneck, [3, 4, 6, 3]
    elif "resnet101" in model_type:
        return Bottleneck, [3, 4, 23, 3]
    elif "resnet152" in model_type:
        return Bottleneck, [3, 8, 36, 3]

def get_model_layout2d(model_type):
    if "resnet18" in model_type:
        return Block2d, [2, 2, 2, 2]
    elif "resnet34" in model_type:
        return Block2d, [2, 4, 6, 3]
    elif "resnet50" in model_type:
        return Bottleneck2d, [3, 4, 6, 3]
    elif "resnet101" in model_type:
        return Bottleneck2d, [3, 4, 23, 3]
    elif "resnet152" in model_type:
        return Bottleneck2d, [3, 8, 36, 3]

class Block(nn.Module): # building block for ResNet18 and 34
    expansion = 1
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x.clone()
        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        x += identity
        x = self.relu(x)

        return x

class Bottleneck(nn.Module): # building block for ResNet50, 101 and 152
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm1d(out_channels*self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        x += identity
        x = self.relu(x)

        return x

class ResNet_1d(nn.Module):
    def __init__(self, block_type, num_blocks_list, num_modes, num_filters, kernel_height):
        super().__init__()
        self.in_channels = num_filters

        # use kernel height == image height to calculate 1D convolution along temporal axis
        self.conv1 = nn.Conv2d(num_modes, num_filters, kernel_size=(kernel_height, 7), stride=2, padding=(0, 3), bias=False)
        self.batch_norm1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block_type, num_blocks_list[0], planes=num_filters)
        self.layer2 = self.make_layer(block_type, num_blocks_list[1], planes=2*num_filters, stride=2)
        self.layer3 = self.make_layer(block_type, num_blocks_list[2], planes=4*num_filters, stride=2)
        self.layer4 = self.make_layer(block_type, num_blocks_list[3], planes=8*num_filters, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512*block_type.expansion, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.squeeze(x) # to remove height dimension
        # in case batch_size is 1 (e.g. during testing), the batch_size dim is squeezed 
        if x.dim() == 2: 
            x = torch.unsqueeze(x, dim=0)
        x = self.relu(self.batch_norm1(x))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def make_layer(self, block_type, num_blocks, planes, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes*block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*block_type.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes*block_type.expansion)
            )
        
        layers.append(block_type(self.in_channels, planes, downsample, stride))
        self.in_channels = planes*block_type.expansion

        for i in range(num_blocks-1):
            layers.append(block_type(self.in_channels, planes))

        return nn.Sequential(*layers)
        
    
class ResNet_lstm(ResNet_1d):
    def __init__(self, block_type, num_blocks_list, num_modes, num_filters, kernel_height):
        super().__init__(block_type, num_blocks_list, num_modes, num_filters, kernel_height)
        
        hidden_size = 256
        # overwrite first conv layer to accept input with 1 channel, since we are passing each M-mode separately
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=(kernel_height, 7), stride=2, padding=(0, 3), bias=False)

        self.lstm = nn.LSTM(input_size=512*block_type.expansion, hidden_size=hidden_size,
            num_layers=1, batch_first=True, dropout=0)
        self.fc = nn.Linear(hidden_size, 1000)

    def forward(self, x):
        x_seq = []
        # x has format [batch_size, num_modes, H, W] 
        for t in range(x.size(1)): 
            # x_t has format [batch_size, 1, H, W] 
            x_t = x[:, t:t+1, :, :] 
            x_t = self.conv1(x_t)
            x_t = torch.squeeze(x_t) # to remove height dimension
            # in case batch_size is 1 (e.g. during testing), the batch_size dim is squeezed 
            if x_t.dim() == 2: 
                x_t = torch.unsqueeze(x_t, dim=0) 
            x_t = self.relu(self.batch_norm1(x_t))
            x_t = self.max_pool(x_t)

            x_t = self.layer1(x_t)
            x_t = self.layer2(x_t)
            x_t = self.layer3(x_t)
            x_t = self.layer4(x_t)

            x_t = self.avgpool(x_t) 
            # x_t has format [batch_size, num_filters, 1] 
            x_t = torch.squeeze(x_t) 
 
            # in case batch_size is 1 (e.g. during testing), the batch_size dim is squeezed 
            if x_t.dim() == 1: 
                x_t = torch.unsqueeze(x_t, dim=0) 
                 
            # x_t has format [batch_size, num_filters] 
            x_seq.append(x_t) 
 
        # need format [batch_size, seq_len, input_size] for lstm 
        x = torch.stack(x_seq, dim=1) 
        # x has format [batch_size, num_modes, num_filters] 
        x, _ = self.lstm(x) 
        # x has format [batch_size, num_modes, hidden_size] 
        x = x[:, -1, :] # use last entry in sequence 
        x = self.fc(x) 
 
        return x 


# 2d models
class Block2d(nn.Module): # building block for ResNet18 and 34
    expansion = 1
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x.clone()
        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        x += identity
        x = self.relu(x)

        return x

class Bottleneck2d(nn.Module): # building block for ResNet50, 101 and 152
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        x += identity
        x = self.relu(x)

        return x

class ResNet_2d(nn.Module):
    def __init__(self, block_type, num_blocks_list, num_modes, num_filters):
        super().__init__()
        self.in_channels = num_filters

        self.conv1 = nn.Conv2d(num_modes, num_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block_type, num_blocks_list[0], planes=num_filters)
        self.layer2 = self.make_layer(block_type, num_blocks_list[1], planes=2*num_filters, stride=2)
        self.layer3 = self.make_layer(block_type, num_blocks_list[2], planes=4*num_filters, stride=2)
        self.layer4 = self.make_layer(block_type, num_blocks_list[3], planes=8*num_filters, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512*block_type.expansion, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.squeeze(x) # to remove height dimension
        # in case batch_size is 1 (e.g. during testing), the batch_size dim is squeezed 
        if x.dim() == 2: 
            x = torch.unsqueeze(x, dim=0)
        x = self.relu(self.batch_norm1(x))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def make_layer(self, block_type, num_blocks, planes, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes*block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*block_type.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*block_type.expansion)
            )
        
        layers.append(block_type(self.in_channels, planes, downsample, stride))
        self.in_channels = planes*block_type.expansion

        for i in range(num_blocks-1):
            layers.append(block_type(self.in_channels, planes))

        return nn.Sequential(*layers)
        
    
class ResNet_lstm2d(ResNet_2d):
    def __init__(self, block_type, num_blocks_list, num_modes, num_filters):
        super().__init__(block_type, num_blocks_list, num_modes, num_filters)
        
        hidden_size = 256
        # overwrite first conv layer to accept input with 1 channel, since we are passing each M-mode separately
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=7, stride=2, padding=3, bias=False)

        self.lstm = nn.LSTM(input_size=512*block_type.expansion, hidden_size=hidden_size,
            num_layers=1, batch_first=True, dropout=0)
        self.fc = nn.Linear(hidden_size, 1000)

    def forward(self, x):
        x_seq = []
        # x has format [batch_size, num_modes, H, W] 
        for t in range(x.size(1)): 
            # x_t has format [batch_size, 1, H, W] 
            x_t = x[:, t:t+1, :, :] 
            x_t = self.conv1(x_t)
            x_t = torch.squeeze(x_t) # to remove height dimension
            # in case batch_size is 1 (e.g. during testing), the batch_size dim is squeezed 
            if x_t.dim() == 3: 
                x_t = torch.unsqueeze(x_t, dim=0) 
            x_t = self.relu(self.batch_norm1(x_t))
            x_t = self.max_pool(x_t)

            x_t = self.layer1(x_t)
            x_t = self.layer2(x_t)
            x_t = self.layer3(x_t)
            x_t = self.layer4(x_t)

            x_t = self.avgpool(x_t) 
            # x_t has format [batch_size, num_filters, 1] 
            x_t = torch.squeeze(x_t) 
 
            # in case batch_size is 1 (e.g. during testing), the batch_size dim is squeezed 
            if x_t.dim() == 1: 
                x_t = torch.unsqueeze(x_t, dim=0) 
                 
            # x_t has format [batch_size, num_filters] 
            x_seq.append(x_t) 
 
        # need format [batch_size, seq_len, input_size] for lstm 
        x = torch.stack(x_seq, dim=1) 
        # x has format [batch_size, num_modes, num_filters] 
        x, _ = self.lstm(x) 
        # x has format [batch_size, num_modes, hidden_size] 
        x = x[:, -1, :] # use last entry in sequence 
        x = self.fc(x) 
 
        return x 


