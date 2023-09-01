import torch
import torch.nn as nn

from models.resnet import build_encoder
from models.projector import Projector

class CLNet(nn.Module):
    def __init__(
            self, 
            encoder="resnet18", 
            pretrained=False, 
            linear_layers=[2048, 128],
            in_channels=1, 
            normalize=True):
        super(CLNet, self).__init__()
        """
        args:
            encoder: encoder name
            pretrained: whether to use pretrained weights
            feat_dim: feature dimension output by encoder
            linear_layers: projector specification
            useclf: whether to use angle order prediction task 
            normalize: whether to normalize feature vectors output by encoder
        """
        assert len(linear_layers) == 2
        hidden_size, output_size = linear_layers
        self.encoder = build_encoder(encoder, pretrained, in_channels=in_channels, normalize=normalize)
        self.early_fusion = False
        if in_channels > 1:
            self.early_fusion = True
        if ("resnet18" in encoder) or ("resnet34" in encoder):
            feat_dim = 512
        else:
            feat_dim = 2048
        self.projector = Projector(feat_dim, hidden_size, output_size)
        self.use_proj = normalize
        
    def forward(self, x):
        '''
        x: (batch_size, num_clips, num_modes, in_channels, height, width)
        num_modes: number of positive pairs used to calculate contrastive loss
        in_channels: number of M-mode images of each patient used to calculate joint represntation
        '''
        assert x.size(3) == 1 # single channel input
        output = []
        for clip in range(x.size(1)):
            x_c = x[:, clip] # (batch_size, num_modes, 1, height, width)
            if self.early_fusion:
                x_c = torch.squeeze(x_c, 2) # (batch_size, num_modes, height, width)
                x_seq = self.encoder(x_c) # (batch_size, feature_dim)
            else:
                x_seq = []
                for t in range(x_c.size(1)):
                    x_t = self.encoder(x_c[:, t, :, :, :]) 
                    if self.use_proj:
                        x_t = self.projector(x_t) # (batch_size, output_size)
                    x_seq.append(x_t) 
                x_seq = torch.stack(x_seq, dim=1) # (batch_size, num_modes, feature_dim/output_size)
            output.append(x_seq) 

        output = torch.stack(output, dim=1) # (batchsize, num_clips, [num_modes,] output_size)
        
        return output



