import torch
import torch.nn as nn
from models.clnet import CLNet

class Regressor(nn.Module):
    def __init__(
            self, 
            encoder="resnet18", 
            linear_layers=[2048, 128], 
            saved_pretrained_model="", 
            in_channels=1, 
            modes=10, 
            combine=None, 
            clf="linear", 
            freeze=False, 
            normalize=False, 
            pretrained=False):
        super(Regressor, self).__init__()
        """
        args:
            encoder: encoder name
            feature_dim: feature dimension output by encoder
            linear_layers: projector specification
            saved_pretrained_model: file path to load pretrained weights from contrastive learning
            freeze: whether to freeze the encoder part
            normalize: whether to normalize feature vectors output by encoder
            pretrained: whether to use pretrained weights (only valid when saved_pretrained_model = None)
        """
        self.combine = combine
        if ("resnet18" in encoder) or ("resnet34" in encoder):
            feature_dim = 512
        else:
            feature_dim = 2048
        clNet = CLNet(encoder, pretrained, linear_layers, in_channels, normalize)
        if saved_pretrained_model != None:
            checkpoint = torch.load(saved_pretrained_model)
            clNet.load_state_dict(checkpoint["state_dict"])
        self.enc = clNet
        if freeze:  # only fine-tuning the last linear layer 
            for p in self.parameters():
                p.requires_grad = False
        if combine == "concat":
            input_size = feature_dim * modes
        else:
            input_size = feature_dim
        if combine == "lstm":
            self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=256, num_layers=1, batch_first=True, dropout=0)
            input_size = 256
        if clf == "linear":
            self.fc = nn.Linear(input_size, 1)
        elif clf == "mlp":
            self.fc = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(),
                nn.Linear(input_size, 1)
            )
        else:
            raise NotImplementedError
        
    def forward(self, x):
        '''
        x: (batch_size, num_clips, num_modes, in_channels, height, width)
        '''      
        assert x.size(3) == 1 # single channel input
        x_feat = self.enc(x) # (batchsize, num_clips, [num_modes,] feature_dim)
        outputs = []
        for clip in range(x_feat.size(1)):
            x_joint = x_feat[:, clip] # (batchsize, [num_modes,] feature_dim)
            if self.combine == "concat":
                x_joint = x_joint.reshape((x_joint.size(0), -1)) # (batchsize, num_modes*feature_dim)
            elif self.combine == "avg":
                x_joint = torch.mean(x_joint, dim=1) # (batchsize, feature_dim)
            elif self.combine == "lstm":
                x_joint, _ = self.lstm(x_joint)
                x_joint = x_joint[:, -1, :] # (batchsize, feature_dim)
            output = self.fc(x_joint) # (batchsize, 1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1) # (batchsize, num_clips, 1)

        return torch.mean(outputs, dim=1)


