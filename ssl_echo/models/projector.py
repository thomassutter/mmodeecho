import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.layernorm1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = F.normalize(x)
        return x
