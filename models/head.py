import timm
from torch import nn

class Head(nn.Module):
    def __init__(self, input_features, output_features):
        super(Head, self).__init__()
        self.fc = nn.Linear(input_features, output_features)

    def forward(self, x):
        x = self.fc(x)
        return x