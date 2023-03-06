from torch import nn

class Neck(nn.Module):
    def __init__(self, in_features, out_features):
        super(Neck, self).__init__()
        self.neck = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(0.3),
                nn.Linear(in_features, out_features*2),
                nn.Linear(out_features*2, out_features),
            )


    def forward(self, x):
        x = self.neck(x)
        return x