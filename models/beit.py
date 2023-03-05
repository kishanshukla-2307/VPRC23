import timm
from torch import nn

class BEiT(nn.Module):
    def __init__(self, emb_dim):
        self.backbone = timm.create_model(self.model_name, pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x.pooler_output