import timm
from torch import nn

class MobileNet(nn.Module):
    def __init__(self, emb_dim):
        self.backbone = timm.create_model(self.model_name, pretrained=True)
        self.backbone.classifier = nn.Linear(1280, self.emb_dim)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.fc(x)

        return x