import timm
from torch import nn

class MobileNet(nn.Module):
    def __init__(self, model_name, emb_dim):
        super(MobileNet, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone.classifier = nn.Linear(1280, emb_dim)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.fc(x)

        return x