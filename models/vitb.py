from torch import nn
from .head import Head
from .neck import Neck

class CustomVitB(nn.Module):
    def __init__(self, base_model, emb_dim, num_classes):
        super(CustomVitB, self).__init__()
        self.backbone = base_model
        self.neck = Neck(self.backbone.token_embedding.embedding_dim, emb_dim)
        self.head = Head(emb_dim, num_classes)

    def forward(self, x):
        x = self.backbone.encode_image(x)
        x = self.neck(x)
        x = self.head(x)
        return x