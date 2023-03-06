from torch import nn
from .head import Head

class CustomVitB(nn.Module):
    def __init__(self, base_model, emb_dim):
        super(CustomVitB, self).__init__()
        self.backbone = base_model
        self.head = Head(self.backbone.token_embedding.embedding_dim, emb_dim)

    def forward(self, x):
        x = self.backbone.encode_image(x)
        x = self.head(x)
        return x