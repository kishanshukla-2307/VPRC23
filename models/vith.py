from torch import nn
from .head import ArcMarginProduct_subcenter
from .neck import Neck

class CustomVitH(nn.Module):
    def __init__(self, base_model, emb_dim, num_classes):
        super(CustomVitH, self).__init__()
        self.backbone = base_model
        self.neck = Neck(self.backbone.token_embedding.embedding_dim, emb_dim)
        self.head = ArcMarginProduct_subcenter(emb_dim, num_classes)

    def forward(self, x):
        x = self.backbone.encode_image(x)
        x = self.neck(x)
        x = self.head(x)
        return x