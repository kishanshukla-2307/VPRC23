
from models.mobile_net import MobileNet
from models.beit import BEiT
from models.vitb import CustomVitB
from models.vith import CustomVitH
from pyexpat import model
import open_clip

def build_model(config, model_name):
    if model_name == 'MobNet':
        return build_mobnet(config)
    elif model_name == 'BEiT':
        return build_BEiT(config)
    elif model_name == 'VitB':
        return build_VitBClip(config)
    elif model_name == 'VitH':
        return build_VitHClip(config)
    else:
        raise NotImplementedError(f'{model_name} model not found')

def build_mobnet(config):
    backbone = MobileNet(config['MobNet']['model_name'], config['MobNet']['num_classes'])
    return backbone, None

def build_BEiT(config):
    backbone = BEiT(config['BEiT']['model_name'], config['BEiT']['embedding_dim'])
    return backbone, None

def build_VitBClip(config):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    
    model = CustomVitB(model, config['VitB']['num_classes'])

    for param in model.backbone.parameters():
        param.requires_grad = False

    return model, preprocess

def build_VitHClip(config):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    
    model = CustomVitH(model, config['VitH']['embedding_dim'], config['VitH']['num_classes'])

    for param in model.backbone.parameters():
        param.requires_grad = False

    return model, preprocess