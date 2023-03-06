
from models.mobile_net import MobileNet
from models.beit import BEiT
from models.vitB import VitB
from pyexpat import model
import open_clip

def build_model(config, model_name):
    if model_name == 'MobNet':
        return build_mobnet(config)
    elif model_name == 'BEiT':
        return build_BEiT(config)
    elif model_name == 'VitB':
        return build_VitClip(config)
    else:
        raise NotImplementedError(f'{model_name} model not found')

def build_mobnet(config):
    backbone = MobileNet(config['MobNet']['model_name'], config['MobNet']['num_classes'])
    return backbone, None

def build_BEiT(config):
    backbone = BEiT(config['BEiT']['model_name'], config['BEiT']['embedding_dim'])
    return backbone, None

def build_VitClip(config):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    
    model = VitB(model, config['VitB']['num_classes'])

    for param in model.backbone.parameters():
        param.requires_grad = False

    return model, preprocess