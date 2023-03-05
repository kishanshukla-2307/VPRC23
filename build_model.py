
from models.mobile_net import MobileNet
from models.beit import BEiT
from pyexpat import model

def build_model(config, model_name):
    if model_name == 'MobNet':
        return build_mobnet(config)
    elif model_name == 'BEiT':
        return build_BEiT(config)
    else:
        print('model not found')

def build_mobnet(config):
    backbone = MobileNet(config['MobNet']['model_name'], config['MobNet']['embedding_dim'])
    return backbone

def build_BEiT(config):
    backbone = BEiT(config['BEiT']['model_name'], config['BEiT']['embedding_dim'])
    return backbone
