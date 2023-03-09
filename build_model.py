
from models.mobile_net import MobileNet
from models.beit import BEiT
from models.vitb import CustomVitB
from models.vith import CustomVitH
from pyexpat import model
import open_clip
import torch
from collections import OrderedDict

def build_model(config, logger, model_name):
    logger.info("Building model...")
    if model_name == 'MobNet':
        return build_mobnet(config)
    elif model_name == 'BEiT':
        return build_BEiT(config)
    elif model_name == 'VitB':
        return build_VitBClip(config)
    elif model_name == 'VitH':
        return build_VitHClip(config, logger)
    else:
        raise NotImplementedError(f'{model_name} model not found')

def build_mobnet(config):
    backbone = MobileNet(config['MobNet']['model_name'], config['MobNet']['num_classes'])
    return backbone, None

def build_BEiT(config):
    backbone = BEiT(config['BEiT']['model_name'], config['BEiT']['embedding_dim'])
    return backbone, None

def build_VitBClip(config):
    if config['VitB']['load_saved']:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained=False)
        model = CustomVitB(model, config['VitB']['embedding_dim'], config['VitB']['num_classes'])
        ckpt = torch.load(config['VitB']['ckpt_path'],
                          map_location=config['system']['device'])
        
        new_state_dict = ckpt['state_dict']
        # for k, v in ckpt.items():
        #     name = k.replace("module.", "")
        #     new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
        model = CustomVitB(model, config['VitB']['embedding_dim'], config['VitB']['num_classes'])

    for param in model.backbone.parameters():
        param.requires_grad = False

    return model, preprocess


def build_VitHClip(config, logger):
    if config['VitH']['load_saved']:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained=False)
        model = CustomVitH(model, config['VitH']['embedding_dim'], config['VitH']['num_classes'])
        ckpt = torch.load(config['VitH']['ckpt_path'],
                          map_location=config['system']['device'])
        
        new_state_dict = ckpt['state_dict']
        # for k, v in ckpt.items():
        #     name = k.replace("module.", "")
        #     new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        model = CustomVitH(model, config['VitH']['embedding_dim'], config['VitH']['num_classes'])

    for param in model.backbone.parameters():
        param.requires_grad = False

    return model, preprocess


