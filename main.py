from collections import OrderedDict
import os
from tqdm import tqdm
from train import *
from data_handler import *
from dataset import *
from samplers import *
from loss import *
from build_model import build_model
from logger import create_logger

import argparse
import yaml

def load_config(config_filepath='./config.yaml'):
    with open(config_filepath, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    return config

def set_data_handler(train_csv_path, test_csv_path, split_ratio, preprocessor):
    data_handler = DataHandler(train_csv_path, test_csv_path)
    data_handler.load_data()
    data_handler.split(split_ratio)
    data_handler.set_transformation(preprocessor)
    return data_handler

def set_criterion(config, model_name):
    if config[model_name]['loss_fn'] == 'arc_face':
        return ArcFaceLoss()
    elif config[model_name]['loss_fn'] == 'triplet_loss':
        return BatchAllTripletLoss()
    elif config[model_name]['loss_fn'] == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif config[model_name]['loss_fn'] == 'dense_cross_entropy':
        return DenseCrossEntropy()
    else:
        raise NotImplementedError('Loss function not found!!')

def set_train_loader(config, model_name, train_set):
    if config[model_name]['sampler'] == 'default':
        return torch.utils.data.DataLoader(train_set, batch_size=config[model_name]['batch_size'], shuffle=True)
    elif config[model_name]['sampler'] == 'group_based':
        return torch.utils.data.DataLoader(train_set, batch_sampler=batch_sampler)
    else:
        raise NotImplementedError('{} sampler not implemented!'.format(config[model_name]['sampler']))

def set_validation_loader(config, model_name, validation_set):
    return torch.utils.data.DataLoader(validation_set, batch_size=config[model_name]['batch_size'], shuffle=False)

def set_test_loader(config, model_name, test_set):
    return torch.utils.data.DataLoader(test_set, batch_size=config[model_name]['batch_size'], shuffle=False)

def set_optimizer(config, model_name, model):
    if config[model_name]['optimizer']['name'] == 'sgd':
        return torch.optim.SGD(model.parameters(),
                              lr=config[model_name]['optimizer']['lr'],
                              momentum=config[model_name]['optimizer']['momentum'],
                              weight_decay=config[model_name]['optimizer']['decay'])
    elif config[model_name]['optimizer']['name'] == 'adam':
        return torch.optim.Adam(model.parameters(),
                              lr=config[model_name]['optimizer']['lr'])
    elif config[model_name]['optimizer']['name'] == 'adamw':
        return torch.optim.Adam(model.parameters(),
                              lr=config[model_name]['optimizer']['lr'],
                              weight_decay=config[model_name]['optimizer']['decay'])
    else:
        raise NotImplementedError('Optimizer not found!!')
        
def save_checkpoint(config, model_name, model, optimizer, epoch, outdir):
    """Saves checkpoint to drive"""
    if config[model_name]['load_saved']:
        epoch += config[model_name]['epoch_offset']
    filename = "{}_{:04d}.pth".format(model_name, epoch)
    directory = outdir
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    state = OrderedDict([
        ('config', config),
        ('state_dict', weights),
        ('optimizer', optimizer.state_dict()),
        ('epoch', epoch),
    ])

    torch.save(state, filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', dest='model', action='store',
                    default='MobNet',
                    help='Model name')
    args = parser.parse_args()

    if args.model == None:
        raise ('Model name not provided')

    config = load_config()
    
    logger = create_logger(output_dir=config['system']['output']+"/"+args.model, name=f"{args.model}")
    logger.info(f"config: {config}")

    model, preprocessor = build_model(config, logger, args.model)
    model.to(config['system']['device'])

    data_handler = set_data_handler(config['system']['train_csv_path'], config['system']['test_csv_path'], config['system']['split_ratio'], preprocessor)

    train_set = Product10kDataset(data_handler.train['name'].to_numpy(), 
                    data_handler.train['class'].to_numpy(),
                    data_handler.train['group'].to_numpy(),
                    config['system']['train_samples_path'],
                    data_handler.transform, 
                    offline_strategy=False)

    validation_set = Product10kDataset(data_handler.validation['name'].to_numpy(),
                        data_handler.validation['class'].to_numpy(),
                        data_handler.train['group'].to_numpy(),
                        config['system']['train_samples_path'],
                        data_handler.transform,
                        offline_strategy=False)

    test_set = Product10kDataset(data_handler.test['name'].to_numpy(),
                    data_handler.test['class'].to_numpy(),
                    data_handler.test['Usage'].to_numpy(),
                    config['system']['test_samples_path'],
                    data_handler.transform,
                    offline_strategy=False)

    batch_sampler = Batch_Sampler(data_handler.train['class'].to_numpy(), data_handler.train['group'].to_numpy(), config[args.model]['batch_size'])
                                                                                                                                               
    loss_fn = set_criterion(config, args.model)
    optimizer = set_optimizer(config, args.model, model)
    

    train_loader = set_train_loader(config, args.model, train_set)
    validation_loader = set_validation_loader(config, args.model, validation_set)
    test_loader = set_test_loader(config, args.model, test_set)

    # batch_test = next(iter(train_loader))

    logger.info("Start training")

    epochs_iter = tqdm(range(config[args.model]['epochs']), dynamic_ncols=True, 
                    desc='Epochs', position=0)
    
    best_acc = 0.0
    for epoch in epochs_iter:
        logger.info(f"----------[Epoch {epoch}]----------")  
        train_epoch(config, logger, model, train_loader, loss_fn, optimizer, epoch)
        save_checkpoint(config, args.model, model, optimizer, epoch, config['system']['output'] + "/" + args.model)
        # epoch_avg_acc = self.validation(self.model, self.val_loader, self.criterion, epoch)

    # # print("best accuracy: ", best_acc)

    # print("test mAP: ", self.test(validation_set, test_set))