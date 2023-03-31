import os
import torch
import argparse


### Custom library import
from utils import utils
from dataset import data_loader


def main(args):
    if torch.backends.mps.is_available():
        device = torch.device('mps:0')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'

    writer, loggers = prep_train(args)
    models_dict = get_models(args)
    optimizer, scheduler = get_optimizer(args, models_dict)
    train_loaders, val_loaders = get_data_loaders(args)

    if args.resume_file:
        checkpoint = torch.load(args.resume_file)
    else:
        args.start_epoch = 0


    for epoch in range(args.start_epoch, args.max_epoch):    
        args.epoch = epoch

        train(args, train_loaders, models_dict, optimizer, scheduler, writer, loggers)

        validation(args, val_loaders, models_dict, writer, loggers)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### Results save related 
    parser.add_argument("--save_dir", help="directory that train results saved.")

    ### Data related
    parser.add_argument("--data_json_path", help="data json file path.")
    
    ### Model related 

    ### Train related
    parser.add_argument("--resume_file", help="checkpoint file path.")
    

    
    args = parser.parse_args()

    main(args)


