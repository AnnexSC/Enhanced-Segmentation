import torch


### Custom library import
import transform

def get_data_loaders(args):
    """
    Get train loaders and validation loaders
    """
    return train_loaders, val_loaders


class MyDataset(torch.utils.data.Dataset):
    """
    Class to load datasaet
    """
    def __init__(self):
        # To Do

    def __getitem__(self, index):
        # To Do

    def __len__(self):
        # To Do