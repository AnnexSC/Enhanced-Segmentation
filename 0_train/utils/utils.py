
def train(args, train_loaders, models_dict, optimizer, scheduler, writer, loggers):
    """
    Model Trainer
    """
    # To Do

def validation(args, val_loaders, models_dict, writer, loggers):
    """
    Model validator
    """
    # To Do 


def prep_train(args):
    """
    Create results directories, setup logger, Tensorboard.
    """
    # To Do
    return writer, loggers

def get_models(args):
    """
    Get model to train.
    """
    return models_dict

def get_optimizer(args, models_dict):
    """
    Get optimizer and scheduler for train.
    """
    return optimizer, scheduler