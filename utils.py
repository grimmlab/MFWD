import numpy as np
import random
import os
import torch
import timm

def seed_all(seed):
    """
    sets the initial seed for numpy and pytorch to get reproducible results.
    One still need to restart the kernel to get reproducible results, as discussed in:
    https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeding everything to seed {seed}")
    return


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

class EarlyStopping:
    """Stops the training if the validation score doesn't improve for a given patience.
    Works with scores that need to be maximized"""
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.do_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.do_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def get_model(encoder_name, pretrained, num_classes):
    model = timm.create_model(
        encoder_name, pretrained=pretrained, num_classes=num_classes)
    return model


def save_model(model, encoder, num_classes, epoch, lr, lr_factor, save_path, fname):
    save_dict = {"model_state_dict": model.state_dict(),
                 "encoder": encoder,
                 "num_classes": num_classes,
                 "epoch": epoch,
                 "lr": lr,
                 "lr_scheduler_factor": lr_factor}
    torch.save(save_dict, save_path/f"{fname}.pth")

def load_model(save_path, fname):
    loaded_model = torch.load(
        save_path/f"{fname}.pth", map_location=torch.device("cpu"))
    encoder = loaded_model["encoder"]
    num_classes = loaded_model["num_classes"]
    model = get_model(encoder, pretrained=False, num_classes=num_classes)
    model.load_state_dict(loaded_model["model_state_dict"])
    return model