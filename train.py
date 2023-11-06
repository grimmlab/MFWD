import torch
from pathlib import Path
import numpy as np
from utils import seed_all, EarlyStopping, get_model, save_model
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from dataset import get_dataloaders
import argparse


def validate_batch(loader, model, device):
    model = model.to(device)
    model.eval()
    gts = []
    preds = []
    with torch.no_grad():
        with tqdm(loader, unit="batch", leave=True) as tepoch:
            for inputs, targets in tepoch:
                inputs = inputs.float().to(device)
                targets = targets.long().to(device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                preds.extend(predictions.cpu().numpy())
                gts.extend(targets.cpu().numpy())
    f1_s = f1_score(np.array(gts), np.array(preds), average="weighted")
    return f1_s


def train_epoch(train_loader, model, optimizer, criterion, epoch):
    running_loss = 0.0
    train_l = tqdm(train_loader, total=len(train_loader))
    for imgs, lbls in train_l:
        imgs = imgs.to("cuda")
        lbls = lbls.to("cuda")
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_l.set_postfix_str(f'[{epoch + 1}] loss: {running_loss:.3f}')
    return





def create_train_parser():
    my_parser = argparse.ArgumentParser(description='Script used for training a model')

    my_parser.add_argument('--run_number',
                           type=str,
                           help='Number of this run')

    my_parser.add_argument('--encoder',
                           type=str,
                           help='Name of an encoder (feature extractor), implemented: efficientnet_b0, resnet10',
                           default="efficientnet_b0")

    my_parser.add_argument('--batch_size',
                           type=int,
                           help='Number of patches in a batch', default=512)

    my_parser.add_argument('--lr',
                           type=float,
                           help='Learning rate', default=1e-3)
    
    my_parser.add_argument('--lr_scheduler_factor',
                           type=float,
                           help='Factor to reduce the learning rate', default=0.5)
    
    my_parser.add_argument('--lr_scheduler_patience',
                           type=int,
                           help='Number of epochs for the learning rate scheduler to wait', default=5)
    
    my_parser.add_argument('--max_epochs',
                           type=int,
                           help='Maximal number of epochs to train for', default=50)

    my_parser.add_argument('--validate_every_n_epochs',
                           type=int,
                           help='Validate every n epochs', default=1)

    my_parser.add_argument('--es_patience',
                           type=int,
                           help='patience for early stopping', default=5)

    args = my_parser.parse_args()
    return args



if __name__ == "__main__":
    seed_all(seed=42)
    args = create_train_parser()
    train_folder = Path("/data/MFWD/patches/train")
    val_folder = Path("/data/MFWD/patches/validation")
    class_map_f = Path("/data/MFWD/class_map.txt")
    model_save_path = Path("./models")
    model_save_path.mkdir(parents=True, exist_ok=True)
    log_path = Path("./logs")
    log_path.mkdir(parents=True, exist_ok=True)
    img_size = (224, 224)
    n_workers = 4
    fname = f"{args.run_number}_{args.encoder}"
    val_log = []

    print(args)
    train_loader, val_loader, labels = get_dataloaders(
        train_folder, val_folder, class_map_f, batch_size=args.batch_size, img_size=img_size, n_workers=n_workers)
    
    model = get_model(args.encoder, pretrained=True, num_classes=len(labels))
    early_stop = EarlyStopping(patience=args.es_patience)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", verbose=True, factor=args.lr_scheduler_factor, min_lr=1e-5, patience=args.lr_scheduler_patience)
    model = model.to("cuda")
    last_lr = args.lr
    for epoch in range(args.max_epochs):  # loop over the dataset
        train_epoch(train_loader, model, optimizer, criterion, epoch)

        if epoch % args.validate_every_n_epochs == args.validate_every_n_epochs-1:
            f1_s = validate_batch(val_loader, model, device="cuda")
            val_log.append([epoch, f1_s])
            early_stop(f1_s)
            scheduler.step(metrics=f1_s)
            if early_stop.do_stop:
                print(f"Stopped early at epoch: {epoch+1}")
                break
    save_model(model, args.encoder, len(labels), epoch, args.lr, args.lr_scheduler_factor, save_path=model_save_path, fname=fname)
    val_log = np.asarray(val_log)
    np.savetxt(f"{str(log_path)}/{fname}.csv", val_log, delimiter=",", fmt=["%d", "%.5f"])