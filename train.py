import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from datetime import datetime

import config
from model import CustomCNNModel
from custom_dataset import MiniImagenetDataset


def create_dataloaders() -> (DataLoader, DataLoader):
    mini_imagenet_ds = MiniImagenetDataset(data_path=config.imagenet_data_dir, transform=config.transform)
    val_split = int(len(mini_imagenet_ds) * 0.2)
    train_split = len(mini_imagenet_ds) - val_split
    train_set, val_set = random_split(mini_imagenet_ds, [train_split, val_split])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=config.shuffle)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=config.shuffle)
    return train_loader, val_loader


def validate(model: CustomCNNModel, val_loader: DataLoader, criterion: nn.CrossEntropyLoss):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Validating:", unit="batch") as tbar:
            for batch in tbar:
                images, labels = batch
                out = model(images.to(config.device))
                loss = criterion(out, labels.to(config.device))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def run_epoch(model: CustomCNNModel, train_loader: DataLoader, val_loader: DataLoader, epoch: int, optimizer: optim,
              criterion: nn.CrossEntropyLoss):
    model.train()
    running_loss = 0
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch") as tbar:
        for batch in tbar:
            images, labels = batch
            out = model(images.to(config.device))
            loss = criterion(out, labels.to(config.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    average_loss = running_loss / len(train_loader)
    average_val_loss = validate(model=model, val_loader=val_loader, criterion=criterion)
    print(f'Training loss: {average_loss} | Validation loss: {average_val_loss}')
    return model


def run():
    model = CustomCNNModel(n_classes=config.n_classes).to(config.device)
    train_loader, val_loader = create_dataloaders()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing).to(config.device)
    current_datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    for epoch in range(config.epochs):
        run_epoch(model=model, train_loader=train_loader, val_loader=val_loader, epoch=epoch, optimizer=optimizer,
                  criterion=criterion)
        torch.save(model.state_dict(), f'model_{current_datetime_str}_epoch{epoch}.pt')


if __name__ == '__main__':
    run()
