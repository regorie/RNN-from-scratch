import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class Trainer():
    def __init__(self, model, train_loader, test_loader, val_loader, optimizer, scheduler, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for batch in tqdm(self.train_loader):
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(source, target[:-1], mode='train')

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)

            trg = target[1:].view(-1)

            loss = self.criterion(output, trg)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)
    
    def train(self, epoch):

        for ep in range(epoch):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            print("Epoch: ", ep+1, " Train loss: ", train_loss, " Val loss: ", val_loss, " Learning rate: ", self.scheduler.get_last_lr())
            if val_loss <= self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()

    def evaluate(self):
        self.model.eval()
        loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                source = batch["source"].to(self.device)
                target = batch["target"].to(self.device)

                output = self.model(source, target[:-1], mode='test')
                output_dim = output.shape[-1]
                output = output.view(-1, output_dim)

                target = target[1:].view(-1)

                curr_loss = self.criterion(output, target)
                loss += curr_loss.item()
        return loss / len(self.test_loader)
    
    def validate(self):
        self.model.eval()
        loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                source = batch["source"].to(self.device)
                target = batch["target"].to(self.device)

                output = self.model(source, target[:-1], mode='test')
                output_dim = output.shape[-1]
                output = output.view(-1, output_dim)

                target = target[1:].view(-1)

                curr_loss = self.criterion(output, target)
                loss += curr_loss.item()
        return loss / len(self.val_loader)