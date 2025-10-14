import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class Trainer():
    def __init__(self, model, train_loader, test_loader, val_loader, optimizer, criterion, device, learning_rate_update_point=5):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.learning_rate_update_point = learning_rate_update_point

        self.best_val_loss = float('inf')
        self.best_model_state = None

        self.loss_list = []

    def train_epoch(self, test_interval=None):
        self.model.train()
        epoch_loss = 0
        iter = 0
        self.loss_list.append([])
        for batch in tqdm(self.train_loader):
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(source, target[:-1], src_lengths=batch['src_lengths'], mode='train')

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)

            trg = target[1:].view(-1)

            loss = self.criterion(output, trg)
            loss.backward()
            #nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1000)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            epoch_loss += loss.item()

            iter += 1
            if test_interval and iter % test_interval == 0:
                self.loss_list[-1].extend([self.validate()])
                self.model.train()

        return epoch_loss / len(self.train_loader)
    
    def train(self, epoch, test_interval):

        for ep in range(epoch):
            train_loss = self.train_epoch(test_interval)
            val_loss = self.validate()
            print("Epoch: ", ep+1, " Train loss: ", train_loss, " Val loss: ", val_loss, " Learning rate: ", self.optimizer.param_groups[0]['lr'])
            if (ep+1) >= self.learning_rate_update_point:
                self.optimizer.param_groups[0]['lr'] *= 0.5
            
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

                output = self.model(source, target[:-1], src_lengths=batch['src_lengths'], mode='test')
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

                output = self.model(source, target[:-1], src_lengths=batch['src_lengths'], mode='test')
                output_dim = output.shape[-1]
                output = output.view(-1, output_dim)

                target = target[1:].view(-1)

                curr_loss = self.criterion(output, target)
                loss += curr_loss.item()
        return loss / len(self.val_loader)
    