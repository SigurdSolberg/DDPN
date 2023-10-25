import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TrainingLoop():

    def __init__(self, model, optimizer, loss_function, device, scheduler = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.scheduler = scheduler

    def train(self, train_loader, val_loader, epochs, verbose = True):

        self.model.train()

        for epoch in range(epochs):

            train_loss = 0
            val_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)

            if self.scheduler is not None:
                self.scheduler.step()
            if verbose and epoch % 5 == 0:
                val_loss, val_accuracy = self.test(val_loader)
                train_loss, train_accuracy = self.test(train_loader)
                print(f'Epoch {epoch}: Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | Train accuracy: {train_accuracy:.3f} | Val accuracy: {val_accuracy:.3f}, LR: {self.scheduler.get_last_lr()[0]:.3f}')
        
        return train_loss, val_loss
    
    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_function(output, target).item()
                _, pred = torch.max(output.data, 1)
                _, target = torch.max(target.data, 1)
                correct += (pred == target).sum().item()
        test_loss /= len(test_loader.dataset)
        return test_loss, correct / len(test_loader.dataset)
    
class DHDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype = torch.float)  # X is a list of nested tensors, each nested tensor contains all the persistence diagrams of that cloud
        self.y = torch.tensor(y, dtype = torch.float)  # 1-hot encoding. 5 classes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()
        return self.x[index], self.y[index]
