import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from copy import deepcopy

class NODE():
    def __init__(self, model, init_lr, datasets, epochs, batch_size, device='cuda:0'):
        super(NODE, self).__init__()
        self.datasets = datasets
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = deepcopy(model).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, client_idx, model_parameters):
        self.model.load_state_dict(model_parameters)

        #TODO 아래의 train dataset을 가져오는 파트 바꿔야함, Input 형식도 동일
        #x_train = torch.tensor(self.datasets[client_idx]['x'], dtype=torch.float32).to(self.device)
        #y_train = torch.tensor(self.datasets[client_idx]['y'], dtype=torch.long).to(self.device)

        self.model.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(x_train.size()[0])
            for i in range(0, x_train.size()[0], self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_x, batch_y = x_train[indices], y_train[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            print(f"Client {client_idx}, Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr