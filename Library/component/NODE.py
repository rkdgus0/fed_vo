import os
import sys
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from time import time, strftime, gmtime
from copy import deepcopy
from torch.utils.data import DataLoader
from Library.train_util import whole_loss_function, flow_loss_function, pose_loss_function


class NODE():
    def __init__(self, model, optimizer, scheduler, datasets, epoch, batch_size, worker_num, device='cuda:0'):
        super(NODE, self).__init__()
        self.datasets = datasets
        self.epoch = epoch
        self.batch_size = batch_size
        self.worker_num = worker_num
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, node_idx, model_parameters, model_name):
        self.model.load_state_dict(model_parameters) #global model

        # setting node's train data
        train_data = self.datasets[node_idx]
        trainDataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, num_workers=self.worker_num)
        self.model.train()

        for curr_iter in range(self.epoch):
            t1 = time()
            for sample in trainDataloader:
                self.optimizer.zero_grad()
                
                # Local Model Update
                if model_name.lower() == 'vonet': 
                    loss, flow_loss, pose_loss, trans_loss, rot_loss = whole_loss_function(self.model, sample, 10, 1e-6, self.device)
                elif model_name.lower() == 'flownet' or model_name.lower() == 'matchingnet':
                    loss = flow_loss_function(self.model, sample, self.device)
                elif model_name.lower() == 'flowposenet' or model_name.lower() == 'posenet':
                    loss, trans_loss, rot_loss = pose_loss_function(self.model, sample, 1e-6, self.device)
                loss.backward()
                
                self.optimizer.step()
            self.scheduler.step()
            t2 = time()
            iter_time = strftime("%Hh %Mm %Ss", gmtime(t2-t1))
            print(f"Node {node_idx+1}, Iteration {curr_iter+1}/{self.epoch}, Local Loss: {loss.item()}, Time: {iter_time}")

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


from Library.datasets.dataset_util import ToTensor, Compose, CropCenter, DownscaleFlow, make_intrinsics_layer, dataset_intrinsics
from Library.datasets.dataset import initial_dataset
from Network.VONet import VONet
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR



if __name__ == '__main__':
    data_name = 'tartanair'
    root_dir = '/scratch/jeongeon/tartanAir/train_data'
    easy_hard = 'easy'
    node_num = 3
    transform = Compose([CropCenter((640, 448)), DownscaleFlow(), ToTensor()])
    test_environments = ['ocean', 'office2', 'soulcity', 'carwelding', 'abandonedfactory_night', 'seasonsforest', 'neighborhood', 'japanesealley', 'oldtown', 'carwelding', 'seasonsforest_winter', 'westerndesert', 'office', 'hospital', 'gascola', 'amusement', 'testDatset']
    epoch = 2
    batch_size = 64
    worker = 32
    sequence = 'P001'
    
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(VONet())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    print("Init Dataset...")
    train_data, test_data, node_envs = initial_dataset(data_name, root_dir, easy_hard, sequence, node_num, transform, test_environments)
    print("Success to init dataset!\n")

    print("Init model parameter...")
    #Node = compose_node(args, model, optimizer, scheduler, train_data)
    
    model_parameter = model.state_dict()
    print("Success to init model parameter!\n")

    print("Init Node component...")
    Node =  NODE(model, optimizer, scheduler, init_lr=0.01, datasets=train_data, epoch=epoch, batch_size=batch_size, worker_num=worker, device=device)

    print(f'Node num: {len(train_data)}')
    for node_idx, train_dataset in enumerate(train_data):
        node_state_dicts = []
        participating_node = []
        if len(train_dataset) > 0:
            print(f"Node {node_idx+1} Train dataset size: {len(train_dataset)}")
            print(f"Node {node_idx+1} Train dataset environment name: {node_envs[node_idx]}")
            Node.train(node_idx, model_parameter)
            participating_node.append(node_idx)
            node_state_dict = Node.model.state_dict()
            node_state_dicts.append(node_state_dict)
            
        else:
            pass

    print(f"Test dataset size: {len(test_data)}")
    