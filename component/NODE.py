import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from Library.train_util import process_pose_sample
from copy import deepcopy

class NODE():
    def __init__(self, model, scheduler, init_lr, datasets, iteration, batch_size, worker_num, device='cuda:0'):
        super(NODE, self).__init__()
        self.datasets = datasets
        self.iteration = iteration
        self.batch_size = batch_size
        self.worker_num = worker_num
        self.device = device
        self.model = deepcopy(model).to(device)
        self.optimizer = torch.optim.Adam(model.module.flowPoseNet.parameters(), lr=init_lr)
        self.scheduler = deepcopy(scheduler)
        self.criterion = nn.CrossEntropyLoss()
        

    def train(self, node_idx, mode, iteration, model_parameters):
        self.model.load_state_dict(model_parameters) #global model

        # setting node's train data
        train_data = self.datasets[node_idx]
        trainDataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.worker_num)
        #trainDataiter = iter(trainDataloader)

        for curr_iter in range(iteration):
            print(f"Node {node_idx}, Iteration {curr_iter+1}/{self.iteration} Start!")
            for sample in trainDataloader:
                self.model.train()
                self.optimizer.zero_grad()

                if mode == 'whole':
                    total_loss,trans_loss,rot_loss = process_pose_sample(self.model, sample, self.device)
                    total_loss.backward()
                    
                    # print(f"total loss: {total_loss}, trans loss: {trans_loss}, rot loss: {rot_loss}")

                self.optimizer.step()
                self.scheduler.step()
        
            print(f"Node {node_idx}, Iteration {curr_iter+1}/{self.iteration}, Loss: {total_loss.item()}")

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


from Library.Datasets.dataset_util import ToTensor, Compose, CropCenter, DownscaleFlow, make_intrinsics_layer, dataset_intrinsics
from Library.Datasets.dataset import initial_dataset
from Network.VONet import VONet
from torch.optim.lr_scheduler import LambdaLR


if __name__ == '__main__':
    data_name = 'tartanair'
    root_dir = '/scratch/jeongeon/tartanAir'
    mode = 'easy'
    node_num = 17
    transform = Compose([CropCenter((640, 480)), DownscaleFlow(), ToTensor()])
    test_environments = ['ocean', 'zipfile']
    train_type = 'whole'
    iteration = 3

    def lambda_controller(current_round):
        if current_round < 0.5 * 100:
            return 1.0
        elif current_round < 0.875 * 100:
            return 0.2
        else:
            return 0.04
        
    model = torch.nn.DataParallel(VONet())
    optimizer = torch.optim.Adam(model.module.flowPoseNet.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_controller)

    print("Init Dataset...")
    train_data, test_data, node_envs = initial_dataset(data_name, root_dir, mode, node_num, transform, test_environments)
    print("Success to init dataset!\n")

    print("Init model parameter...")
    #Node = compose_node(args, model, optimizer, scheduler, train_data)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_parameter = model.state_dict()
    print("Success to init model parameter!\n")

    print("Init Node component...")
    Node =  NODE(model, scheduler, init_lr=0.01, datasets=train_data, iteration=iteration, batch_size=4, worker_num=1, device=device)

    '''print(f'Node num: {len(train_data)}')
    for node_idx, train_dataset in enumerate(train_data):
        print(f"Node {node_idx+1} Train dataset size: {len(train_dataset)}")
        print(f"Node {node_idx+1} Train dataset environment name: {node_envs[node_idx]}")
        Node.train(node_idx, train_type, iteration, model_parameter)

    print(f"Test dataset size: {len(test_data)}")'''
    