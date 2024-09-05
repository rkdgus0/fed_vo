import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from copy import deepcopy
import torch
import numpy as np
import random
from component.NODE import NODE

class SERVER():
    def __init__(self, model, nodes, NUM_NODE, test_data, iteration, avg_method='fedavg', num_node_data=None, device='cuda:0'):
        super(SERVER, self).__init__()
        self.model = deepcopy(model).to(device)
        self.NUM_NODE = NUM_NODE
        self.nodes = nodes
        self.test_data = test_data
        self.avg_method = avg_method
        self.iteration = iteration
        self.num_node_data = deepcopy(num_node_data)
        self.device = device

    # NUM_NODE 개수만큼 NODE를 선언해서 NODE.train으로 학습, 학습한 모델을 Average하는 코드
    #TODO 일단 구성에 문제는 없어보이고, node의 pose/flowpose/whole train 함수 짜기
    def train(self):
        model_parameter = self.model.state_dict()
        node_state_dicts = []
        participating_node = []

        #for train_type in ['whole']:
        for node_idx in range(self.NUM_NODE):
            self.nodes.train(node_idx, self.iteration, model_parameter)
            participating_node.append(node_idx)
            node_state_dict = self.nodes.model.state_dict()
            node_state_dicts.append(node_state_dict)

        avg_ratio = self.calc_avg_ratio(node_state_dicts, participating_node)
        #avg_ratio_tensor = torch.tensor(avg_ratio, dtype=torch.float32).view(-1, 1, 1, 1, 1)
        avg_model = self.average_model(node_state_dicts, avg_ratio)
        self.model.load_state_dict(avg_model)

        node_state_dicts.clear()
        avg_model.clear()
   
   #TODO Test dataset을 이용해서, Global Model의 성능을 확인하는 코드 추가 요망(evaluation code와 동일할 것으로 보임)
    def test(self):
        
        return

    # 모델 가중치 aggregation method(Fedavg: 데이터 수, Equal: 동등)
    def calc_avg_ratio(self, models, participating_node):
        ratio = []
        if self.avg_method == 'fedavg':
            for node_idx in participating_node:
                ratio.append(self.num_node_data[node_idx])
        elif self.avg_method == 'Equal':
            ratio = [1] * len(models)
        else:
            print("invalid global model average method!")
        
        ratio = [r / sum(ratio) for r in ratio]
        return ratio

    def set_lr(self, lr):
        for param_group in self.nodes.optimizer.param_groups:
            param_group['lr'] = lr

    def average_model(self, state_dicts, avg_ratio):
        if len(state_dicts) != len(avg_ratio):
            raise ValueError("The number of state_dicts and avg_ratio must be the same")

        keys = state_dicts[0].keys()
        for state_dict in state_dicts:
            if state_dict.keys() != keys:
                raise ValueError("All state_dicts must have the same keys")

        averaged_state_dict = {}
        for key in keys:
            weighted_sum = 0.0
            for i, state_dict in enumerate(state_dicts):
                weighted_sum += state_dict[key] * avg_ratio[i] / sum(avg_ratio)
            averaged_state_dict[key] = weighted_sum

        return averaged_state_dict

from Library.Datasets.dataset_util import ToTensor, Compose, CropCenter, DownscaleFlow, make_intrinsics_layer, dataset_intrinsics
from Library.Datasets.dataset import initial_dataset
from Network.VONet import VONet
from torch.optim.lr_scheduler import ExponentialLR
from time import time

if __name__ == '__main__':
    data_name = 'tartanair'
    root_dir = '/scratch/jeongeon/tartanAir'
    mode = 'easy'
    node_num = 16
    transform = Compose([CropCenter((640, 480)), DownscaleFlow(), ToTensor()])
    test_environments = ['ocean', 'zipfile', 'office2']
    # , 'soulcity', 'carwelding', 'abandonedfactory_night', 'seasonsforest', 'neighborhood', 'japanesealley', 'oldtown', 'carwelding', 'seasonsforest_winter', 'westerndesert', 'office', 'hospital', 'gascola', 'amusement'
    train_type = 'whole'
    iteration = 1
    batch_size = 16
    worker = 4
    avg_method = 'fedavg'
    global_round = 5
    
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(VONet())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    print("Init Dataset...")
    t1 = time()
    train_data, test_data, node_envs = initial_dataset(data_name, root_dir, mode, node_num, transform, test_environments)
    '''for i, train_dataset in enumerate(train_data):
        print(f"Node {i+1} Train dataset size: {len(train_dataset)}")
        print(f"Node {i+1} Train dataset environment name: {node_envs[i]}")'''
    t2 = time()
    print(f"Success to init dataset! Time(sec): {round(t2-t1,2)}\n")

    print("Init Node component...")
    t1 = time()
    Node =  NODE(model, scheduler, init_lr=0.01, datasets=train_data, iteration=iteration, batch_size=batch_size, worker_num=worker, device=device)
    num_node_data = []
    for node_idx, train_dataset in enumerate(train_data):
        num_node_data.append(len(train_dataset))
    Server = SERVER(model, Node, node_num, test_data, iteration, avg_method, num_node_data, device=device)
    t2 = time()
    print(f'Node num: {len(train_data)}')
    print(f"Success to init component! Time(sec): {round(t2-t1,2)}\n")

    for R in range(global_round):
        print(f"global round {R+1} Start!\n")
        Server.train()
    