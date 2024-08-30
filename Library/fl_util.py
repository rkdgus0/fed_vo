import os
import sys
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import random
import torch
import numpy as np

from torch.optim.lr_scheduler import LambdaLR

from Network.VONet import VONet
from component.NODE import NODE
from component.SERVER import SERVER

# ===== SERVER/NODE Compose function =====
def compose_server(args, model, nodes, test_data, train_data):
    NUM_NODE = args.node_num
    avg_method = args.avg_method
    iteration = args.local_iteration
    num_node_data = []
    for node_idx in range(len(train_data)):
        num_node_data.append = len(train_data[node_idx])

    return SERVER(model, nodes, NUM_NODE, test_data, iteration, avg_method, num_node_data)

def compose_node(args, model, scheduler, splited_datasets):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return NODE(model, scheduler, init_lr=args.learning_rate, datasets=splited_datasets, iteration=args.local_iteration, 
                batch_size=args.batch_size, worker_num=args.worker_num, device=device)

#FIXME 현재 코드 상, lambda_controller가 작동하지 않을 것.
def lambda_controller(args, current_round):
        if current_round < 0.5 * args.globla_round:
            return 1.0
        elif current_round < 0.875 * args.globla_round:
            return 0.2
        else:
            return 0.04

def init_model(optimizer, lr):
    model = torch.nn.DataParallel(VONet())
    if optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.module.flowPoseNet.parameters(), lr=lr)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_controller)
    return model, optimizer, scheduler


# ===== Dataset Preprocessing =====
# 폐기. Datasets/dataset.py에서 데이터셋 정리함
'''
def initial_dataset(dataset, node_num):
    train_index =[[] for _ in range(node_num)]

    # dataset information
    focalx, focaly, centerx, centery = dataset_intrinsics(dataset.lower())

    # dataset load path
    if dataset.lower() == 'tartanair':
        # 정확한 path 입력하기
        path = '/home/data/jeongeon/tartanair'
        # tartanair의 
        train_class = ['ocean', '']
        test_class = ['a', 'b']
    elif dataset.lower() == '~~~~':
        path = '/root/volume/code/python/tartanvo/data/pose_left_paths.txt'
    
    # split dataset
    # (모두가 동등한 수의 클래스를 가진 데이터셋;non-iid)
    #TODO 모든 클래스의 데이터를 일부분만 가지는 데이터셋도 구성하기
    mod = len(train_class) % node_num
    for i in range(node_num-1):
        train_index[i] = train_class[i * mod : (i+1) * mod]
    
    train_index[node_num] = train_class[node_num * mod :]

    #XXX 해당 path에는 여러 클래스가 존재함. 
    # 해당 path 아래에는 해당 클래스 이름으로 구성된 폴더명/(EASY or HARD)/pose_left_paths.txt에서 pose를 알 수 있음
    # 19개의 클래스를 TestSet과 TrainSet(각 node의 SubDataset)으로 구성해야함
    #XXX 클래스명을 split한 뒤에, Path와 함께 return해서 train code에서 데이터 불러와서 돌리는 함수로 구성하는 것으로 생각중

    return path, test_index, train_index'''