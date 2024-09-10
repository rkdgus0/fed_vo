import os
import sys
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import random
import torch
import numpy as np

from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

from Network.VONet import VONet
from Network.PWC import PWCDCNet as FlowNet
from Network.VOFlowNet import VOFlowRes as FlowPoseNet

from Library.component.NODE import NODE
from Library.component.SERVER import SERVER

# ===== SERVER/NODE Compose function =====
def compose_server(args, model, nodes, test_data, train_data, device):
    NUM_NODE = args.node_num
    avg_method = args.avg_method
    batch_size = args.batch_size
    worker_num = args.worker_num
    
    num_node_data = []
    for node_idx, train_dataset in enumerate(train_data):
        num_node_data.append(len(train_dataset))

    return SERVER(model, nodes, NUM_NODE, test_data, avg_method, 
                  num_node_data, batch_size, worker_num, device)

def compose_node(args, model, optimizer, scheduler, train_data, device):
    init_lr = args.learning_rate
    iteration = args.local_iteration
    batch_size = args.batch_size
    worker_num = args.worker_num

    return NODE(model, optimizer, scheduler, init_lr, train_data,  
                iteration, batch_size, worker_num, device)

#FIXME 현재 코드 상, lambda_controller가 작동하지 않을 것.
def lambda_controller(args, current_round):
        if current_round < 0.5 * args.globla_round:
            return 1.0
        elif current_round < 0.875 * args.globla_round:
            return 0.2
        else:
            return 0.04

def init_model(model_name, optimizer, lr):
    if model_name.lower() == 'vonet':
        model = torch.nn.DataParallel(VONet())
    elif model_name.lower() == 'flownet' or model_name.lower() == 'matchingnet':
        model = torch.nn.DataParallel(FlowNet())
    elif model_name.lower() == 'flowposenet' or model_name.lower() == 'posenet':
        model = torch.nn.DataParallel(FlowPoseNet())
    
    if optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == 'sgd':
        optimizer = torch.optim.sgd(model.parameters(), lr=lr)
    
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    return model, optimizer, scheduler

# ===== Model Parameter Save/Load function =====
def save_checkpoint(model, optimizer, scheduler, global_round, iteration, filepath):
    torch.save({
        'global_round': global_round,
        'local_iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, filepath)

def load_checkpoint(model, optimizer=None, scheduler=None, filepath="",map_location='cuda:0'):
    if filepath=="":
        return 0
    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    iteration = checkpoint['iteration']
    print(f"successfully load model from {filepath}")
    return iteration

# ===== Test Trajectory Plot function =====
def plot_traj(gtposes, estposes, vis=False, savefigname=None, title=''):
    fig = plt.figure(figsize=(4,4))
    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    plt.plot(gtposes[:,0],gtposes[:,1], linestyle='dashed',c='k')
    plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth','TartanVO'])
    plt.title(title)
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)

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