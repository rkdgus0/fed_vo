import os
import sys
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

from Network.VONet import VONet
from Network.PWC import PWCDCNet as FlowNet
from Network.PWC import pwc_dc_net
from Network.VOFlowNet import VOFlowRes as FlowPoseNet

from Library.component.NODE import NODE
from Library.component.SERVER import SERVER

# ===== SERVER/NODE Compose function =====
def compose_server(args, model, nodes, test_data, train_data, device):
    NUM_NODE = args.node_num
    avg_method = args.avg_method
    batch_size = args.batch_size
    worker_num = args.worker_num
    if args.easy_hard.lower() == 'hard':
        easy_hard = 'Hard'
    else:
        easy_hard = 'Easy'
    test_data_path = f"{args.data_path}/ocean/{easy_hard}/P001/pose_left.txt"

    
    num_node_data = []
    for node_idx, train_dataset in enumerate(train_data):
        num_node_data.append(len(train_dataset))

    return SERVER(model, nodes, NUM_NODE, test_data, test_data_path, avg_method, 
                  num_node_data, batch_size, worker_num, device)

def compose_node(args, model, optimizer, scheduler, train_data, device):
    epoch = args.local_epoch
    batch_size = args.batch_size
    worker_num = args.worker_num

    return NODE(model, optimizer, scheduler, train_data,  
                epoch, batch_size, worker_num, device)

#FIXME 현재 코드 상, lambda_controller가 작동하지 않을 것.
def lambda_controller(global_round, current_round):
        if current_round < 0.5 * globla_round:
            return 1.0
        elif current_round < 0.875 * globla_round:
            return 0.2
        else:
            return 0.04

def init_model(model_name, optimizer, lr):
    if model_name.lower() == 'vonet':
        model = torch.nn.DataParallel(VONet())
    elif model_name.lower() == 'flownet' or model_name.lower() == 'matchingnet':
        #model = torch.nn.DataParallel(FlowNet())
        model = pwc_dc_net('data/pwc_net_chairs.pth.tar')
        model = torch.nn.DataParallel(model)
    elif model_name.lower() == 'flowposenet' or model_name.lower() == 'posenet':
        model = torch.nn.DataParallel(FlowPoseNet())
    
    if optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == 'sgd':
        optimizer = torch.optim.sgd(model.parameters(), lr=lr)
    
    scheduler = ExponentialLR(optimizer, gamma=0.998)

    return model, optimizer, scheduler

# ===== Model Parameter Save/Load function =====
def save_checkpoint(model, optimizer, scheduler, global_round, epoch, filepath):
    torch.save({
        'global_round': global_round,
        'local_epoch': epoch,
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
    epoch = checkpoint['epoch']
    print(f"successfully load model from {filepath}")
    return epoch

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

