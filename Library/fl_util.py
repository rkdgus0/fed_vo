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
from Network.VOFlowNet import VOFlowRes as PoseNet

from Library.component.NODE import NODE
from Library.component.SERVER import SERVER

# ===== SERVER/NODE Compose function =====
def compose_server(args, model, nodes, test_data, train_data, device):
    NUM_NODE = args.node_num
    avg_method = args.avg_method
    batch_size = args.batch_size
    worker_num = args.worker_num
    test_data_name = args.test_data_name
    if args.easy_hard.lower() == 'hard':
        easy_hard = 'Hard'
    else:
        easy_hard = 'Easy'
    test_data_path = f"{args.test_data_path}/pose_left.txt"

    num_node_data = []
    for node_idx, train_dataset in enumerate(train_data):
        num_node_data.append(len(train_dataset))

    return SERVER(model, nodes, NUM_NODE, test_data_name, test_data, test_data_path, 
                  avg_method, num_node_data, batch_size, worker_num, device)

def compose_node(args, model, optimizer, scheduler, train_data, device):
    epoch = args.local_epoch
    batch_size = args.batch_size
    worker_num = args.worker_num

    return NODE(model, optimizer, scheduler, train_data,  
                epoch, batch_size, worker_num, device)

#FIXME 현재 코드 상, lambda_controller가 작동하지 않을 것.
def lr_lambda(iteration):
    if iteration < 0.5 * total_iterations:
        return 1.0
    elif iteration < 0.875 * total_iterations:
        return 0.2
    else:
        return 0.04

def init_model(model_name, optimizer, lr, device, model_path=None):
    if model_name.lower() == 'vonet':
        model = torch.nn.DataParallel(VONet())
    elif model_name.lower() == 'flownet' or model_name.lower() == 'matchingnet':
        #model = torch.nn.DataParallel(FlowNet())
        model = pwc_dc_net('data/pwc_net_chairs.pth.tar')
        model = torch.nn.DataParallel(model)
    elif model_name.lower() == 'flowposenet' or model_name.lower() == 'posenet':
        model = torch.nn.DataParallel(PoseNet())
    
    if optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == 'sgd':
        optimizer = torch.optim.sgd(model.parameters(), lr=lr)
    
    scheduler = ExponentialLR(optimizer, gamma=1.0)
    load_checkpoint(model, model_path, device)
    return model, optimizer, scheduler

# ===== Model Parameter Save/Load function =====
def save_checkpoint(model, optimizer, scheduler, global_round, epoch, model_path):
    torch.save({
        'global_round': global_round,
        'local_epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, model_path)

'''def load_checkpoint(model, optimizer, scheduler, model_path, map_location):
    if model_path==None:
        return 0
    checkpoint = torch.load(model_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    global_round = checkpoint['global_round']
    epoch = checkpoint['local_epoch']
    print(f"successfully load model from {model_path}")
    return epoch'''

def load_checkpoint(model, model_path, device):
    if model_path is None:
        raise ValueError("Invalid File Path!")
    checkpoint = torch.load(model_path, map_location=device)

    # Partial model update
    model_state_dict = model.state_dict()
    pretrained_model = checkpoint['model_state_dict']

    model_dict = {k: v for k, v in pretrained_model.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
    model_state_dict.update(model_dict)
    
    model.load_state_dict(model_state_dict)

    print(f"Successfully loaded model from {model_path}")

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

