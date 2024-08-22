import random
import torch
import numpy as np

from component.NODE import NODE
from component.SERVER import SERVER
from sklearn.utils import shuffle

# ===== SERVER/NODE Compose function =====
def compose_server(args, model, nodes, test_data):
    NUM_NODE = args.node_num
    avg_method = args.avg_method
    num_node_data = []
    #TODO num_node_data를 받아오는 코드 구성(형태: [node1_data_num, node2_data_num,...])

    return SERVER(model, nodes, NUM_NODE, test_data, avg_method, num_node_data)

def compose_node(args, model, scheduler, splited_datasets):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return NODE(model, scheduler, init_lr=args.learning_rate, datasets=splited_datasets, epochs=args.n_epochs, 
                batch_size=args.batch_size, device=device)


# ===== Dataset Preprocessing =====
def dataset_intrinsics(dataset='tartanair'):
    if dataset == 'kitti':
        focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
    elif dataset == 'euroc':
        focalx, focaly, centerx, centery = 458.6539916992, 457.2959899902, 367.2149963379, 248.3750000000
    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0
    else:
        return None
    return focalx, focaly, centerx, centery

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

    return path, test_index, train_index