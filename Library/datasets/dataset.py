import os
import sys
import glob
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets.dataset_util import ToTensor, Compose, CropCenter, DownscaleFlow, make_intrinsics_layer, dataset_intrinsics
from datasets.transformation import pos_quats2SEs, pose2motion, SEs2ses

# root_dir (str): 데이터셋의 루트 디렉토리 경로
# environments (list of str): 사용할 환경 이름 리스트 (ex: ['ocean', 'amusement'])
# test_environments (list of str): 테스트 세트로 사용할 라벨 리스트
# transform (callable, optional): 이미지에 적용할 변환 함수
# easy_hard(easy, hard, both): 데이터 사용 방법 (Easy/Hard/Both of Easy and Hard)
class TartanAirDataset(Dataset):
    def __init__(self, data_name, root_dir, easy_hard, sequence, environments=None, transform=None):
        self.data_name = data_name
        self.root_dir = root_dir
        self.transform = transform
        self.easy_hard = easy_hard
        self.sequence = sequence
        self.motions = []
        self.image_files = []
        self.flow_files = []
        self.sequence_starts = []
        self.flownorm = 20.0
        self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics(self.data_name)

        if environments is not None:
            environments = [env for env in environments if os.path.exists(os.path.join(root_dir, env))]

        if data_name.lower() != 'tartanair':
            image_dirs = glob.glob(os.path.join(root_dir, 'image_left'))
            pose_files = glob.glob(os.path.join(root_dir, 'pose_left.txt'))
            flow_dir = os.path.join(root_dir, 'flow') if os.path.exists(os.path.join(root_dir, 'flow')) else None

            for image_dir, pose_file in zip(image_dirs, pose_files):
                self.load_data_from_dirs(image_dir, pose_file, flow_dir)

        else:
            for environment in environments:
                env_path = os.path.join(root_dir, environment, self.easy_hard, self.sequence)
                image_dirs = glob.glob(os.path.join(env_path, 'image_left'))
                pose_files = glob.glob(os.path.join(env_path, 'pose_left.txt'))
                flow_dirs = glob.glob(os.path.join(env_path, 'flow'))

                for image_dir, pose_file, flow_dir in zip(image_dirs, pose_files, flow_dirs):
                    self.load_data_from_dirs(image_dir, pose_file, flow_dir)

    def load_data_from_dirs(self, image_dir, pose_file, flow_dir=None):
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        start_idx = len(self.image_files)  # 시퀀스 시작 위치
        self.sequence_starts.append(start_idx)
        self.image_files.extend(image_files)

        if flow_dir:
            flow_files = sorted(glob.glob(os.path.join(flow_dir, '*flow.npy')))
            self.flow_files.extend(flow_files)
            self.flow_files.append(flow_files[-1])  # 더미 값 추가

        # pose list to motion and matrix
        self.poselist = np.loadtxt(pose_file).astype(np.float32)
        assert self.poselist.shape[1] == 7
        poses = pos_quats2SEs(self.poselist)
        matrix = pose2motion(poses)
        motions = SEs2ses(matrix).astype(np.float32)
        self.motions.extend(motions)
        self.motions.append(motions[-1])  # 더미 값 추가

        assert len(self.motions) == len(self.image_files)

        
    def __len__(self):
        return max(len(self.image_files)-1, 0)

    def __getitem__(self, idx):
        if idx in self.sequence_starts and idx > 0:
            idx += 1
        img_path1 = self.image_files[idx].strip()
        img_path2 = self.image_files[idx+1].strip()

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        
        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        
        res = {'img1': img1, 'img2': img2, 'intrinsic': intrinsicLayer}
        
        if len(self.flow_files) != 0:
            flowfile = self.flow_files[idx].strip()
            flow = np.load(flowfile) / self.flownorm
            res['flow'] = flow

        if self.transform:
            res = self.transform(res)
        
        if self.motions is None:
            return res
        
        else:
            res['motion'] = self.motions[idx]
            return res
        
        

def initial_dataset(train_data_name, test_data_name, train_dir, test_dir, easy_hard, sequence, node_num, transform, test_environments=['ocean'], split_mode='basic'):
    """
    Returns:
        tuple: (train_dataset, test_dataset)으로 분할된 데이터셋
        train_dataset: node 개수만큼의 2차원 list
        각 node_dataset: 
    """
    
    # test dataset
    if easy_hard.lower() == 'easy':
        easy_hard = 'Easy'
        test_eh = 'Easy'
    elif easy_hard.lower() == 'hard':
        easy_hard = 'Hard'
        test_eh = 'Hard'
    else:
        easy_hard = '*'
        test_eh = 'Easy'
    test_dataset = TartanAirDataset(data_name=test_data_name, root_dir=test_dir, easy_hard=test_eh, sequence='P001', environments=test_environments, transform=transform)
    
    # train dataset
    # node에 순차적으로 배정
    all_environments = os.listdir(train_dir)
    if split_mode == 'basic':
        train_environments = [env for env in all_environments if env not in set(test_environments)]
    elif split_mode == 'all':
        train_environments = [env for env in all_environments]
    elif split_mode == 'test':
        train_environments = test_environments
    train_datasets = [[] for _ in range(node_num)]
    node_env_mapping = [[] for _ in range(node_num)]
    
    # 환경 개수가 node 수보다 적으면 에러 반출
    if len(train_environments) < node_num:
        raise ValueError("The number of environment is little than the number of node!")
    
    for i, environment in enumerate(train_environments):
        node_index = i % node_num
        node_env_mapping[node_index].append(environment)
    
    train_datasets = [TartanAirDataset(data_name=train_data_name, root_dir=train_dir, easy_hard=easy_hard, sequence=sequence, environments=node_envs, transform=transform)
                      for node_envs in node_env_mapping if node_envs]
    
    print("[Test Dataset]")
    if test_data_name.lower() == 'tartanair':
        print(f"  Number of Test Environment: {len(test_environments)}")
    
    print(f"  [Server] Number of Data: {len(test_dataset.motions)}")
    print("[Train Dataset]")
    print(f"  Number of Train Environment: {len(train_environments)}")
    total_num_data = 0
    for node_idx in range(node_num):
        node_num_data = 0
        node_env = node_env_mapping[node_idx]
        for i, environment in enumerate(node_env):
            node_num_data += len(node_env[i])
        total_num_data += node_num_data
        print(f"  [Node {i+1}] Env: {node_env_mapping[node_idx]}, Number of Data: {node_num_data}")
    print(f"  [Node] Number of Train Environment data: {total_num_data}")

    return train_datasets, test_dataset, total_num_data

#테스트용
if __name__ == '__main__':

    data_name = 'tartanair'
    root_dir = '/scratch/jeongeon/tartanAir'
    easy_hard = 'easy'
    node_num = 17
    transform = Compose([CropCenter((640, 480)), DownscaleFlow(), ToTensor()])
    test_environments = ['ocean','zipfile']
    
    print("Init Datasets...")
    train, test, env = initial_dataset(data_name, root_dir, easy_hard, node_num, transform, test_environments)
    print("Success to Init Datasets...")
    print(f'Node num: {len(train)}')
    for i, train_dataset in enumerate(train):
        print(f"Node {i+1} Train dataset size: {len(train_dataset)}")
        print(f"Node {i+1} Train dataset environment name: {env[i]}")

    print(f"Test dataset size: {len(test)}")
    print(f"Test dataset environment name: {test_environments}")
