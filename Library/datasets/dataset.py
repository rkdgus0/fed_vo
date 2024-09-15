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
        self.sequence = sequence
        self.motions = []
        self.image_files = []
        self.flow_files = []
        self.sequence_starts = []
        self.flownorm = 20.0
        
        self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics(self.data_name)
        #self.focalx, self.focaly, self.centerx, self.centery = 320.0, 320.0, 320.0, 240.0
        
        if easy_hard.lower() == 'easy':
            self.easy_hard = 'Easy'
        elif easy_hard.lower() == 'hard':
            self.easy_hard = 'Hard'
        else:
            self.easy_hard = '*'
        
        if environments is None:
            environments = os.listdir(root_dir)
            print("Invalid Input Environment!")
        else:
            environments = [env for env in environments if os.path.exists(os.path.join(root_dir, env))]

        for environment in environments:
            env_path = os.path.join(root_dir, environment, self.easy_hard, self.sequence)
            image_dirs = glob.glob(os.path.join(env_path, 'image_left'))
            pose_files = glob.glob(os.path.join(env_path, 'pose_left.txt'))
            flow_dirs = glob.glob(os.path.join(env_path, 'flow'))

            for flow_dir, image_dir, pose_file in zip(flow_dirs,image_dirs, pose_files):
                image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
                flow_files = sorted(glob.glob(os.path.join(flow_dir, '*flow.npy')))
                start_idx = len(self.image_files)  # 시퀀스 시작 위치
                self.sequence_starts.append(start_idx)
                self.image_files.extend(image_files)
                self.flow_files.extend(flow_files)
                self.flow_files.append(flow_files[-1]) #더미 값 추가

                # pose list to motion and matrix
                poselist = np.loadtxt(pose_file).astype(np.float32)
                assert(poselist.shape[1]==7)
                poses = pos_quats2SEs(poselist)
                matrix = pose2motion(poses)
                motions = SEs2ses(matrix).astype(np.float32)
                self.motions.extend(motions)
                self.motions.append(motions[-1]) #더미 값 추가
                assert(len(self.motions) == len(self.image_files))
                assert(len(self.motions) == len(self.flow_files))
                

                
            #print(f"[{environment}] motion len: {len(self.motions)}, img len: {len(self.image_files)}")
        
    def __len__(self):
        return max(len(self.image_files)-1, 0)

    def __getitem__(self, idx):
        if idx in self.sequence_starts and idx > 0:
            idx += 1
        img_path1 = self.image_files[idx].strip()
        img_path2 = self.image_files[idx+1].strip()

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        
        flowfile = self.flow_files[idx].strip()
        flow = np.load(flowfile) / self.flownorm

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        
        res = {'img1': img1, 'img2': img2, 'intrinsic': intrinsicLayer, 'flow': flow}
        
        if self.transform:
            res = self.transform(res)
        
        if self.motions is None:
            return res
        
        else:
            res['motion'] = self.motions[idx]
            return res
        
        

def initial_dataset(data_name, root_dir, easy_hard, sequence, node_num, transform, test_environments=['ocean']):
    """
    Returns:
        tuple: (train_dataset, test_dataset)으로 분할된 데이터셋
        train_dataset: node 개수만큼의 2차원 list
        각 node_dataset: 
    """
    
    # test dataset
    test_dataset = TartanAirDataset(data_name=data_name, root_dir=root_dir, easy_hard=easy_hard, sequence='P001', environments=test_environments, transform=transform)
    
    # train dataset
    # node에 순차적으로 배정
    all_environments = os.listdir(root_dir)
    train_environments = [env for env in all_environments if env not in set(test_environments)]
    train_datasets = [[] for _ in range(node_num)]
    node_env_mapping = [[] for _ in range(node_num)]
    
    # 환경 개수가 node 수보다 적으면 에러 반출
    if len(train_environments) < node_num:
        raise ValueError("The number of environment is little than the number of node!")
    
    for i, environment in enumerate(train_environments):
        node_index = i % node_num
        node_env_mapping[node_index].append(environment)
    
    train_datasets = [TartanAirDataset(data_name=data_name, root_dir=root_dir, easy_hard=easy_hard, sequence=sequence, environments=node_envs, transform=transform)
                      for node_envs in node_env_mapping if node_envs]
    
    print("[Test Dataset]")
    print(f"  Number of Test Environment: {len(test_environments)}")
    print(f"  [Server] Number of Data: {len(test_dataset.motions)}")
    print("[Train Dataset]")
    print(f"  Number of Train Environment: {len(train_environments)}")
    for i, environment in enumerate(train_environments):
        node_index = i % node_num
        print(f"  [Node {i+1}] Env: {environment}, Number of Data: {len(train_datasets[node_index].motions)}")
    
    return train_datasets, test_dataset, node_env_mapping

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
