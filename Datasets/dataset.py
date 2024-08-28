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
from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, make_intrinsics_layer
from Datasets.transformation import pos_quats2SEs, pose2motion, SEs2ses

# root_dir (str): 데이터셋의 루트 디렉토리 경로
# environments (list of str): 사용할 환경 이름 리스트 (ex: ['ocean', 'amusement'])
# test_environments (list of str): 테스트 세트로 사용할 라벨 리스트
# transform (callable, optional): 이미지에 적용할 변환 함수
# mode(easy, hard, both): 데이터 사용 방법 (Easy/Hard/Both of Easy and Hard)
class TartanAirDataset(Dataset):
    def __init__(self, root_dir, mode, environments=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = {}
        
        #self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics('tartanair')
        self.focalx, self.focaly, self.centerx, self.centery = 320.0, 320.0, 320.0, 240.0
        
        if mode.lower() == 'easy':
            self.mode = 'Easy'
        elif mode.lower() == 'hard':
            self.mode = 'Hard'
        else:
            self.mode = '*'
        
        if environments is None:
            environments = os.listdir(root_dir)
        else:
            environments = [env for env in environments if os.path.exists(os.path.join(root_dir, env))]

        for environment in environments:
            env_path = os.path.join(root_dir, environment)
            image_dirs = glob.glob(os.path.join(env_path, self.mode, '*/image_left'))
            pose_files = glob.glob(os.path.join(env_path, self.mode, '*/pose_left.txt'))

            self.data[environment] = {'images': [], 'motions': []}
            
            for image_dir, pose_file in zip(image_dirs, pose_files):
                image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
                poselist = np.loadtxt(pose_file).astype(np.float32)
                assert(poselist.shape[1] == 7)
                
                poses = pos_quats2SEs(poselist)
                matrix = pose2motion(poses)
                motions = SEs2ses(matrix).astype(np.float32)
                
                self.data[environment]['images'].extend(image_files)
                self.data[environment]['motions'].extend(motions)
        
    def __len__(self):
        return sum(len(data['images']) - 1 for data in self.data.values())

    #XXX 환경별로 따로 traindataset을 가져야하나?
    def __getitem__(self, idx):
        env_names = list(self.data.keys())
        total_images = 0
        for env in env_names:
            num_images = len(self.data[env]['images']) - 1
            if total_images + num_images >= idx:
                env = env
                local_idx = idx - total_images
                break
            total_images += num_images
        
        img_path1 = self.data[env]['images'][local_idx].strip()
        img_path2 = self.data[env]['images'][local_idx + 1].strip()
        motion = self.data[env]['motions'][local_idx]
        
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        
        res = {'img1': img1, 'img2': img2, 'motion': motion, 'intrinsic': intrinsicLayer}
        
        if self.transform:
            res = self.transform(res)
        
        return res

def initial_dataset(root_dir, mode, node_num, transform, test_environments=['ocean', 'amusement']):
    """
    Returns:
        tuple: (train_dataset, test_dataset)으로 분할된 데이터셋
        train_dataset: node 개수만큼의 2차원 list
        각 node_dataset: 
    """
    
    # test dataset
    test_dataset = TartanAirDataset(root_dir=root_dir, mode=mode, environments=test_environments, transform=transform)
    
    # train dataset
    # node에 순차적으로 배정(환견 개수가 node 수보다 적으면 에러 반출)
    all_environments = os.listdir(root_dir)
    train_environments = [env for env in all_environments if env not in test_environments]
    train_datasets = [[] for _ in range(node_num)]
    node_env_mapping = [[] for _ in range(node_num)]
    
    if len(train_environments) < node_num:
        raise ValueError("The number of environment is little than the number of node!")
    
    for i, environment in enumerate(train_environments):
        node_index = i % node_num
        node_env_mapping[node_index].append(environment)
    
    train_datasets = [TartanAirDataset(root_dir=root_dir, mode=mode, environments=node_envs, transform=transform)
                      for node_envs in node_env_mapping if node_envs]
    
    return train_datasets, test_dataset, node_env_mapping

#테스트용
if __name__ == '__main__':

    root_dir = '../data/tartanAir'
    mode = 'both'
    node_num = 2
    transform = Compose([CropCenter((640, 480)), DownscaleFlow(), ToTensor()])
    test_environments = ['ocean']

    train, test, env = initial_dataset(root_dir, mode, node_num, transform, test_environments)
    
    for i, train_dataset in enumerate(train):
        print(f"Node {i+1} Train dataset size: {len(train_dataset)}")
        print(f"Node {i+1} Train dataset environment name: {env[i]}")

    print(f"Test dataset size: {len(test)}")
