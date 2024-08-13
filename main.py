import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanDataset import TartanDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO
from utils.train_utils import load_checkpoint, save_checkpoint, train_pose_batch
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='C_VO with FL')
'''
    parser.add_argument('--worker_num', type=int, default=16,
                        help='data loader worker number (default: 16)')
    # Check the Size of Image(KITTI: W1226/H370, EuRoC_V102: W752/H480), 크기 달라도 상관X
    parser.add_argument('--image_width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image_height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--model_name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--kitti_intrinsics_file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test_dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose_file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save_flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
'''
    # ===== FL Setting ======
    # exp_name: Model의 weight와 학습 이후 Summary log 저장 path 설정 
    parser.add_argument('--exp_name', '-exp', type=str, default='Test',
                        help="Saving Path for weight&Summary (default: Test)")
    # worker_num: dataset을 GPU로 넘기는 과정에서의 preprocessing을 하는 subprocess 수(많을수록 빠르나, 자원 차지가 많아짐)
    parser.add_argument('--worker_num', '-worker', type=int, default=16,
                        help='data loader worker number (default: 16)')
    # node_num: 연합학습(FL)에서 학습에 참여하는 Node의 수
    parser.add_argument('--node_num', '-node', type=int, default=1,
                        help='number of nodes, different from worker_num (default: 1)')
    parser.add_argument('--avg_method', '-avg', type=str, default='fedavg',
                        help='average method (Select: fedavg, equal), (default: fedavg)')

    # ===== Model Setting =====
    # model: Pretrained된 model weight 저장 경로, .pkl 파일로 받음
    parser.add_argument('--model', '-model', type=str, default='',
                        help="name of pretrained model (default: '')")
    # batch_size: model 학습과 evaluation에서 사용할 data 개수 조정
    parser.add_argument('--batch_size', '-batch', type=int, default=1,
                        help='batch size (default: 1)')
    # global_round: 전체 학습 round
    parser.add_argument('--global_round', '-round', type=int, default=100000,
                        help='total number of Global round (default: 1)')
    # local_iteration: 각 node의 iteration 횟수
    parser.add_argument('--local_epoch', '-epoch', type=int, default=1,
                        help='total number of local iteration per each node (default: 1)')
    # learning_rate: model의 learning rate
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                        help='model learning rate (default: 1e-4)')
    
    # ====== Dataset Setting ======
    # dataset: train과 test에 사용할 dataset
    parser.add_argument('--dataset', '-dataset', type=str, default='KITTI',
                        help="Dataset Name (select: KITTI, EuroC, TartanAir), (default: KITTI)")
    # image_width: 이미지 크기 조정에 사용할 parameter
    parser.add_argument('--image_width', '-img_w', type=int, default=640,
                        help='image width (select: multiple of 64), (default: 640)')
    # image_height: 이미지 크기 조정에 사용할 parameter
    parser.add_argument('--image_height', '-img_h', type=int, default=448,
                        help='image height (select: multiple of 64), (default: 448)')

    args = parser.parse_args()
    return args

def lr_lambda(current_round):
        if current_round < 0.5 * args.globla_round:
            return 1.0
        elif current_round < 0.875 * args.globla_round:
            return 0.2
        else:
            return 0.04
    
if __name__ == '__main__':

    args = get_args()

    # load trajectory data from a folder
    datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics('tartanair')
    with open('/root/volume/code/python/tartanvo/data/pose_left_paths.txt', 'r') as f:
        posefiles = f.readlines()

    iteration = 0 
    num_epochs = 100
    learning_rate = 1e-4
    total_iterations = 100000
    
    model = TartanVO()
    model = torch.nn.DataParallel(model.vonet)
    optimizer = torch.optim.Adam(model.module.flowPoseNet.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)
    summaryWriter = SummaryWriter('runs/exp1')
    start_epoch,iteration = load_checkpoint(model, optimizer, scheduler, args.model_name)

    for epoch in range(num_epochs):
        for posefile in posefiles:
            posefile = posefile.strip()
            print(posefile)
            transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
            trainDataset = TartanDataset( posefile = posefile, transform=transform, 
                                                focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
            
            trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.worker_num)
            trainDataiter = iter(trainDataloader)

            for batch_idx, sample in enumerate(trainDataiter):
                total_loss,flow_loss,pose_loss,trans_loss,rot_loss = train_pose_batch(model.vonet, optimizer, sample)
                iteration += 1
                scheduler.step()
                if iteration % 10 == 0:
                    summaryWriter.add_scalar('Loss/train_total', total_loss, iteration)
                    summaryWriter.add_scalar('Loss/train_flow', flow_loss, iteration)
                    summaryWriter.add_scalar('Loss/train_pose', pose_loss, iteration)
                    summaryWriter.add_scalar('Loss/train_trans', trans_loss, iteration)
                    summaryWriter.add_scalar('Loss/train_rot', rot_loss, iteration)
                    print(f"Epoch {epoch + 1}, Step {iteration}, Loss: {total_loss}")
                    print(f"Flow Loss: {flow_loss}")
                    print(f"Pose Loss: {pose_loss}")

        model_save_path = f'models/model_iteration_{iteration}.pth'
        save_checkpoint(model.vonet, optimizer, scheduler, epoch, iteration, model_save_path)


        



