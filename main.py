import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Library.Datasets.dataset_util import ToTensor, Compose, CropCenter, DownscaleFlow
from Library.Datasets.dataset import initial_dataset
#from Datasets.transformation import ses2poses_quat
#from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO
#from utils.train_utils import load_checkpoint, save_checkpoint, train_pose_batch

from Network.VONet import VONet
from Library.fl_util import compose_node, compose_server, init_model
from Library.train_util import save_checkpoint
import argparse

#TODO Parser 파라미터 변경해야할 수 있음.
def get_args():
    parser = argparse.ArgumentParser(description='C_VO with FL')

    # ===== FL Setting ======
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
    parser.add_argument('--model', '-model', type=str, default='vonet',
                        help="name of pretrained model (default: 'vonet')")
    # batch_size: model 학습과 evaluation에서 사용할 data 개수 조정
    parser.add_argument('--optimizer', '-opt', type=str, default='Adam',
                        help="name of model's optimizer (default: 'Adam')")
    parser.add_argument('--batch_size', '-batch', type=int, default=1,
                        help='batch size (default: 1)')
    # global_round: 전체 학습 round
    parser.add_argument('--global_round', '-round', type=int, default=100000,
                        help='total number of Global round (default: 1)')
    # local_iteration: 각 node의 iteration 횟수
    parser.add_argument('--local_iteration', '-iter', type=int, default=1,
                        help='total number of local iteration per each node (default: 1)')
    # learning_rate: model의 learning rate
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                        help='model learning rate (default: 1e-4)')
    
    # ====== Dataset Setting ======
    # dataset: train과 test에 사용할 dataset
    parser.add_argument('--data_name', '-dataset', type=str, default='tartanair',
                        help="Dataset name(select: tartanair, euroc, kitti), (default: tartanair)")
    parser.add_argument('--data_path', '-path', type=str, default='/sctrach/jeongeon/tartanAir',
                        help="Dataset folde path, (default: /sctrach/jeongeon/tartanAir)")
    parser.add_argument('--easy_hard', '-e_h', type=str, default='easy',
                        help="Dataset type(select: easy, hard, both), (default: easy)")
    # image_width: 이미지 크기 조정에 사용할 parameter
    parser.add_argument('--image_width', '-img_w', type=int, default=640,
                        help='image width (select: multiple of 64), (default: 640)')
    # image_height: 이미지 크기 조정에 사용할 parameter
    parser.add_argument('--image_height', '-img_h', type=int, default=448,
                        help='image height (select: multiple of 64), (default: 448)')

    #TODO Wandb 환경 세팅하기
    # ====== Wandb Setting ======
    # exp_name: Model의 weight와 학습 이후 Summary log 저장 path 설정 
    parser.add_argument('--exp_name', '-exp', type=str, default='Test',
                        help="Saving Path for weight&Summary (default: Test)")
    
    args = parser.parse_args()
    return args


    
if __name__ == '__main__':

    args = get_args()

    NUM_NODE = args.node_num
    GLOBAL_ROUND = args.global_round
    DATASET_NAME = args.data_name
    EXP_NAME = args.exp_name

    print('init the model & optimizer & scheduler ..')
    model, optimizer, scheduler = init_model(args.model, args.optimizer, args.learning_rate)
    print('Success to compose model!\n')

    print('init dataset and split the dataset..')
    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
    # Can change test environments
    train_data, test_data, node_env_mapping = initial_dataset(data_name=DATASET_NAME, root_dir=args.data_path, mode=args.easy_hard, 
                                                              node_num=NUM_NODE, transform=transform, test_environments=['ocean', 'amusement'])
    print('success to split Server & Node dataset(test, train)!\n')

    print('init Server and Nodes..')
    Node = compose_node(args, model, optimizer, scheduler, train_data)
    Server = compose_server(args, model, optimizer, scheduler, Node, test_data, train_data)
    print('Success to compose Server & Nodes!\n')
    
    summaryWriter = SummaryWriter(f'runs/{EXP_NAME}')
    #start_epoch,iteration = load_checkpoint(model, optimizer, scheduler, args.model_name)

    for round in range(GLOBAL_ROUND):
        #TODO 밑의 코드는 train 코드니까 따로 함수로 빼두기. 아마 Library/util에 만들기
        Server.train()

        '''for posefile in posefiles:
            posefile = posefile.strip()
            print(posefile)
            #transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
            #trainDataset = TartanDataset( posefile = posefile, transform=transform, 
            #                                    focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
            
            #trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker_num)
            #trainDataiter = iter(trainDataloader)

            for batch_idx, sample in enumerate(trainDataiter):
                total_loss,flow_loss,pose_loss,trans_loss,rot_loss = train_pose_batch(model, optimizer, sample)
                iteration += 1
                scheduler.step()

            if round % 10 == 0:
                summaryWriter.add_scalar('Loss/train_total', total_loss, round)
                summaryWriter.add_scalar('Loss/train_flow', flow_loss, round)
                summaryWriter.add_scalar('Loss/train_pose', pose_loss, round)
                summaryWriter.add_scalar('Loss/train_trans', trans_loss, round)
                summaryWriter.add_scalar('Loss/train_rot', rot_loss, round)
                print(f"Epoch {round + 1}, Loss: {total_loss}")
                print(f"Flow Loss: {flow_loss}")
                print(f"Pose Loss: {pose_loss}")
'''
        model_save_path = f'models/exp_{EXP_NAME}.pth'
        #save_checkpoint(model.vonet, optimizer, scheduler, round, iteration, model_save_path)


        



