import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from time import time, strftime, gmtime
import torch
import argparse
#from torch.utils.tensorboard import SummaryWriter

from Network.VONet import VONet

from Library.fl_util import compose_node, compose_server, init_model, save_checkpoint, plot_traj

from Library.datasets.dataset import initial_dataset
from Library.datasets.dataset_util import ToTensor, Compose, CropCenter, DownscaleFlow


#TODO Parser 파라미터 변경해야할 수 있음.
def get_args():
    parser = argparse.ArgumentParser(description='C_VO with FL')

    # ===== FL Setting ======
    # worker_num: dataset을 GPU로 넘기는 과정에서의 preprocessing을 하는 subprocess 수(많을수록 빠르나, 자원 차지가 많아짐)
    parser.add_argument('--worker_num', '-worker', type=int, default=4,
                        help='data loader worker number (default: 4)')
    # node_num: 연합학습(FL)에서 학습에 참여하는 Node의 수
    parser.add_argument('--node_num', '-node', type=int, default=16,
                        help='number of nodes, different from worker_num (default: 16)')
    parser.add_argument('--avg_method', '-avg', type=str, default='fedavg',
                        help='average method (Select: fedavg, equal), (default: fedavg)')

    # ===== Model Setting =====
    # model: Pretrained된 model weight 저장 경로, .pkl 파일로 받음
    parser.add_argument('--model', '-model', type=str, default='vonet',
                        help="name of pretrained model (default: 'vonet')")
    # batch_size: model 학습과 evaluation에서 사용할 data 개수 조정
    parser.add_argument('--optimizer', '-opt', type=str, default='adam',
                        help="name of model's optimizer (Select: adam, sgd) (default: 'adam')")
    parser.add_argument('--batch_size', '-batch', type=int, default=1,
                        help='batch size (default: 1)')
    # global_round: 전체 학습 round
    parser.add_argument('--global_round', '-round', type=int, default=5,
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
    parser.add_argument('--data_path', '-path', type=str, default='/scratch/jeongeon/tartanAir',
                        help="Dataset folde path, (default: /scratch/jeongeon/tartanAir)")
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
    # eval_round: Evaluation을 수행할 Round 주기
    parser.add_argument('--eval_round', '-eval_round', type=int, default=1,
                        help='Evaluation round (default: 1)')
    
    args = parser.parse_args()
    return args


    
if __name__ == '__main__':

    args = get_args()

    NUM_NODE = args.node_num
    GLOBAL_ROUND = args.global_round
    LOCAL_ROUND = args.local_iteration
    DATASET_NAME = args.data_name
    EXP_NAME = args.exp_name
    TEST_ENVS=['ocean', 'amusement', 'zipfile']

    print('===== init the model & optimizer & scheduler ..')
    t1 = time()
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, optimizer, scheduler = init_model(args.model, args.optimizer, args.learning_rate)
    t2 = time()
    print(f'===== Success to compose model! (Time(sec): {round(t2-t1,2)})\n')

    print('===== init dataset and split the dataset..')
    t1 = time()
    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
    # Can change test environments
    train_data, test_data, node_env_mapping = initial_dataset(data_name=DATASET_NAME, root_dir=args.data_path, mode=args.easy_hard, 
                                                              node_num=NUM_NODE, transform=transform, test_environments=TEST_ENVS)
    t2 = time()
    print(f'===== success to split Server & Node dataset(test, train)! (Time(sec): {round(t2-t1,2)})\n')

    print('===== init Server and Nodes..')
    t1 = time()
    Node = compose_node(args, model, optimizer, scheduler, train_data, device)
    Server = compose_server(args, model, Node, test_data, train_data, device)
    t2 = time()
    print(f'===== Success to compose Server & Nodes! (Time(sec): {round(t2-t1,2)})\n')
    
    #summaryWriter = SummaryWriter(f'runs/{EXP_NAME}')
    #start_epoch,iteration = load_checkpoint(model, optimizer, scheduler, args.model_name)

    print(f'Federated Collaborative VO start!')
    print(f'===== [CVO] Device: {device} (CPU: {args.worker_num})')
    print(f'===== [CVO] EXP NAME: {args.exp_name} (Node: {args.node_num}, Average Method: {args.avg_method})')
    print(f'===== [CVO] Model: {args.model} (Optimizer: {args.optimizer}, Learning Rate: {args.learning_rate})')
    print(f'===== [CVO] Dataset: {args.data_name} (Batch Size: {args.batch_size})')
    print(f'===== [CVO] Image Crop(Width, Height): {args.image_width}, {args.image_height}')
    print(f'===== [CVO] Dataset Path: {args.data_path} (Easy or Hard: {args.easy_hard})')
    print(f'===== [CVO] Global Round: {GLOBAL_ROUND}, Local Iteration: {LOCAL_ROUND} Start!\n')
    
    t00 = time()
    for R in range(1, GLOBAL_ROUND+1):
        print(f'===== Global Round {R} Train start!')
        t1 = time()
        Server.train()
        t2 = time()
        round_time = strftime("%Hh %Mm %Ss", gmtime(t2-t1))
        print(f'[Global Round {R}] Train Time(sec): {round_time}\n')

        if R % args.eval_round == 0 or R == GLOBAL_ROUND+1:
            print(f'===== Global Round {R} Evaluate start!')
            t1 = time()
            result = Server.test()
            print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(result['ate_score'], result['kitti_score'][0], result['kitti_score'][1]))
            plot_traj(result['gt_aligned'], result['est_aligned'], vis=False, savefigname='results/['+EXP_NAME+'] Round_'+R+'.png', title='ATE %.4f' %(result['ate_score']))
            np.savetxt('results/'+EXP_NAME+'.txt',result['est_aligned'])
            t2 = time()
            eval_time = strftime("%Hh %Mm %Ss", gmtime(t2-t1))
            print(f'[Global Round {R}] Evaluation Time(sec): {eval_time}\n')

    model_save_path = f'models/exp_{EXP_NAME}.pth'
    save_checkpoint(model, optimizer, scheduler, GLOBAL_ROUND, LOCAL_ROUND, model_save_path)
    t01 = time()
    total_time = strftime("%Hh %Mm %Ss", gmtime(t01-t00))
    print(f'===== Federated Collaborative VO Finished! (Time(sec): {total_time})\n')