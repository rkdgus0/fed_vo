import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from Library.component.NODE import NODE
from Library.evaluator.tartanair_evaluator import TartanAirEvaluator
from Library.datasets.transformation import ses2poses_quat

class SERVER():
    def __init__(self, model, nodes, NUM_NODE, test_data_name, test_data, test_data_path, avg_method, num_node_data, batch_size, worker_num, device='cuda:0'):
        super(SERVER, self).__init__()
        self.model = model.to(device)
        self.NUM_NODE = NUM_NODE
        self.nodes = nodes
        self.test_data = test_data
        self.test_data_path = test_data_path
        self.avg_method = avg_method
        self.num_node_data = deepcopy(num_node_data)
        self.batch_size = batch_size
        self.worker_num = worker_num
        self.device = device
        self.evaluator = TartanAirEvaluator()
        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32)
        self.kittitype = test_data_name.lower() == 'kitti'

    # NUM_NODE 개수만큼 NODE를 선언해서 NODE.train으로 학습, 학습한 모델을 Average하는 코드
    def train(self, model_name):
        model_parameter = self.model.state_dict()
        node_state_dicts = []
        participating_node = []

        #for train_type in ['whole']:
        for node_idx in range(self.NUM_NODE):
            if len(self.nodes.datasets[node_idx]) == 0:
                pass
            else:
                self.nodes.optimizer.zero_grad()
                #before_update = {name: param.clone() for name, param in self.model.named_parameters()}
                self.nodes.train(node_idx, model_parameter, model_name)
                participating_node.append(node_idx)
                node_state_dict = self.nodes.model.state_dict()
                node_state_dicts.append(node_state_dict)

        avg_ratio = self.calc_avg_ratio(node_state_dicts, participating_node)
        print(f"avg ratio: {avg_ratio}")
        #avg_ratio_tensor = torch.tensor(avg_ratio, dtype=torch.float32).view(-1, 1, 1, 1, 1)
        avg_model = self.average_model(node_state_dicts, avg_ratio)

        self.model.load_state_dict(avg_model)
        
        participating_node.clear()
        node_state_dicts.clear()
        avg_model.clear()

   
   #TODO flownet/ posenet model 사용시, vonet을 선언해서 해당 모듈에 load_state_dict를 해야함
   # model_name을 받아와야함.
    def test(self):
        pose_preds = []
        pose_gts = []
        
        testDataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.worker_num)

        self.model.eval()

        with torch.no_grad():
            for sample in testDataloader:
                img1 = sample['img1'].to(self.device)
                img2 = sample['img2'].to(self.device)
                intrinsic = sample['intrinsic'].to(self.device)
                motion_gt = sample['motion'].to(self.device)

                _, pose_pred = self.model([img1, img2, intrinsic])
                pose_pred = pose_pred.cpu().numpy() * self.pose_std
                # transition의 size를 gt와 동일하게 설정
                scale = np.linalg.norm(motion_gt[:,:3].cpu().numpy(), axis=1)
                trans_pred = pose_pred[:,:3]
                trans_pred = trans_pred/np.linalg.norm(trans_pred,axis=1).reshape(-1,1)*scale.reshape(-1,1)
                pose_pred[:,:3] = trans_pred
                
                pose_preds.append(pose_pred)
        
        pose_preds = ses2poses_quat(np.concatenate(pose_preds, axis=0))
        pose_gts = np.loadtxt(self.test_data_path).astype(np.float32)
        result = self.evaluator.evaluate_one_trajectory(pose_gts, pose_preds, scale=True, kittitype=self.kittitype)
        return result

    # 모델 가중치 aggregation method(Fedavg: 데이터 수, Equal: 동등)
    def calc_avg_ratio(self, models, participating_node):
        ratio = []
        if self.avg_method == 'fedavg':
            for node_idx in participating_node:
                ratio.append(self.num_node_data[node_idx])
        elif self.avg_method == 'Equal':
            ratio = [1] * len(models)
        else:
            print("invalid global model average method!")
        
        ratio = [r / sum(ratio) for r in ratio]
        return ratio

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def average_model(self, state_dicts, avg_ratio):
        total_sample_num = sum(avg_ratio)
        temp_sample_num = avg_ratio[0]

        if len(state_dicts) != len(avg_ratio):
            raise ValueError("The number of state_dicts and avg_ratio must be the same")

        state_avg = deepcopy(state_dicts[0])
        for key in state_avg.keys():
            for i in range(1, len(state_dicts)):
                state_avg[key] = state_avg[key] + torch.mul(state_dicts[i][key], torch.div(avg_ratio[i], temp_sample_num))
            state_avg[key] = torch.mul(state_avg[key], torch.div(temp_sample_num, total_sample_num))
        
        return state_avg

#테스트용
from Library.datasets.dataset_util import ToTensor, Compose, CropCenter, DownscaleFlow, make_intrinsics_layer, dataset_intrinsics
from Library.datasets.dataset import initial_dataset
from Network.VONet import VONet
from torch.optim.lr_scheduler import ExponentialLR
from time import time
import matplotlib.pyplot as plt
import os
from time import time, strftime, gmtime


    
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


if __name__ == '__main__':
    data_name = 'tartanair'
    root_dir = '/scratch/jeongeon/tartanAir/train_data'
    easy_hard = 'easy'
    sequence= 'P001'
    node_num = 17
    transform = Compose([CropCenter((640, 448)), DownscaleFlow(), ToTensor()])
    test_environments = ['ocean']
    # , 'office2', 'soulcity', 'carwelding', 'abandonedfactory_night', 'seasonsforest', 'neighborhood', 'japanesealley', 'oldtown', 'carwelding', 'seasonsforest_winter', 'westerndesert', 'office', 'hospital', 'gascola', 'amusement'
    iteration = 1
    batch_size = 64
    worker = 32
    avg_method = 'fedavg'
    global_round = 3
    init_lr = 0.0005
    
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(VONet())
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = ExponentialLR(optimizer, gamma=0.95)

    print("Init Dataset...")
    t1 = time()
    train_data, test_data, node_envs = initial_dataset(data_name, root_dir, easy_hard, sequence, node_num, transform, test_environments)
    '''for i, train_dataset in enumerate(train_data):
        print(f"Node {i+1} Train dataset size: {len(train_dataset)}")
        print(f"Node {i+1} Train dataset environment name: {node_envs[i]}")'''
    t2 = time()
    print(f"Success to init dataset! Time(sec): {round(t2-t1,2)}\n")

    print("Init Node component...")
    t1 = time()
    Node =  NODE(model, init_lr=init_lr, datasets=train_data, iteration=iteration, batch_size=batch_size, worker_num=worker, device=device)
    num_node_data = []
    for node_idx, train_dataset in enumerate(train_data):
        num_node_data.append(len(train_dataset))
    print(f"num_node_data: {num_node_data}")
    Server = SERVER(model, Node, node_num, test_data, avg_method, num_node_data, batch_size, worker, device=device)
    t2 = time()
    print(f'Node num: {len(train_data)}')
    print(f"Success to init component! Time(sec): {round(t2-t1,2)}\n")

    for R in range(global_round):
        t1 = time()
        print('Evaluate Start')
        result = Server.test()
        print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(result['ate_score'], result['kitti_score'][0], result['kitti_score'][1]))
        plot_traj(result['gt_aligned'], result['est_aligned'], vis=False, savefigname=f'../../results/debug_{str(R)}.png', title='ATE %.4f' %(result['ate_score']))
        #np.savetxt(f'../../results/debug_{str(R)}.txt',result['est_aligned'])
        print(f"Trajectory saved as debug_{str(R)}")
        t2 = time()
        eval_time = strftime("%Hh %Mm %Ss", gmtime(t2-t1))
        print(f'[Global Round {R}] Evaluation Time(sec): {eval_time}\n')

        print(f"global round {R+1} Start!")
        t1 = time()
        Server.train()

        t2 = time()
        round_time = strftime("%Hh %Mm %Ss", gmtime(t2-t1))
        print(f'[Global Round {R+1}] Train Time(sec): {round_time}\n')

    t1 = time()
    result = Server.test()
    print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(result['ate_score'], result['kitti_score'][0], result['kitti_score'][1]))
    plot_traj(result['gt_aligned'], result['est_aligned'], vis=False, savefigname=f'../../results/debug_{str(R)}.png', title='ATE %.4f' %(result['ate_score']))
    #np.savetxt('../../results/debug.txt',result['est_aligned'])
    print(f"Trajectory saved as debug_final")
    t2 = time()
    eval_time = strftime("%Hh %Mm %Ss", gmtime(t2-t1))
    print(f'[Global Round {R+1}] Evaluation Time(sec): {eval_time}\n')
