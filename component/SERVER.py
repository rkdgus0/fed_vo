import copy
import torch
import numpy as np
import random
from Library.train import flow_train, flowpose_train, whole_train

class SERVER():
    def __init__(self, model, nodes, NUM_NODE, test_data, avg_method='fedavg', num_node_data=None):
        super(SERVER, self).__init__()
        self.model = model
        self.NUM_NODE = NUM_NODE
        self.nodes = nodes
        self.test_data = test_data
        self.avg_method = avg_method
        self.num_node_data = copy.deepcopy(num_node_data)

    # NUM_NODE 개수만큼 NODE를 선언해서 NODE.train으로 학습, 학습한 모델을 Average하는 코드
    #TODO 일단 구성에 문제는 없어보이고, node의 pose/flowpose/whole train 함수 짜기
    def train(self):
        uploaded_models = []
        participating_node = []

        for train_type in ['flow', 'flowpose', 'whole']:
            for node_idx in range(self.NUM_NODE):
                if train_type == 'flow':
                    self.nodes.flow_train(node_idx, local_epoch)
                elif train_type == 'flowpose':
                    self.nodes.flowpose_train(node_idx, local_epoch)
                elif train_type == 'whole':
                    self.nodes.whole_train(node_idx, local_epoch)
                participating_node.append(node_idx)
                uploaded_models.append(copy.deepcopy(self.nodes.model.state_dict()))

            avg_ratio = self.calc_avg_ratio(uploaded_models, participating_node)
            avg_model = self.average_model(uploaded_models, avg_ratio)
            self.model.load_state_dict(avg_model)

            for node_idx in participating_node:
                self.nodes.model.load_state_dict(avg_model)

            uploaded_models.clear()
            avg_model.clear()
   
   #TODO Test dataset을 이용해서, Global Model의 성능을 확인하는 코드 추가 요망(evaluation code와 동일할 것으로 보임)
    def test(self):
        
        return

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
        return ratio

    def set_lr(self, lr):
        for param_group in self.nodes.optimizer.param_groups:
            param_group['lr'] = lr

    # calc_avg_ratio의 가중치를 기준으로 local model을 합쳐, global model 생성
    @staticmethod
    def average_model(models, avg_ratio=None):
        new_weights = []
        if avg_ratio is None:
            for weights_list_tuple in zip(*models):
                avg_weights = [torch.mean(torch.stack([w for w in weights_list_tuple]), dim=0) for _ in weights_list_tuple[0]]
                new_weights.append(avg_weights)
        elif avg_ratio:
            for weights_list_tuple in zip(*models):
                avg_weights = [torch.sum(torch.stack([w for w in weights_list_tuple]) * torch.tensor(avg_ratio).view(-1, 1, 1), dim=0) / sum(avg_ratio) for _ in weights_list_tuple[0]]
                new_weights.append(avg_weights)
        return new_weights