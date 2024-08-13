import torch

class BASE:
    def __init__(self):
        pass

    @staticmethod
    def average_model(models, avg_ratio=None):
        # 평균화된 모델 가중치를 저장할 리스트
        new_weights = list()

        if avg_ratio is None:
            # 각 레이어별로 가중치 평균 계산
            for weights_list_tuple in zip(*models):
                # 각 레이어의 가중치 평균 계산
                avg_weights = [torch.mean(torch.stack(weights), dim=0) for weights in zip(*weights_list_tuple)]
                new_weights.append(avg_weights)
        else:
            # 가중치 평균화 비율이 주어진 경우, 가중치를 기반으로 평균 계산
            for weights_list_tuple in zip(*models):
                # 각 레이어의 가중치 가중 평균 계산
                avg_weights = [torch.sum(torch.stack(weights) * torch.tensor(avg_ratio).view(-1, 1, 1), dim=0) / sum(avg_ratio) for weights in zip(*weights_list_tuple)]
                new_weights.append(avg_weights)
                
        return new_weights