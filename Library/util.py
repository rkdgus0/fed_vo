import random
import torch
import numpy as np

from component.NODE import NODE
from component.SERVER import SERVER
from sklearn.utils import shuffle


def compose_server(args, model, nodes, test_data):
    NUM_MEC = args.num_mec
    avg_method = args.avg_method
    num_node_data = []
    #TODO num_node_data를 받아오는 코드 구성(형태: [node1_data_num, node2_data_num,...])

    return SERVER(model, nodes, NUM_NODE, test_data, avg_method, num_node_data)

def compose_node(args, model, splited_datasets):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return NODE(model, init_lr=args.learning_rate, datasets=splited_datasets, epochs=args.n_epochs, 
                batch_size=args.batch_size, device=device)