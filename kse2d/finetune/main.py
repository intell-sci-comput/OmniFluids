from model import Student2D
from train import train
from tools import setup_seed, param_flops, parse_filename

import json
import sys
import copy
from datetime import datetime
import random
import argparse
import numpy as np
import re
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def main(cfg):
    if not os.path.exists(f'log'):
        os.mkdir(f'log')
    if not os.path.exists(f'log/log_{cfg.file_name}'):
        os.mkdir(f'log/log_{cfg.file_name}')
    if not os.path.exists(f'model'):
        os.mkdir(f'model')
    setup_seed(cfg.seed)


    model_path = cfg.model_path
    parse_filename(cfg, model_path)   
    net = Student2D(s=cfg.size, K=cfg.K, modes=cfg.modes, width=cfg.width, n_layers=cfg.n_layers).to(cfg.device)
    net.load_state_dict(torch.load(model_path, map_location=cfg.device), strict=False)

    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}_{dateTimeObj.date().day}_{dateTimeObj.time().hour}_{dateTimeObj.time().minute}_{dateTimeObj.time().second}'
    cfg.model_name = f'net-seed-{cfg.seed}-K-{cfg.K}-modes-{cfg.modes}-width-{cfg.width}-n_layers-{cfg.n_layers}-size-{cfg.size}-{timestring}'
    logfile = f'log/log_{cfg.file_name}/log-seed-{cfg.seed}-rollout_DT-{cfg.rollout_DT}-lr-{cfg.lr}-K-{cfg.K}-modes-{cfg.modes}-width-{cfg.width}-n_layers-{cfg.n_layers}-size-{cfg.size}-{timestring}.csv'
    final_time = datetime.now()
    sys.stdout = open(logfile, 'w')
    print('--------args----------')
    for k in list(vars(cfg).keys()):
        print('%s: %s' % (k, vars(cfg)[k]))
    print('--------args----------\n')
    param_flops(net)
    train(cfg, net)
    sys.stdout.flush()
    print(f'FINAL_TIME_{final_time.date().month}_{final_time.date().day}_{final_time.time().hour}_{final_time.time().minute}_{final_time.time().second}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters of pretraining')

    parser.add_argument('--model_path', type=str, 
                        default='../distillation/model/net-seed-0-K-4-modes-16-width-64-n_layers-8-size-32-8_9_21_33_43.pt',
                        help='path of model') 
    
    parser.add_argument('--data_path', type=str, 
                        default='../data/',
                        help='path of data') 
    
    parser.add_argument('--file_name', type=str, default='',
            help='file name')
    
    parser.add_argument('--data_name', type=str, default='',
            help='data file name')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Used device')
    
    parser.add_argument('--seed', type=int, default=0,
            help='seed')
    
    # model hyper-parameter
    parser.add_argument('--modes', type=int, default=24,
            help='modes in FFNO')
    
    parser.add_argument('--width', type=int, default=60,
            help='width in FFNO')
    
    parser.add_argument('--n_layers', type=int, default=8,
            help='number of layers in FFNO')
    
    parser.add_argument('--size', type=int, default=32,
            help='size of neural operator')

    parser.add_argument('--model_name', type=str, default='',
            help='model name')

    parser.add_argument('--K', type=int, default=8,
            help='batchsize of the operator learning')
    # hyper-parameter during training
    parser.add_argument('--batch_size', type=int, default=10,
            help='batchsize of the operator learning')
    
    parser.add_argument('--rollout_DT', type=float, default=1.0,
            help='the length of time interval of each roll-out')

    parser.add_argument('--lr', type=float, default=0.002,
            help='lr of optim')

    parser.add_argument('--weight_decay', type=float, default=0.00,
            help='lr of optim')

    parser.add_argument('--num_iterations', type=int, default=1000,
            help='num_iterations of optim')
    
    parser.add_argument('--num_train', type=int, default=2,
            help='num of training data')

    cfg = parser.parse_args()
    cfg.data_name = '_0.2_0.5'
    cfg.file_name = 'search_param' + cfg.data_name
    main(cfg)

    cfg.data_name = '_1_1'
    cfg.file_name = 'search_param' + cfg.data_name
    main(cfg)