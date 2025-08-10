from model import OmniFluids2D
from train import train
from tools import setup_seed, param_flops

import json
import sys
import copy
from datetime import datetime
import random
import argparse
import numpy as np
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
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}_{dateTimeObj.date().day}_{dateTimeObj.time().hour}_{dateTimeObj.time().minute}_{dateTimeObj.time().second}'
    cfg.model_name = f'net-seed-{cfg.seed}-K-{cfg.K}-modes-{cfg.modes}-width-{cfg.width}-n_layers-{cfg.n_layers}-output_dim-{cfg.output_dim}-size-{cfg.size}-{timestring}'
    logfile = f'log/log_{cfg.file_name}/log-seed-{cfg.seed}-rollout_DT-{cfg.rollout_DT}-lr-{cfg.lr}-K-{cfg.K}-modes-{cfg.modes}-width-{cfg.width}-n_layers-{cfg.n_layers}-output_dim-{cfg.output_dim}-size-{cfg.size}-{timestring}.csv'
    sys.stdout = open(logfile, 'w')

    print('--------args----------')
    for k in list(vars(cfg).keys()):
        print('%s: %s' % (k, vars(cfg)[k]))
    print('--------args----------\n')
    net = OmniFluids2D(s=cfg.size, K=cfg.K, modes=cfg.modes, width=cfg.width, output_dim=cfg.output_dim, n_layers=cfg.n_layers)
    param_flops(net)
    sys.stdout.flush()
    train(cfg, net)

    final_time = datetime.now()
    print(f'FINAL_TIME_{final_time.date().month}_{final_time.date().day}_{final_time.time().hour}_{final_time.time().minute}_{final_time.time().second}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters of pretraining')

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
    parser.add_argument('--modes', type=int, default=128,
            help='modes in FFNO')
    
    parser.add_argument('--width', type=int, default=80,
            help='width in FFNO')
    
    parser.add_argument('--n_layers', type=int, default=12,
            help='number of layers in FFNO')
    
    parser.add_argument('--output_dim', type=int, default=50,
            help='output size of FFNO')
    
    parser.add_argument('--size', type=int, default=256,
            help='size of neural operator')

    parser.add_argument('--model_name', type=str, default='',
            help='model name')

    parser.add_argument('--K', type=int, default=4,
            help='batchsize of the operator learning')

    parser.add_argument('--batch_size', type=int, default=10,
            help='batchsize of the operator learning')
            
    parser.add_argument('--val_size', type=int, default=10,
            help='number of validation set during training')
    
    parser.add_argument('-loss_mode', type=str, default='cn',
            help='the way of buliding loss function, mid or cn')
    
    parser.add_argument('--rollout_DT', type=float, default=0.2,
            help='the length of time interval of each roll-out')

    parser.add_argument('--lr', type=float, default=0.002,
            help='lr of optim')

    parser.add_argument('--weight_decay', type=float, default=0.00,
            help='lr of optim')

    parser.add_argument('--num_iterations', type=int, default=20000,
            help='num_iterations of optim')

    cfg = parser.parse_args()
    cfg.file_name = 'search_param'
    cfg.data_name = '_multi'
    main(cfg)