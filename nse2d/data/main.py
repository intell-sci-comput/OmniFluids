from generate_data import generate_ns_data

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters of data generation')

    parser.add_argument('--device', type=str, default='cuda:3',
                        help='Used device')
    
    parser.add_argument('--seed', type=int, default=1,
            help='seed')
    
    parser.add_argument('--s', type=int, default=1024,
            help='data original size')

    parser.add_argument('--sub', type=int, default=2,
            help='ratio of down sampling')
    
    parser.add_argument('--max_frequency', type=int, default=10,
            help='max_frequency of forcing')
    
    parser.add_argument('--amplitude', type=float, default=0.5,
            help='amplitude')

    parser.add_argument('--N', type=int, default=10,
            help='Number of the data generation')
    
    parser.add_argument('--re', type=str, default="[400, 4000]",
            help='Re in NSE')
    
    parser.add_argument('--name', type=str, default='test',
            help='the name of the data set')
    
    parser.add_argument('--T', type=float, default=10.0,
            help='final time')

    parser.add_argument('--dt', type=float, default=1e-4,
            help='dt')

    parser.add_argument('--record_ratio', type=int, default=10,
            help='record ratio')
    

    for re in ["[500, 2500]", "[4000, 4000]", "[2000, 2000]"]:
        cfg = parser.parse_args()
        cfg.re = re
        cfg.T = 10.0
        cfg.sub = 4
        cfg.seed = 1
        cfg.N = 10
        cfg.record_ratio = 10
        print(re)
        cfg.device = 'cuda:2'
        cfg.re = eval(cfg.re)
        if cfg.re[0]!=cfg.re[1]:
            cfg.name = 'test_multi'
        else:
            cfg.name = 'test' + str(cfg.re[0])

        save_path = f'./log'
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        with open(f'{save_path}/cfg_{cfg.name}.txt', 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
        sys.stdout.flush()
        generate_ns_data(cfg)


    for re in ["[4000, 4000]", "[2000, 2000]"]:
        cfg = parser.parse_args()
        cfg.re = re
        cfg.T = 10.0
        cfg.sub = 4
        cfg.seed = 0
        cfg.N = 10
        cfg.record_ratio = 10
        print(re)
        cfg.device = 'cuda:2'
        cfg.re = eval(cfg.re)
        if cfg.re[0]!=cfg.re[1]:
            cfg.name = 'train_multi'
        else:
            cfg.name = 'train' + str(cfg.re[0])

        save_path = f'./log'
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        with open(f'{save_path}/cfg_{cfg.name}.txt', 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
        sys.stdout.flush()
        generate_ns_data(cfg)


    for re in ["[4000, 4000]", "[2000, 2000]"]:
        cfg = parser.parse_args()
        cfg.re = re
        cfg.T = 10.0
        cfg.sub = 4
        cfg.seed = 2
        cfg.N = 2
        cfg.record_ratio = 10
        cfg.device = 'cuda:2'
        print(re)
        cfg.re = eval(cfg.re)
        if cfg.re[0]!=cfg.re[1]:
            cfg.name = 'val_multi'
        else:
            cfg.name = 'val' + str(cfg.re[0])

        save_path = f'./log'
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        with open(f'{save_path}/cfg_{cfg.name}.txt', 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
        sys.stdout.flush()
        generate_ns_data(cfg)