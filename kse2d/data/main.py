from generate_data import *

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

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Used device')
    
    parser.add_argument('--mode', type=str, default='train',
            help='train or test')
    
    parser.add_argument('--s', type=int, default=128,
            help='data original size')

    parser.add_argument('--sub', type=int, default=1,
            help='ratio of down sampling')
    
    parser.add_argument('--N', type=int, default=2,
            help='Number of the data generation')
    
    parser.add_argument('--T', type=float, default=5.0,
            help='final time')

    parser.add_argument('--dt', type=float, default=1e-5,
            help='dt')

    parser.add_argument('--record_ratio', type=int, default=10,
            help='record ratio')
    
    parser.add_argument('--param1', type=str, default="[0.1, 0.5]",
            help='param1 in KSE')

    parser.add_argument('--param2', type=str, default="[0.1, 0.5]",
            help='param1 in KSE')
    
    cfg = parser.parse_args()

    for mode in ['test']:
        cfg.mode = mode
        cfg.param1 = '[0.1, 0.5]'
        cfg.param2 = '[0.1, 0.5]'
        cfg.param1 = eval(cfg.param1)
        cfg.param2 = eval(cfg.param2)
        if cfg.param1[0]!=cfg.param1[1]:
            cfg.name = cfg.mode + '_multi'
        else:
            cfg.name = cfg.mode + '_' + str(cfg.param1[0]) + '_' + str(cfg.param2[0])

        save_path = f'./log'
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        with open(f'{save_path}/cfg_{cfg.name}.txt', 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
        sys.stdout.flush()
        generate_ks_data(cfg)


    cfg = parser.parse_args()

    for mode in ['test', 'train', 'val']:
        cfg.mode = mode
        cfg.param1 = '[0.2, 0.2]'
        cfg.param2 = '[0.5, 0.5]'
        cfg.param1 = eval(cfg.param1)
        cfg.param2 = eval(cfg.param2)
        if cfg.param1[0]!=cfg.param1[1]:
            cfg.name = cfg.mode + '_multi'
        else:
            cfg.name = cfg.mode + '_' + str(cfg.param1[0]) + '_' + str(cfg.param2[0])

        save_path = f'./log'
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        with open(f'{save_path}/cfg_{cfg.name}.txt', 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
        sys.stdout.flush()
        generate_ks_data(cfg)


    cfg = parser.parse_args()

    for mode in ['test', 'train', 'val']:
        cfg.mode = mode
        cfg.param1 = '[1, 1]'
        cfg.param2 = '[1, 1]'
        cfg.param1 = eval(cfg.param1)
        cfg.param2 = eval(cfg.param2)
        if cfg.param1[0]!=cfg.param1[1]:
            cfg.name = cfg.mode + '_multi'
        else:
            cfg.name = cfg.mode + '_' + str(cfg.param1[0]) + '_' + str(cfg.param2[0])

        save_path = f'./log'
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        with open(f'{save_path}/cfg_{cfg.name}.txt', 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
        sys.stdout.flush()
        generate_ks_data(cfg)