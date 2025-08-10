from tools import Init_generation, Force_generation

import math
import sys
import copy
import json
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


EPS = 1e-7

def test(config, net, test_data, test_param, test_data_dict):
    device = config.device
    T_final = test_data_dict['T']
    data_ratio = test_data_dict['record_ratio']
    rollout_DT = config.rollout_DT  
    # model_ratio = round(1.0/rollout_DT)
    sub = round(data_ratio / (1.0/rollout_DT))
    total_iter = round(T_final / rollout_DT)
    w_0 = test_data[:, :, :, 0:1].to(device)
    net.eval()
    w_pre = w_0
    with torch.no_grad():
        for _ in range(total_iter):
            w_0 = net(w_pre[..., -1:], test_param.to(device)).detach()[..., -1:]
            w_pre = torch.concat([w_pre, w_0], dim=-1)

    rela_err = []
    print('_________Training__________')
    for time_step in range(1, total_iter+1):
        w = w_pre[..., time_step]
        w_t = test_data[..., sub * time_step].to(device)
        rela_err.append((torch.norm((w- w_t).reshape(w.shape[0], -1), dim=1) / torch.norm(w_t.reshape(w.shape[0], -1), dim=1)).mean().item())
        if time_step % 1 == 0:
            print(time_step, 'relative l_2 error', rela_err[time_step-1])
    print('Mean Relative l_2 Error', np.mean(rela_err))
    return np.mean(rela_err)


def train(config, net):
    device = config.device
    data_name = config.data_name

    data_dict_path = config.data_path+f'log/cfg_test{data_name}.txt'
    with open(data_dict_path, 'r') as file:
        file_content = file.read()
    test_data_dict = json.loads(file_content)

    data_dict_path = config.data_path+f'log/cfg_val{data_name}.txt'
    with open(data_dict_path, 'r') as file:
        file_content = file.read()
    val_data_dict = json.loads(file_content)

    data_dict_path = config.data_path+f'log/cfg_train{data_name}.txt'
    with open(data_dict_path, 'r') as file:
        file_content = file.read()
    train_data_dict = json.loads(file_content)

    data_size = test_data_dict['s'] // test_data_dict['sub']
    size = config.size
    sub = max(1, data_size//size)
    num_train = config.num_train

    train_data = torch.load(config.data_path+f'dataset/data_train{data_name}')[:num_train, ::sub, ::sub, ...].float()
    train_param = torch.load(config.data_path+f'dataset/f_train{data_name}')[:num_train,::sub, ::sub, ...].float().to(device)
    print(train_data.shape)
    test_data = torch.load(config.data_path+f'dataset/data_test{data_name}')[:, ::sub, ::sub, ...].float()
    test_param = torch.load(config.data_path+f'dataset/f_test{data_name}')[:, ::sub, ::sub, ...].float()
    print(test_data.shape)
    val_data = torch.load(config.data_path+f'dataset/data_val{data_name}')[:, ::sub, ::sub, ...].float()
    val_param = torch.load(config.data_path+f'dataset/f_val{data_name}')[:, ::sub, ::sub, ...].float()
    print(val_data.shape)

    rollout_DT = config.rollout_DT  
    train_step = round(train_data_dict['record_ratio'] / (1.0/rollout_DT))
    optimizer = optim.Adam(net.parameters(), config.lr, weight_decay=config.weight_decay)
    print('-----------START_VAL_ERROR-----------')
    val_error = test(config, net, val_data, val_param, val_data_dict)
    print('-----------START_TEST_ERROR-----------')
    test_error = test(config, net, test_data, test_param, test_data_dict)
    torch.save(net.state_dict(), f'model/{config.model_name}.pt')
    for step in range(config.num_iterations):
        net.train()
        w0 = torch.zeros(train_data.shape[0], size, size, 1, device=device)
        w_gth1 = torch.zeros(train_data.shape[0], size, size, 1, device=device)
        w_gth2 = torch.zeros(train_data.shape[0], size, size, 1, device=device)
        for i in range(num_train):
            t = random.randint(0, train_data.shape[-1] - 2 * train_step -1)
            w0[i, ..., 0] = train_data[i, ..., t].to(device)
            w_gth1[i, ..., 0] = train_data[i, ..., t+train_step].to(device)
            w_gth2[i, ..., 0] = train_data[i, ..., t+2*train_step].to(device)
        w_pre = net(w0, train_param)[..., -1:]
        loss = torch.square(w_pre - w_gth1).mean()
        loss = (loss + EPS).sqrt()
        w_pre = net(w_pre, train_param)[..., -1:]
        loss = loss + (torch.square(w_pre - w_gth2).mean() + EPS).sqrt() 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(step, '########################')
            print('training loss', step, loss.detach().item())
            print('-----------VAL_ERROR-----------')
            val_error_now = test(config, net, val_data, val_param, val_data_dict)
            if val_error_now < val_error:
                val_error = val_error_now
                print('-----------TEST_ERROR-----------')
                test(config, net, test_data, test_param, test_data_dict)
                print('-----------SAVING NEW MODEL-----------')
                torch.save(net.state_dict(), f'model/{config.model_name}.pt')
            sys.stdout.flush()
    print('----------------------------FINAL_RESULT-----------------------------')
    net.load_state_dict(torch.load(f'model/{config.model_name}.pt'))
    test(config, net, test_data, test_param, test_data_dict)
    sys.stdout.flush()