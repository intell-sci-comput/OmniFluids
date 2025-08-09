from tools import Init_generation
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



def train(config, net_t, net_s):
    device = config.device
    data_name = config.data_name
    data_dict_path = config.data_path+f'log/cfg_test_{data_name}.txt'
    with open(data_dict_path, 'r') as file:
        file_content = file.read()
    test_data_dict = json.loads(file_content)
    param1, param2 = test_data_dict['param1'],  test_data_dict['param2']
    data_size = test_data_dict['s'] // test_data_dict['sub']
    batch_size = config.batch_size
    size = config.size
    sub = max(1, data_size//config.student_size)
    test_data = torch.load(config.data_path+f'dataset/test_{data_name}')[:, ::sub, ::sub, ...].float()
    test_param = torch.load(config.data_path+f'dataset/param_test_{data_name}').float()
    print(test_data.shape)
    student_sub = max(1, config.size//config.student_size)

    GRF = Init_generation(size, device=device, dtype=torch.float32)
    teacher_step = round(config.rollout_DT / config.rollout_DT_teacher)

    num_iterations = config.num_iterations
    net_s = net_s.to(device)
    optimizer = optim.Adam(net_s.parameters(), config.lr, weight_decay=config.weight_decay)
    scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=config.num_iterations+1, max_lr=config.lr)

    for step in range(num_iterations+1):
        w0_train = GRF(batch_size, )[..., None]
        param = torch.ones(batch_size, 2, device=device)
        param[:, 0] = (param1[1] - param1[0]) * torch.rand(batch_size, device=device) + param1[0]
        param[:, 1] = (param2[1] - param2[0]) * torch.rand(batch_size, device=device) + param2[0]
        w_gth = copy.copy(w0_train)
        net_t.eval()
        with torch.no_grad():
            for i in range(teacher_step):
                w_gth = net_t(w_gth, param).detach()[..., -1:]
        w_gth = w_gth[:, ::student_sub, ::student_sub, ...]
        w0_train = w0_train[:, ::student_sub, ::student_sub, ...]
        for _ in range(10):
            net_s.train()
            w_s = net_s(w0_train, param)
            loss = torch.mean((w_s - w_gth)**2 + EPS).sqrt()
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if step % 10 == 0:
            print('training loss', step, loss.detach().item())
            torch.save(net_s.state_dict(), f'model/{config.model_name}.pt')
            test(config, net_s, test_data, test_param, test_data_dict)
        sys.stdout.flush()
    print('----------------------------FINAL_RESULT-----------------------------')
    net_s.load_state_dict(torch.load(f'model/{config.model_name}.pt'))
    test(config, net_s, test_data, test_param, test_data_dict)
    sys.stdout.flush()


