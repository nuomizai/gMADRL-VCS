import os

os.environ['MKL_NUM_THREADS'] = '1'
import argparse

import copy
import datetime
from datetime import datetime
import glob
import importlib
import math
import numpy as np
import os.path
# import paramiko
import random
import socket
import stat
from statistics import mean
import sys
import time
import torch
import torch.nn as nn
from typing import List
# from torch.utils.tensorboard import SummaryWriter
import shutil
# from scipy.linalg import fractional_matrix_power
import torch.optim as optim
import matplotlib as mpl
import matplotlib.pyplot as plt
# import shapefile  # 使用pyshp
# import utm
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.multiprocessing as mp
# from tqdm import tqdm
import traceback
import einops
import torch.nn.functional as F
from numpy.linalg import norm
import re


# ============= hydra, logger import ===============
import hydra
from my_logger import my_logger, setup_my_logger
from pathlib import Path

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

project_name = os.path.abspath(__file__).split('/')[-2]
min_value = 1e-5

UAV_RETURN = 0
UAV_FLIGHT = 1

UGV_RELEASE = 0
UGV_STOP = 1

MAIN_STREET=0
CROSS_STREET=1
SUB_STREET=2
EDGE_STREET=3

def calculate_azimuth_angle(st_loc, ed_loc, degree=True):
    st_loc = np.array(st_loc)
    ed_loc = np.array(ed_loc)
    vec = ed_loc - st_loc
    north_vec = np.array([0, 1])
    rho = np.arcsin(np.cross(north_vec, vec) / np.linalg.norm(vec))
    rho = np.rad2deg(rho)
    theta = np.arccos(np.dot(north_vec, vec) / np.linalg.norm(vec))
    if degree:
        theta = np.rad2deg(theta)
        if rho < 0:
            return 360-theta
        else:
            return theta
    else:
        return theta

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def dijkstra(graph, start_node_id, goal_node_id):
    if start_node_id == goal_node_id:
        return [start_node_id, goal_node_id], 0  # 返回最短路径和最短路径长度
    open_dict = {}
    closed_dict = {}
    open_dict[start_node_id] = 0  # 将起点放入 open_dict 中
    parent_dict = {start_node_id: None}  # 存储节点的父子关系。键为子节点，值为父节点。方便做最后路径的回溯
    while True:
        if open_dict is None:
            print('搜索失败， 结束！')
            break
        distance, min_dis_node_id = min(zip(open_dict.values(), open_dict.keys()))  # 取出距离最小的节点
        open_dict.pop(min_dis_node_id)  # 将其从 open_dict 中去除
        closed_dict[min_dis_node_id] = distance  # 将节点加入 closed_dict 中
        if min_dis_node_id == goal_node_id:  # 如果节点为终点
            min_dis = distance
            shortest_path = [goal_node_id]  # 记录从终点回溯的路径
            father_node_id = parent_dict[goal_node_id]
            while father_node_id != start_node_id:
                shortest_path.append(father_node_id)
                father_node_id = parent_dict[father_node_id]
            shortest_path.append(start_node_id)
            return shortest_path[::-1], min_dis  # 返回最短路径和最短路径长度
        for node_id in graph[min_dis_node_id]['next_node_list']:  # 遍历当前节点的邻接节点
            if node_id not in closed_dict.keys():  # 邻接节点不在 closed_dict 中
                if node_id in open_dict.keys():  # 如果节点在 open_dict 中
                    if distance + 1 < open_dict[node_id]:
                        open_dict[node_id] = distance + 1  # 更新节点的值
                        parent_dict[node_id] = min_dis_node_id  # 更新继承关系
                else:  # 如果节点不在 open_dict 中
                    open_dict[node_id] = distance + 1  # 计算节点的值，并加入 open_dict 中
                    parent_dict[node_id] = min_dis_node_id  # 更新继承关系


def dis_p2p(point1, point2):
    dis = math.sqrt(math.pow((point2[0] - point1[0]), 2) + math.pow((point2[1] - point1[1]), 2))
    return dis


def vector_dot_product(vector1, vector2):
    result = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    return result


def vector_length_counter(vector):
    length = math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))
    return length


def vector_unitize(vector):
    vector_length = vector_length_counter(vector)
    unitized_vector = (vector[0] / vector_length, vector[1] / vector_length)
    return unitized_vector


def vector1_project2_vector2(vector1, vector2):
    u = vector_dot_product(vector1, vector2) / (vector_length_counter(vector2)) ** 2
    target_vector = (vector2[0] * u, vector2[1] * u)
    return target_vector


def vector1_antiproject2_vector2(vector1, vector2):
    u = (vector_length_counter(vector1)) ** 2 / vector_dot_product(vector1, vector2)
    target_vector = (vector2[0] * u, vector2[1] * u)
    return target_vector


def global_dict_init():  # 初始化
    global _global_dict
    _global_dict = {}


def set_global_dict_value(key, value):
    # 定义一个全局变量
    _global_dict[key] = value


def get_global_dict_value(key):
    # 获得一个全局变量，不存在则提示读取对应变量失败
    try:
        return _global_dict[key]
    except:
        print(datetime.now(), '读取' + key + '失败\r\n')

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def interact_batch_gen(step_input_dict, dict_key_list):
    batch_all = []
    for dict_key in dict_key_list:
        batch = []
        for (key, item) in step_input_dict.items():
            batch.append(item[dict_key])
        batch = torch.tensor(np.array(batch), dtype=torch.float32)
        batch_all.append(batch)
    return batch_all

def interact_batch_gen_ugv(step_input_dict, dict_key_list):
    batch_all = []
    for dict_key in dict_key_list:
        batch = []
        for (key, item) in step_input_dict.items():
            batch.append(item[dict_key])
        # batch = torch.tensor(np.array(batch), dtype=torch.float32)
        batch_all.append(batch)
    return batch_all

class Machine:
    def __init__(self, ip, port, username, kwd, code_root_path, python_root_path):
        self.ip = ip
        self.port = port
        self.username = username
        self.kwd = kwd
        self.code_root_path = code_root_path
        self.python_root_path = python_root_path


machine_list = []
# 毕设 77 56 65 59
# 实验 75 66 50
machine_list.append(Machine('10.1.114.75', 22, 'liuchi', 'LIUCHI-linc-2021!', '/data1/wjf',
                            '/home/liuchi/anaconda3/envs/ugv/bin/python3'))
# machine_list.append(Machine('10.1.114.65', 22, 'linc', 'liuchi123456', '/home/linc/wjf',
#                             '/home/linc/anaconda3/envs/wjf38/bin/python3'))
# machine_list.append(Machine('10.1.114.77', 22, 'liuchi', 'LIUCHI-linc-2021!', '/data2/wjf',
#                             '/home/liuchi/miniconda3/envs/wjf38/bin/python3'))
#
# machine_list.append(Machine('10.1.114.76', 22, 'liuchi', 'LIUCHI-linc-2021!', '/data2/wjf',
#                             '/home/liuchi/miniconda3/envs/wjf37/bin/python3'))
#
#
# machine_list.append(Machine('10.1.114.59', 22, 'liuchi', 'liuchi123456', '/home/liuchi/wjf',
#                             '/home/liuchi/anaconda3/envs/wjf38/bin/python3'))
# machine_list.append(Machine('10.1.114.50', 22, 'liuchi', '#86xf$50s!', '/data2/wjf',
#                             '/home/liuchi/anaconda3/envs/wjf36/bin/python3'))
# machine_list.append(Machine('10.1.114.56', 22, 'liuchi', 'LIUCHI-linc-2021!', '/data2/wjf',
#                             '/home/liuchi/miniconda3/envs/wjf37/bin/python3'))
#
# machine_list.append(Machine('10.1.114.66', 22, 'liuchi', 'liuchi-linc2022', '/data2/wjf',
#                             '/home/liuchi/anaconda3/envs/wjf36/bin/python3'))

# ..

# machine_list.append(Machine('192.168.42.178', 22, 'wangyu', 'woshiyushen@2333',
#                             '/media/wangyu/697ba05e-c007-4ab3-b361-60bc531773fa/projects',
#                             'home/wangyu/anaconda3/envs/wypt3.7/bin/python3.7'))
# machine_list.append(Machine('10.4.20.55', 22, 'wangyu', 'woshiyushen@2333',
#                             '/media/wangyu/697ba05e-c007-4ab3-b361-60bc531773fa/projects',
#                             'home/wangyu/anaconda3/envs/wypt3.7/bin/python3.7')) # now is '192.168.42.178'
# machine_list.append(Machine('10.1.114.64', 22, 'omnisky', 'normal@BITlinc310', '/data/wangyu',
#                             '/home/omnisky/anaconda3/envs/wy_pt36/bin/python3.7'))


# machine_list.append(Machine('10.1.114.88', 59022, 'zzy', '12131213', '/home/zzy/zhangjianing',
#                             '/home/zzy/anaconda3/envs/mypytorch/bin/python3.7'))
# machine_list.append(Machine('10.1.114.58', 22, 'linc', 'Linc123!', '/home/linc/wy',
#                             '/home/linc/anaconda3/envs/wy_pt37/bin/python3.7'))
# machine_list.append(Machine('10.1.114.60', 22, '', '', '',
#                             ''))
# machine_list.append(Machine('10.1.114.63', 22, 'linc', 'Linc123!', '',
#                             ''))

# machine_list.append(Machine('10.108.17.185', 22, 'zhangjianing', '12131213', '/home2/zhangjianing',
#                             '/home2/zhangjianing/anaconda3/envs/zjnpt2/bin/python3.7'))
# machine_list.append(Machine('10.108.17.251', 22, 'zhangjianing', '12131213', '/home1/zhangjianing',
#                             '/home1/zhangjianing/anaconda3/envs/mypytorch/bin/python3.7'))
# machine_list.append(Machine('10.108.17.161', 22, 'zhangjianing', '12131213', '/data/zhangjianing',
#                             '/data/zhangjianing/anaconda3/envs/mypytorch/bin/python3.7')) # now is '10.108.17.175'
# machine_list.append(Machine('10.108.17.175', 22, 'zhangjianing', '12131213', '/data/zhangjianing',
#                             '/data/zhangjianing/anaconda3/envs/mypytorch/bin/python3.7'))


# machine_list.append(Machine('10.1.1.117', 22, 'zhangjianing', '12131213', '/data1/zhangjianing',
#                             '/data1/zhangjianing/anaconda3/envs/mypytorch/bin/python3.7'))  # no power


def machine_ip2ssh(machine_ip):
    if machine_ip in ['10.108.17.206', '10.108.17.161']:
        machine_ip = '10.108.17.175'
    for machine in machine_list:
        if machine.ip == machine_ip:
            ssh = paramiko.SSHClient()
            policy = paramiko.AutoAddPolicy
            ssh.set_missing_host_key_policy(policy)
            max_try_times = 1e5
            try_times = 0
            success_flag = False
            start_time = datetime.now()
            while True:
                try:
                    saveerr = sys.stderr
                    fsock = open('error.log', 'w')
                    sys.stderr = fsock
                    if machine.ip == '10.1.114.64':
                        ssh.connect(hostname=machine.ip, port=machine.port, username=machine.username)
                    elif machine.ip == '10.1.1.117':
                        ssh.connect(hostname=machine.ip, port=machine.port, username=machine.username,
                                    password=machine.kwd, timeout=5)
                    else:
                        ssh.connect(hostname=machine.ip, port=machine.port, username=machine.username,
                                    password=machine.kwd)
                    sys.stderr = saveerr
                    fsock.close()
                    success_flag = True
                except:
                    pass
                try_times += 1
                if success_flag or try_times >= max_try_times or (datetime.now() - start_time).seconds > 100:
                    break
            if success_flag:
                sftp = ssh.open_sftp()
                return True, machine, ssh, sftp
            else:
                print(str(datetime.now()).split('.')[0], 'machine_ip2ssh ERROR!', machine_ip)
                return False, -1, -1, -1


def judge_available(machine_ip,
                    conf_dict,
                    mode,
                    total_cpu_num,
                    available_cpu_num,
                    total_gpu_num,
                    free_VRAM_list,
                    my_process_num,
                    ):
    available_flag = True
    available_gpu_id = 0
    dataset_conf = conf_dict['dataset_conf']
    method_conf = conf_dict['method_conf']
    d_time1 = datetime.strptime(str(datetime.now().date()) + '2:00', '%Y-%m-%d%H:%M')
    d_time2 = datetime.strptime(str(datetime.now().date()) + '8:00', '%Y-%m-%d%H:%M')
    cur_time = datetime.now()
    if mode == 'train':
        if machine_ip in ['10.1.114.65']:
            # 12 core, tini
            cpu_needed = method_conf['env_num'] + 1 + total_cpu_num * 0.05  # threads
        elif machine_ip in ['10.1.114.59']:
            # 12 core, tini
            cpu_needed = method_conf['env_num'] + 1 + total_cpu_num * 0.5
        elif machine_ip in ['10.1.114.56']:
            # 88 core, slow !!
            cpu_needed = method_conf['env_num'] + 1 + total_cpu_num * 0.65  # threads
        elif machine_ip in ['10.1.114.75']:
            # 112 core, liuchi
            cpu_needed = method_conf['env_num'] + 1 + total_cpu_num * 0.5
        elif machine_ip in ['10.1.114.50']:
            # 72 core, lishuang
            cpu_needed = method_conf['env_num'] + 1 + total_cpu_num * 0.65
        elif machine_ip in ['10.1.114.66']:
            # 96 core
            cpu_needed = method_conf['env_num'] + 1 + total_cpu_num * 0.65
        elif machine_ip in ['10.1.114.77']:
            # 112 core, liuchi
            cpu_needed = method_conf['env_num'] + 1 + total_cpu_num * 0.6
        else:
            cpu_needed = method_conf['env_num'] + 1 + total_cpu_num * 0.6  # threads

        if machine_ip in ['10.1.114.77']:
            if my_process_num >= 0:
                available_cpu_num = 0
        elif machine_ip in ['10.1.114.75']:
            if my_process_num >= 0:
                available_cpu_num = 0
        elif machine_ip in ['10.1.114.76']:
            if my_process_num >= 6:
                available_cpu_num = 0
        elif machine_ip in ['10.1.114.56']:
            if my_process_num >= 2:
                available_cpu_num = 0
        elif machine_ip in ['10.1.114.50']:
            if my_process_num >= 2:
                available_cpu_num = 0
        elif machine_ip in ['10.1.114.66']:
            if my_process_num >= 0:
                available_cpu_num = 0
        if machine_ip in ['10.4.20.55', '10.1.114.88']:
            available_cpu_num = 0
        # if cur_time < d_time1 or cur_time > d_time2:
        #     if machine_ip in ['10.1.114.56', '10.1.114.66', '10.1.114.76']:
        #         available_cpu_num = min(int(total_cpu_num * 0.7) - my_process_num * cpu_needed, available_cpu_num)
        #     if machine_ip in ['10.1.114.64']:
        #         available_cpu_num = 0
        #     if machine_ip in ['10.1.114.77']:
        #         if my_process_num > 1:
        #             available_cpu_num = 0
        # if available_cpu_num < cpu_needed and my_process_num > 0:
        if available_cpu_num < cpu_needed:
            available_flag = False
            return available_flag, available_gpu_id

        gpu_needed = 8192  # MB
        max_free_VRAM = 0
        for gpu_id in range(total_gpu_num):
            if machine_ip == '10.1.114.75' and gpu_id in [0, 1]:
                continue
            if machine_ip == '10.1.114.56' and gpu_id in [0]:
                continue
            if machine_ip == '10.1.114.66' and gpu_id in [0]:
                continue
            if machine_ip == '10.1.114.50' and gpu_id in [0, 1, 2]:
                continue
            # if machine_ip == '10.108.17.251' and gpu_id in [2, 3]:
            #     continue
            # if cur_time < d_time1 or cur_time > d_time2:
            #     if machine_ip == '10.1.114.64' and gpu_id == 0:
            #         continue
            #     if machine_ip == '10.1.114.75' and gpu_id == 7:
            #         continue
            #     # if machine_ip == '10.1.114.76' and gpu_id == 0:
            #     #     continue
            if free_VRAM_list[gpu_id] > max_free_VRAM:
                available_gpu_id = gpu_id
                max_free_VRAM = free_VRAM_list[gpu_id]
        if max_free_VRAM < gpu_needed:
            available_flag = False
            available_gpu_id = 0
            return available_flag, available_gpu_id
    elif mode == 'test':
        cpu_needed = method_conf['test_num'] + 1  # threads
        # if my_process_num > 0:
        if available_cpu_num < cpu_needed and my_process_num > 0:
            available_flag = False
            return available_flag, available_gpu_id
    return available_flag, available_gpu_id


def timestamp2datetime_str(timestamp):
    if timestamp is not None:
        timeStruct = time.localtime(timestamp)
        return time.strftime('%Y-%m-%d %H:%M:%S', timeStruct)
    else:
        return 'None'


def update_log_master2local_machine(project_path, local_machine_ip, focus_time_min=None, focus_time_max=None):
    cur_log_master_path = project_path + '_log_master'
    for machine in machine_list:
        if machine.ip == local_machine_ip:
            local_log_master_path = os.path.join(machine.code_root_path, cur_log_master_path.split('/')[-1])
            sftp_update_put_dir(machine.ip, cur_log_master_path, local_log_master_path, focus_time_min=focus_time_min,
                                focus_time_max=focus_time_max)
            break


def update_code2machines(project_path, ignored_dir_list):
    local_dir_path = project_path
    local_project_name = project_path.split('/')[-1]
    for machine in machine_list:
        print(machine.ip)
        remote_project_name = local_project_name.split('_')[0] + '_' + machine.ip.replace('.', '-')
        if local_project_name != remote_project_name:
            remote_dir_path = os.path.join(machine.code_root_path, remote_project_name)
            sftp_put_dir(machine.ip, local_dir_path, remote_dir_path, ignored_dir_list)


# 从远程服务器获取文件到本地
def sftp_get_file(machine_ip, remotefile_path, localfile_path):
    connect_flag, machine, ssh, sftp = machine_ip2ssh(machine_ip)
    if connect_flag:
        try:
            sftp.get(remotefile_path, localfile_path)
        except:
            print(str(datetime.now()).split('.')[0], 'sftp_get_file ERROR1!', machine_ip, remotefile_path,
                  localfile_path)
            print(traceback.format_exc())
        finally:
            ssh.close()
    else:
        print(str(datetime.now()).split('.')[0], 'sftp_get_file ERROR2!', machine_ip, remotefile_path, localfile_path)


# 从本地上传文件到远程服务器
def sftp_put_file(machine_ip, localfile_path, remotefile_path):
    connect_flag, machine, ssh, sftp = machine_ip2ssh(machine_ip)
    if connect_flag:
        try:
            sftp.put(localfile_path, remotefile_path)
        except:
            print(str(datetime.now()).split('.')[0], 'sftp_put_file ERROR1!', machine_ip, localfile_path,
                  remotefile_path)
            print(traceback.format_exc())
        finally:
            ssh.close()
    else:
        print(str(datetime.now()).split('.')[0], 'sftp_put_file ERROR2!', machine_ip, localfile_path, remotefile_path)


# 递归遍历远程服务器指定目录下的所有文件
def _get_all_files_in_remote_dir(sftp, remote_dir_path, ignored_dir_list=None):
    all_files = list()
    if remote_dir_path[-1] == '/':
        remote_dir_path = remote_dir_path[:-1]
    files = sftp.listdir_attr(remote_dir_path)
    for file in files:
        filename = remote_dir_path + '/' + file.filename
        needed = True
        if ignored_dir_list is not None:
            for ignored_dir in ignored_dir_list:
                if remote_dir_path + '/' + ignored_dir in filename:
                    needed = False
                    break
        if needed:
            if stat.S_ISDIR(file.st_mode):  # 如果是文件夹的话递归处理
                all_files.extend(_get_all_files_in_remote_dir(sftp, filename, ignored_dir_list))
            else:
                all_files.append(filename)
    return all_files


def sftp_get_dir(machine_ip, remote_dir_path, local_dir_path, ignored_dir_list=None):
    connect_flag, machine, ssh, sftp = machine_ip2ssh(machine_ip)
    if connect_flag:
        try:
            if remote_dir_path[-1] == "/":
                remote_dir_path = remote_dir_path[:-1]
            if local_dir_path[-1] == "/":
                local_dir_path = local_dir_path[:-1]
            all_files = _get_all_files_in_remote_dir(sftp, remote_dir_path, ignored_dir_list)
            for file in all_files:
                local_filename = file.replace(remote_dir_path, local_dir_path)
                local_filepath = os.path.dirname(local_filename)
                if not os.path.exists(local_filepath):
                    command = 'mkdir -p %s' % local_filepath
                    ret = os.popen(command)
                    ret.readlines()
                sftp.get(file, local_filename)
        except:
            print(str(datetime.now()).split('.')[0], 'sftp_get_dir ERROR1!', machine_ip, remote_dir_path,
                  local_dir_path, ignored_dir_list)
            print(traceback.format_exc())
        finally:
            ssh.close()
    else:
        print(str(datetime.now()).split('.')[0], 'sftp_get_dir ERROR2!', machine_ip, remote_dir_path, local_dir_path,
              ignored_dir_list)


# 递归遍历本地服务器指定目录下的所有文件
def _get_all_files_in_local_dir(local_dir_path, ignored_dir_list=None, focus_time_min=None, focus_time_max=None):
    all_files = list()
    for root, dirs, files in tqdm(os.walk(local_dir_path, topdown=True)):
        for file in files:
            filename = os.path.join(root, file)
            needed = True
            if ignored_dir_list is not None:
                for ignored_dir in ignored_dir_list:
                    if os.path.join(local_dir_path, ignored_dir) in filename:
                        needed = False
                        break
            if focus_time_max is not None:
                if os.path.getmtime(filename) > focus_time_max:
                    needed = False
            if focus_time_min is not None:
                if os.path.getmtime(filename) < focus_time_min:
                    needed = False
            if needed:
                all_files.append(filename)
    return all_files


def sftp_put_dir(machine_ip, local_dir_path, remote_dir_path, ignored_dir_list=None):
    connect_flag, machine, ssh, sftp = machine_ip2ssh(machine_ip)
    if connect_flag:
        try:
            if remote_dir_path[-1] == "/":
                remote_dir_path = remote_dir_path[:-1]
            if local_dir_path[-1] == "/":
                local_dir_path = local_dir_path[:-1]
            all_files = _get_all_files_in_local_dir(local_dir_path, ignored_dir_list=ignored_dir_list)
            for file in all_files:
                remote_filename = file.replace(local_dir_path, remote_dir_path)
                remote_path = os.path.dirname(remote_filename)
                try:
                    sftp.stat(remote_path)
                except:
                    command = 'mkdir -p %s' % remote_path
                    stdin, stdout, stderr = ssh.exec_command(command)
                    out, err = stdout.read(), stderr.read()
                sftp.put(file, remote_filename)
        except:
            print(str(datetime.now()).split('.')[0], 'sftp_put_dir ERROR1!', machine_ip, local_dir_path,
                  remote_dir_path, ignored_dir_list)
            print(traceback.format_exc())
        finally:
            ssh.close()
    else:
        print(str(datetime.now()).split('.')[0], 'sftp_put_dir ERROR2!', machine_ip, local_dir_path, remote_dir_path,
              ignored_dir_list)


def sftp_update_put_dir(machine_ip, local_dir_path, remote_dir_path, focus_time_min=None, focus_time_max=None):
    connect_flag, machine, ssh, sftp = machine_ip2ssh(machine_ip)
    if connect_flag:
        try:
            if remote_dir_path[-1] == "/":
                remote_dir_path = remote_dir_path[:-1]
            if local_dir_path[-1] == "/":
                local_dir_path = local_dir_path[:-1]
            all_files = _get_all_files_in_local_dir(local_dir_path, focus_time_min=focus_time_min,
                                                    focus_time_max=focus_time_max)
            for file in tqdm(all_files):
                remote_filename = file.replace(local_dir_path, remote_dir_path)
                remote_path = os.path.dirname(remote_filename)
                try:
                    sftp.stat(remote_path)
                except:
                    command = 'mkdir -p %s' % remote_path
                    stdin, stdout, stderr = ssh.exec_command(command)
                    out, err = stdout.read(), stderr.read()
                try:
                    sftp.stat(remote_filename)
                except:
                    sftp.put(file, remote_filename)
        except:
            print(str(datetime.now()).split('.')[0], 'sftp_update_put_dir ERROR1!', machine_ip, local_dir_path,
                  remote_dir_path, timestamp2datetime_str(focus_time_min), timestamp2datetime_str(focus_time_max))
            print(traceback.format_exc())
        finally:
            ssh.close()
    else:
        print(str(datetime.now()).split('.')[0], 'sftp_update_put_dir ERROR2!', machine_ip, local_dir_path,
              remote_dir_path, timestamp2datetime_str(focus_time_min), timestamp2datetime_str(focus_time_max))


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def gen_p_list(tr_lb_sth_list):
    if tr_lb_sth_list[0] == 'Nan':
        tr_lb_num = len(tr_lb_sth_list)
        p_list = [1 / tr_lb_num] * tr_lb_num
    else:
        tr_lb_sth_array = np.array(tr_lb_sth_list, dtype=np.float64)
        min_sth = np.min(tr_lb_sth_array)
        tr_lb_sth_array -= min_sth
        tr_lb_sth_array += 1e-4
        sum_sth = np.sum(tr_lb_sth_array)
        tr_lb_sth_array /= sum_sth
        p_list = list(tr_lb_sth_array)
    return p_list


def judge_health(machine_ip, command, mode, conf_dict):
    health_flag = False
    for machine in machine_list:
        if machine.ip == machine_ip:
            connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
            if connect_flag:
                try:
                    stdin, stdout, stderr = ssh.exec_command(
                        "ps -ef --sort=start_time | grep \" " + command + "\" | grep -v grep")
                    out, err = stdout.read().decode(), stderr.read().decode()
                    process_info_list = out.split('\n')[:-1]
                    if len(process_info_list) > 0:
                        pid = int(process_info_list[-1].split()[1])
                        stdin, stdout, stderr = ssh.exec_command(
                            "ps -ef | grep \" " + command.split()[0] + " \" | grep \" " + str(
                                pid) + " \" | grep -v grep")
                        out, err = stdout.read().decode(), stderr.read().decode()
                        more_process_info_list = out.split('\n')[:-1]
                        more_process_maximum = 0
                        if mode == 'train':
                            more_process_maximum = conf_dict['method_conf']['env_num'] + 1
                        elif mode == 'test':
                            more_process_maximum = conf_dict['method_conf']['test_num'] / 10
                        if len(more_process_info_list) >= more_process_maximum:
                            health_flag = True
                except:
                    print(str(datetime.now()).split('.')[0], 'judge_health ERROR1!', machine_ip, command)
                finally:
                    ssh.close()
            else:
                print(str(datetime.now()).split('.')[0], 'judge_health ERROR2!', machine_ip, command)
            break
    return health_flag


def find_all_myprocess(finder_name):
    for machine in machine_list:
        connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
        if connect_flag:
            try:
                stdin, stdout, stderr = ssh.exec_command(
                    "ps -ef | grep \" " + machine.python_root_path + " \" | grep -v grep | grep -v \"" + finder_name + "\"")
                out, err = stdout.read().decode(), stderr.read().decode()
                if out.count(machine.python_root_path) > 0:
                    print(datetime.now(), machine.ip, out.count(machine.python_root_path))
            except:
                print(str(datetime.now()).split('.')[0], 'find_all_myprocess ERROR1!', finder_name, machine.ip)
            finally:
                ssh.close()
        else:
            print(str(datetime.now()).split('.')[0], 'find_all_myprocess ERROR2!', finder_name, machine.ip)


def find_all_myexp(finder_name):
    for machine in machine_list:
        connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
        if connect_flag:
            try:
                stdin, stdout, stderr = ssh.exec_command(
                    "ps -ef | grep \" " + machine.python_root_path + ' ' + machine.code_root_path + "\" | grep -v grep | grep -v \"" + finder_name + "\"")
                out, err = stdout.read().decode(), stderr.read().decode()
                if out.count(machine.python_root_path) > 0:
                    print(datetime.now(), machine.ip, out.count(machine.python_root_path))
                    print(out)
            except:
                print(str(datetime.now()).split('.')[0], 'find_all_myprocess ERROR1!', finder_name, machine.ip)
            finally:
                ssh.close()
        else:
            print(str(datetime.now()).split('.')[0], 'find_all_myprocess ERROR2!', finder_name, machine.ip)


def kill_all(killer_name):
    for machine in machine_list:
        connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
        if connect_flag:
            try:
                print(datetime.now(), machine.ip)
                stdin, stdout, stderr = ssh.exec_command(
                    "ps -ef | grep \" " + machine.python_root_path + " \" | grep -v grep | grep -v \""
                    + killer_name + "\" | awk '{print $2}' | xargs kill -9")
                # print(stdout.read().decode())
                # print(stderr.read().decode())
            except:
                print(str(datetime.now()).split('.')[0], 'kill_all ERROR1!', killer_name, machine.ip)
            finally:
                ssh.close()
        else:
            print(str(datetime.now()).split('.')[0], 'kill_all ERROR2!', killer_name, machine.ip)


def set_dict_value(mydict, keys, val):
    mydict_tmp = mydict
    lastkey = keys[-1]
    for key in keys[:-1]:
        mydict_tmp = mydict_tmp[key]
    if val == 'True':
        mydict_tmp[lastkey] = True
    elif val == 'False':
        mydict_tmp[lastkey] = False
    else:
        mydict_tmp[lastkey] = type(mydict_tmp[lastkey])(val)


def check_dict_key(mydict, keys):
    mydict_tmp = mydict
    flag = True
    for key in keys:
        if not isinstance(mydict_tmp, dict) or key not in mydict_tmp:
            flag = False
            break
        else:
            mydict_tmp = mydict_tmp[key]
    return flag


def gen_conf(args, conf_temp):
    conf = copy.deepcopy(conf_temp)
    for attr in dir(args):
        if attr == 'param_name':
            param_list = getattr(args, attr).split('___')
            for param in param_list:
                if param == 'param_name':
                    if check_dict_key(conf, [param]):
                        set_dict_value(conf, [param], getattr(args, attr))
                    continue
                keys = param.split('__')[:-1]
                val = param.split('__')[-1]
                if check_dict_key(conf, keys):
                    set_dict_value(conf, keys, val)
        else:
            keys = attr.split('__')
            if check_dict_key(conf, keys):
                set_dict_value(conf, keys, getattr(args, attr))
    return conf


def gen_seperate_param_name(param_name, conf_temp):
    param_list = param_name.split('___')
    seperate_param_name = 'spn'
    for param in param_list:
        if param != 'param_name':
            keys = param.split('__')[:-1]
            if check_dict_key(conf_temp, keys):
                seperate_param_name += '___' + param
    return seperate_param_name


def gen_param_name(conf_patch):
    param_name = 'param_name'
    for param in sorted(conf_patch):
        if param != 'dataset_name' and param != 'method_name':
            param_name += '___' + param + '__' + str(conf_patch[param])
    return param_name


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        if ip == '10.1.114.92':
            ip = '10.1.114.77'
        if ip == '192.168.0.159':
            ip = '10.1.114.88'
    finally:
        s.close()
        return ip


def find_finished_exp(log_master_ip, code_version, dataset_name, method_name, param_name, mode):
    found_exp_path_list = []
    for machine in machine_list:
        if machine.ip == log_master_ip:
            # vague_exp_path = \
            #     os.path.join(machine.code_root_path,
            #                  'UGVUAV' + code_version + '_' + log_master_ip.replace('.', '-') + '_log_master',
            #                  dataset_name, method_name, param_name) + '___ip__' + '*'
            vague_exp_path = \
                os.path.join(machine.code_root_path,
                             'UGVUAV' + code_version + '_log_master',
                             dataset_name, method_name, param_name) + '___ip__' + '*'

            if get_host_ip() == log_master_ip:
                for exact_exp_path in sorted(glob.glob(vague_exp_path)):
                    check_path = os.path.join(exact_exp_path, 'check.npy')
                    finished_flag = False
                    if os.path.exists(check_path):
                        check = np.load(check_path, allow_pickle=True)[()]
                        if mode in check:
                            if check[mode] == True:
                                finished_flag = True
                    if finished_flag:
                        # print(exact_exp_path)
                        found_exp_path_list.append(exact_exp_path)
                    else:
                        if mode == 'trained':
                            # shutil.rmtree(exact_exp_path)
                            pass

            else:
                # log_master和本进程不在一个服务器上的时候
                pass
            break
    return found_exp_path_list


def find_finished_exp_vague(log_master_ip, code_version, dataset_name, method_name, must_hyper_param_list,
                            must_not_hyper_param_list):
    found_exp_path_list = []
    for machine in machine_list:
        if machine.ip == log_master_ip:
            # vague_exp_path = \
            #     os.path.join(machine.code_root_path,
            #                  'UGVUAV' + code_version + '_' + log_master_ip.replace('.', '-') + '_log_master',
            #                  dataset_name, method_name) + '/param_name*'
            vague_exp_path = \
                os.path.join(machine.code_root_path,
                             'UGVUAV' + code_version + '_log_master',
                             dataset_name, method_name) + '/param_name*'

            if get_host_ip() == log_master_ip:
                for exact_exp_path in sorted(glob.glob(vague_exp_path)):
                    param_name = exact_exp_path.split('/')[-1]
                    param_list = param_name.split('___')[1:]
                    must_hyper_param_num = 0
                    must_not_hyper_param_num = 0
                    for param in param_list:
                        if param in must_hyper_param_list:
                            must_hyper_param_num += 1
                        if param in must_not_hyper_param_list:
                            must_not_hyper_param_num += 1
                        for must_not_hyper_param in must_not_hyper_param_list:
                            if must_not_hyper_param[-1] == '*' and must_not_hyper_param[:-1] in param:
                                must_not_hyper_param_num += 1
                                break

                    if must_hyper_param_num == len(must_hyper_param_list) and must_not_hyper_param_num == 0:
                        check_path = os.path.join(exact_exp_path, 'check.npy')
                        finished_flag = False
                        if os.path.exists(check_path):
                            check = np.load(check_path, allow_pickle=True)[()]
                            if 'trained' in check:
                                if check['trained'] == True:
                                    finished_flag = True
                            # if 'tested' in check:
                            #     if check['tested'] == True:
                            #         finished_flag = True
                        if finished_flag:
                            found_exp_path_list.append(exact_exp_path)
            else:
                # log_master和本进程不在一个服务器上的时候
                pass
            break
    return found_exp_path_list


def gen_conf_patches(vary_hyper_param_dict, common_hyper_param_dict):
    tmp_conf_patch_list = []
    tmp_conf_patch_list.append(common_hyper_param_dict)
    for vary_hyper_param in vary_hyper_param_dict:
        old_tmp_conf_patch_list = copy.deepcopy(tmp_conf_patch_list)
        tmp_conf_patch_list = []
        for old_conf_patch in old_tmp_conf_patch_list:
            for vary_hyper_param_value in vary_hyper_param_dict[vary_hyper_param]:
                new_conf_patch = copy.deepcopy(old_conf_patch)
                if not (vary_hyper_param == 'uav_ugv_max_dis' and vary_hyper_param_value == 600) \
                        and not (vary_hyper_param == 'uav_uav_max_dis' and vary_hyper_param_value == 400) \
                        and not (vary_hyper_param == 'uav_return_max_dis' and vary_hyper_param_value == 400) \
                        and not (vary_hyper_param == 'max_ugv_move_dis_each_step' and vary_hyper_param_value == 300) \
                        and not (vary_hyper_param == 'wait_step_num' and vary_hyper_param_value == 10) \
                        and not (vary_hyper_param == 'UGV_UAVs_Group_num' and vary_hyper_param_value == 0) \
                        and not (vary_hyper_param == 'uav_num_each_group' and vary_hyper_param_value == 0) \
                        and not (vary_hyper_param == 'lr' and abs(vary_hyper_param_value - 6.25e-5) < 1e-10) \
                        and not (vary_hyper_param == 'eps' and abs(vary_hyper_param_value - 1e-5) < 1e-10) \
                        and not (vary_hyper_param == 'gamma' and abs(vary_hyper_param_value - 0.99) < 1e-10) \
                        and not (vary_hyper_param == 'tau' and abs(vary_hyper_param_value - 0.95) < 1e-10) \
                        and not (vary_hyper_param == 'value_loss_coef' and abs(vary_hyper_param_value - 0.5) < 1e-10) \
                        and not (vary_hyper_param == 'reconstruction_coef' and abs(vary_hyper_param_value - 0.) < 1e-10) \
                        and not (vary_hyper_param == 'max_grad_norm' and abs(vary_hyper_param_value - 0.5) < 1e-10) \
                        and not (vary_hyper_param == 'clip_param' and abs(vary_hyper_param_value - 0.1) < 1e-10) \
                        and not (vary_hyper_param == 'ugv_max_positive_reward' and abs(vary_hyper_param_value - 25) < 1e-10) \
                        and not (vary_hyper_param == 'ugv_positive_factor' and abs(vary_hyper_param_value - 3) < 1e-10) \
                        and not (vary_hyper_param == 'ugv_penalty_factor' and abs(vary_hyper_param_value - 0.1) < 1e-10) \
                        and not (vary_hyper_param == 'uav_max_positive_reward' and abs(vary_hyper_param_value - 15) < 1e-10) \
                        and not (vary_hyper_param == 'uav_penalty_factor' and abs(vary_hyper_param_value - 0.25) < 1e-10) \
                        and not (vary_hyper_param == 'dsp_q' and vary_hyper_param_value == 5) \
                        and not (vary_hyper_param == 'stop_hidden_size' and vary_hyper_param_value == 8) \
                        and not (vary_hyper_param == 'GNN_layer_num' and vary_hyper_param_value == 3) \
                        and not (vary_hyper_param == 'Comm_layer_num' and vary_hyper_param_value == 3)\
                        and not (vary_hyper_param == 'buffer_replay_time' and vary_hyper_param_value == 2):
                        # and not (vary_hyper_param == 'env_num')\
                        # and not (vary_hyper_param == 'train_iter'):
                        # and not (vary_hyper_param == 'uav_positive_factor' and abs(vary_hyper_param_value - 3) < 1e-10):
                        # and not (vary_hyper_param == 'entropy_coef' and abs(vary_hyper_param_value - 0.01) < 1e-10) \
                    new_conf_patch[vary_hyper_param] = vary_hyper_param_value
                tmp_conf_patch_list.append(copy.deepcopy(new_conf_patch))
    return copy.deepcopy(tmp_conf_patch_list)


def gen_fully_conf_patch_list(conf_patch_list, log_master_ip, code_version, mode):
    fully_conf_patch_list = []
    print('-------> conf_patch_list:', conf_patch_list)
    for conf_patch in conf_patch_list:
        tmp_dict = {}
        tmp_dict['conf_patch'] = conf_patch
        param_name = gen_param_name(conf_patch)
        tmp_dict['state_time'] = mp.Value('f', 0.0)
        tmp_dict['state_time'].value = time.time()
        tmp_dict['need_print'] = mp.Value('b', False)
        tmp_dict['param_name'] = param_name
        tmp_dict['machine_ip'] = ''

        tmp_dict['conf_dict'] = {}
        dataset_conf = importlib.import_module('datasets.' + conf_patch['dataset_name'] + '.conf_temp').DATASET_CONF
        env_conf = importlib.import_module('env.conf_temp').ENV_CONF
        method_conf = importlib.import_module('methods.' + conf_patch['method_name'] + '.conf_temp').METHOD_CONF
        log_conf = importlib.import_module('log.conf_temp').LOG_CONF

        class Demo(object):
            pass

        args = Demo()
        args.param_name = param_name
        tmp_dict['conf_dict']['dataset_conf'] = gen_conf(args, dataset_conf)
        tmp_dict['conf_dict']['env_conf'] = gen_conf(args, env_conf)
        tmp_dict['conf_dict']['method_conf'] = gen_conf(args, method_conf)
        tmp_dict['conf_dict']['log_conf'] = gen_conf(args, log_conf)

        found_exp_path_list = find_finished_exp(log_master_ip, code_version, conf_patch['dataset_name'],
                                                conf_patch['method_name'], param_name, mode + 'ed')
        if len(found_exp_path_list) == 0:
            tmp_dict['state'] = mp.Value('i', 0)
            if mode == 'test':
                found_trained_exp_path_list = find_finished_exp(log_master_ip, code_version, conf_patch['dataset_name'],
                                                                conf_patch['method_name'], param_name, 'train' + 'ed')
                if len(found_trained_exp_path_list) == 0:
                    tmp_dict['state'].value = 4
                    tmp_dict['need_print'].value = True
                else:
                    tmp_dict['machine_ip'] = found_trained_exp_path_list[0].split('/')[-1].split('___')[-1].split('__')[
                        -1].replace('-', '.')
        else:
            tmp_dict['state'] = mp.Value('i', 3)
            tmp_dict['need_print'].value = True
            tmp_dict['machine_ip'] = found_exp_path_list[0].split('/')[-1].split('___')[-1].split('__')[-1].replace('-',
                                                                                                                    '.')
        fully_conf_patch_list.append(tmp_dict)
    return fully_conf_patch_list


def gen_fully_machine_state_dict(machine_list):
    fully_machine_state_dict = {}
    for machine in machine_list:
        fully_machine_state_dict[machine.ip] = {}
        fully_machine_state_dict[machine.ip]['keep_alive'] = mp.Value('b', True)
        fully_machine_state_dict[machine.ip]['need_refresh'] = mp.Value('b', True)
        fully_machine_state_dict[machine.ip]['need_pause'] = mp.Value('b', False)
        fully_machine_state_dict[machine.ip]['info'] = {}
        fully_machine_state_dict[machine.ip]['info']['total_cpu_num'] = mp.Value('i', 0)
        fully_machine_state_dict[machine.ip]['info']['available_cpu_num'] = mp.Value('i', 0)
        fully_machine_state_dict[machine.ip]['info']['total_gpu_num'] = mp.Value('i', 0)
        fully_machine_state_dict[machine.ip]['info']['free_VRAM_list'] = mp.Array('i', 20)
        fully_machine_state_dict[machine.ip]['info']['my_process_num'] = mp.Value('i', 0)

        connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
        if connect_flag:
            try:
                stdin, stdout, stderr = ssh.exec_command('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free')
                out = stdout.read().decode()
                free_gpu_list = re.findall('[0-9]+', out)
                free_gpu_list = [int(free_gpu) for free_gpu in free_gpu_list]
                fully_machine_state_dict[machine.ip]['info']['total_gpu_num'].value = len(free_gpu_list)
                fully_machine_state_dict[machine.ip]['info']['free_VRAM_list'] = mp.Array('i', len(free_gpu_list))
                for gpu_id, free_gpu in enumerate(free_gpu_list):
                    fully_machine_state_dict[machine.ip]['info']['free_VRAM_list'][gpu_id] = free_gpu
                stdin, stdout, stderr = ssh.exec_command(
                    "ps -ef | grep \" " + machine.python_root_path + " " + machine.code_root_path +
                    "\" | grep \"main_slave.py\" | grep -v grep")
                out, err = stdout.read().decode(), stderr.read().decode()
                fully_machine_state_dict[machine.ip]['info']['my_process_num'].value = out.count(
                    machine.python_root_path)
            except:
                print(str(datetime.now()).split('.')[0], 'gen_fully_machine_state_dict ERROR1!', machine.ip)
            finally:
                ssh.close()
        else:
            print(str(datetime.now()).split('.')[0], 'gen_fully_machine_state_dict ERROR2!', machine.ip)
    return fully_machine_state_dict


def get_available_cpu_num(machine_ip):
    connect_flag, machine, ssh, sftp = machine_ip2ssh(machine_ip)
    if connect_flag:
        try:
            used_cpu_num_list = []
            used_cpu_num_list2 = []
            for _ in range(2):
                stdin, stdout, stderr = ssh.exec_command('cat /proc/loadavg')
                time.sleep(1)
                tmp_out = stdout.read().decode()
                used_cpu_num_list.append(float(tmp_out.split()[0]))
                used_cpu_num_list2.append(float(tmp_out.split()[3].split('/')[0]))
            used_cpu_num = int(max(max(used_cpu_num_list), mean(used_cpu_num_list2)))
            stdin, stdout, stderr = ssh.exec_command("grep 'processor' /proc/cpuinfo | sort -u | wc -l")
            total_cpu_num = int(float(stdout.read().decode()))
            available_cpu_num = total_cpu_num - used_cpu_num
            ssh.close()
            return available_cpu_num
        except:
            print(str(datetime.now()).split('.')[0], 'get_available_cpu_num ERROR1!', machine_ip)
            return -1
    else:
        print(str(datetime.now()).split('.')[0], 'get_available_cpu_num ERROR2!', machine_ip)
        return -1


def machine_state_refresh_subp(
        machine_ip,
        keep_alive,
        need_refresh,
        need_pause,
        total_cpu_num,
        available_cpu_num,
        total_gpu_num,
        free_VRAM_list,
        my_process_num,
):
    refresh_time = datetime.now()
    while keep_alive.value:
        if need_pause.value:
            time.sleep(1)
        else:
            if need_refresh.value:
                connect_flag, machine, ssh, sftp = machine_ip2ssh(machine_ip)
                if connect_flag:
                    try:
                        used_cpu_num_list = []
                        used_cpu_num_list2 = []
                        for _ in range(3):
                            stdin, stdout, stderr = ssh.exec_command('cat /proc/loadavg')
                            time.sleep(3)
                            tmp_out = stdout.read().decode()
                            used_cpu_num_list.append(float(tmp_out.split()[0]))
                            used_cpu_num_list2.append(float(tmp_out.split()[3].split('/')[0]))
                        used_cpu_num = int(max(max(used_cpu_num_list), mean(used_cpu_num_list2)))
                        stdin, stdout, stderr = ssh.exec_command("grep 'processor' /proc/cpuinfo | sort -u | wc -l")
                        total_cpu_num.value = int(float(stdout.read().decode()))
                        available_cpu_num.value = total_cpu_num.value - used_cpu_num
                        stdin, stdout, stderr = ssh.exec_command('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free')
                        out = stdout.read().decode()
                        free_gpu_list = re.findall('[0-9]+', out)
                        free_gpu_list = [int(free_gpu) for free_gpu in free_gpu_list]
                        total_gpu_num.value = len(free_gpu_list)
                        for gpu_id, free_gpu in enumerate(free_gpu_list):
                            free_VRAM_list[gpu_id] = free_gpu
                        stdin, stdout, stderr = ssh.exec_command(
                            "ps -ef | grep \" " + machine.python_root_path + " "
                            + machine.code_root_path + "\" | grep \"main_slave.py\" | grep -v grep")
                        out, err = stdout.read().decode(), stderr.read().decode()
                        my_process_num.value = out.count(machine.python_root_path)
                        ssh.close()
                        need_refresh.value = False
                        refresh_time = datetime.now()
                    except:
                        print(str(datetime.now()).split('.')[0], 'machine_state_refresh_subp ERROR1!', machine_ip)
                else:
                    print(str(datetime.now()).split('.')[0], 'machine_state_refresh_subp ERROR2!', machine_ip)
            if (datetime.now() - refresh_time).seconds > 60:
                need_refresh.value = True


def remote_exec_exp(machine_ip,
                    gpu_id,
                    log_master_ip,
                    code_version,
                    conf_patch,
                    param_name,
                    state,
                    state_time,
                    need_print,
                    conf_dict,
                    mode,
                    need_refresh,
                    need_pause,
                    ):
    run_success_flag = False
    connect_flag, machine, ssh, sftp = machine_ip2ssh(machine_ip)
    if connect_flag:
        try:
            for machine in machine_list:
                if machine.ip == machine_ip:
                    if machine_ip == '10.1.114.77':
                        project_path = os.path.join(machine.code_root_path,
                                                    'UGVUAV' + code_version + '_' + machine.ip.replace('.', '-'))
                        command = machine.python_root_path + ' ' + os.path.join(
                            project_path, 'main_slave.py')
                        command += ' --log_master_ip ' + log_master_ip
                        command += ' --code_version ' + code_version
                        command += ' --dataset_name ' + conf_patch['dataset_name']
                        command += ' --method_name ' + conf_patch['method_name']
                        command += ' --gpu_id ' + str(0)
                        command += ' --param_name ' + param_name
                        command += ' --mode ' + mode
                        # 后台运行，不会中断
                        super_command = 'CUDA_VISIBLE_DEVICES=' + str(
                            gpu_id) + ' ' + 'nohup ' + command + ' &>/dev/null &'
                    else:
                        project_path = os.path.join(machine.code_root_path,
                                                    'UGVUAV' + code_version + '_' + machine.ip.replace('.', '-'))
                        command = machine.python_root_path + ' ' + os.path.join(
                            project_path, 'main_slave.py')
                        command += ' --log_master_ip ' + log_master_ip
                        command += ' --code_version ' + code_version
                        command += ' --dataset_name ' + conf_patch['dataset_name']
                        command += ' --method_name ' + conf_patch['method_name']
                        command += ' --gpu_id ' + str(gpu_id)
                        command += ' --param_name ' + param_name
                        command += ' --mode ' + mode
                        # 后台运行，不会中断
                        super_command = 'nohup ' + command + ' &>/dev/null &'

                    # file_path = os.path.join(project_path + '_log', conf_patch['dataset_name'],
                    #                          conf_patch['method_name'],
                    #                          param_name + '___ip__' + machine_ip.replace('.', '-'), 'error_log.txt')
                    # super_command = 'nohup ' + command + ' >/dev/null 2>' + file_path + ' &'
                    # if machine_ip == '10.1.114.66':
                    #     print(super_command)
                    ssh.exec_command(super_command)
                    print(super_command)

                    if mode == 'train':
                        time.sleep(60)
                    elif mode == 'test':
                        time.sleep(10)
                    if conf_patch['method_name'] == 'Random':
                        run_success_flag = True
                    elif judge_health(machine.ip, command, mode, conf_dict):
                        run_success_flag = True
                        # wait for report.txt generation
                        if mode == 'train':
                            time.sleep(max(conf_dict['method_conf']['env_num'] * 3 + 10, 60))
                        elif mode == 'test':
                            time.sleep(60)
                    ssh.close()
                    break
        except:
            print(str(datetime.now()).split('.')[0], 'remote_exec_exp ERROR1!')
        finally:
            if run_success_flag:
                state.value = 2
                state_time.value = time.time()
                need_print.value = True
            else:
                state.value = 0
                state_time.value = time.time()
                exp_info = os.path.join(conf_patch['dataset_name'], conf_patch['method_name'], param_name)
                print(str(datetime.now()).split('.')[0], exp_info, 'is ' + mode + ' failed on machine', machine_ip)
            need_refresh.value = True
            need_pause.value = False
    else:
        print(str(datetime.now()).split('.')[0], 'remote_exec_exp ERROR2!')


def distinct_conf_patch_list(conf_patch_list):
    tmp_conf_patch_list = []
    unique_id_dict = {}
    for conf_patch in conf_patch_list:
        key = conf_patch['dataset_name'] + '_' + conf_patch['method_name']
        if key not in unique_id_dict:
            unique_id_dict[key] = []
        param_name = gen_param_name(conf_patch)
        if param_name not in unique_id_dict[key]:
            tmp_conf_patch_list.append(conf_patch)
            unique_id_dict[key].append(param_name)
    return tmp_conf_patch_list


def kill_exp(log_master_ip, code_version, dataset_name, method_name, param_name, mode):
    for machine in machine_list:
        connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
        if connect_flag:
            try:
                project_path = os.path.join(machine.code_root_path,
                                            'UGVUAV' + code_version + '_' + machine.ip.replace('.', '-'))
                command_part1 = machine.python_root_path + ' ' + os.path.join(
                    project_path, 'main_slave.py')
                command_part1 += ' --log_master_ip ' + log_master_ip
                command_part1 += ' --code_version ' + code_version
                command_part1 += ' --dataset_name ' + dataset_name
                command_part1 += ' --method_name ' + method_name
                command_part1 += ' --gpu_id'

                command_part2 = '--param_name ' + param_name
                command_part2 += ' --mode ' + mode

                stdin, stdout, stderr = ssh.exec_command(
                    "ps -ef --sort=start_time | grep \" " + command_part1 +
                    " \" | grep \" " + command_part2 + "\" | grep -v grep")
                out, err = stdout.read().decode(), stderr.read().decode()
                process_info_list = out.split('\n')[:-1]
                for process_info in process_info_list:
                    pid = int(process_info.split()[1])
                    ssh.exec_command(
                        "ps -ef | grep \" " + machine.python_root_path + " \" | grep \" " + str(
                            pid) + " \" | grep -v grep | awk '{print $2}' | xargs kill -9")
            except:
                print(str(datetime.now()).split('.')[0], 'kill_exp ERROR1!', dataset_name, method_name, param_name,
                      machine.ip)
            finally:
                ssh.close()
        else:
            print(str(datetime.now()).split('.')[0], 'kill_exp ERROR2!', machine.ip)


def count_step_time(machine_ip, code_version, dataset_name, method_name, param_name, file_name):
    step_time = 1e6
    for machine in machine_list:
        if machine.ip == machine_ip:
            connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
            if connect_flag:
                try:
                    file_path = os.path.join(machine.code_root_path,
                                             'UGVUAV' + code_version + '_' + machine.ip.replace('.', '-') + '_log',
                                             dataset_name,
                                             method_name,
                                             param_name + '___ip__' + machine.ip.replace('.', '-'),
                                             file_name)
                    sftp.stat(file_path)
                    stdin, stdout, stderr = ssh.exec_command("date -r " + file_path + " +%s")
                    out, err = stdout.read().decode(), stderr.read().decode()
                    last_modified_time = int(out)
                    stdin, stdout, stderr = ssh.exec_command("date +%s")
                    out, err = stdout.read().decode(), stderr.read().decode()
                    cur_time = int(out)
                    step_time = cur_time - last_modified_time
                except:
                    print(str(datetime.now()).split('.')[0], 'count_step_time ERROR1!', machine_ip, code_version,
                          dataset_name, method_name, param_name, file_name)
                finally:
                    ssh.close()
            else:
                step_time = 0
                print(str(datetime.now()).split('.')[0], 'count_step_time ERROR2!', machine_ip)
            break
    return step_time


def check_and_reset(mode,
                    machine_ip,
                    code_version,
                    dataset_name,
                    method_name,
                    param_name,
                    file_name,
                    state,
                    state_time,
                    need_print,
                    conf_dict,
                    ):
    need_redo = False
    env_num = conf_dict['method_conf']['env_num']
    running_time = time.time() - state_time.value
    if mode == 'train' and running_time > 60 * 6:
        step_time = count_step_time(machine_ip,
                                    code_version,
                                    dataset_name,
                                    method_name,
                                    param_name,
                                    file_name
                                    )
        if step_time > 60 * 10:
            need_redo = True
    elif mode == 'test' and running_time > 60 * 5:
        need_redo = True
    if need_redo:
        state.value = 0
        state_time.value = time.time()
        need_print.value = True


def reload_running_exps(log_master_ip, code_version, mode, fully_conf_patch_list):
    for machine in machine_list:
        connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
        if connect_flag:
            try:
                project_path = os.path.join(machine.code_root_path,
                                            'UGVUAV' + code_version + '_' + machine.ip.replace('.', '-'))
                command_part1 = machine.python_root_path + ' ' + os.path.join(project_path, 'main_slave.py')
                command_part1 += ' --log_master_ip ' + log_master_ip
                command_part1 += ' --code_version ' + code_version
                command_part2 = '--mode ' + mode
                print('command:',  "ps -ef --sort=start_time | grep \" " + command_part1 + " \" | grep \" " + command_part2 + "\" | grep -v grep")
                stdin, stdout, stderr = ssh.exec_command(
                    "ps -ef --sort=start_time | grep \" " + command_part1 +
                    " \" | grep \" " + command_part2 + "\" | grep -v grep")
                out, err = stdout.read().decode(), stderr.read().decode()
                exp_info_list = out.split('\n')[:-1]
                for exp_info in exp_info_list:
                    dataset_name = ''
                    method_name = ''
                    param_name = ''
                    for id, exp_info_piece in enumerate(exp_info.split()):
                        if exp_info_piece == '--dataset_name':
                            dataset_name = exp_info.split()[id + 1]
                        if exp_info_piece == '--method_name':
                            method_name = exp_info.split()[id + 1]
                        if exp_info_piece == '--param_name':
                            param_name = exp_info.split()[id + 1]
                    for fully_conf_patch in fully_conf_patch_list:
                        if fully_conf_patch['state'].value == 0 and \
                                fully_conf_patch['param_name'] == param_name and \
                                fully_conf_patch['conf_patch']['dataset_name'] == dataset_name and \
                                fully_conf_patch['conf_patch']['method_name'] == method_name:
                            fully_conf_patch['state'].value = 2
                            fully_conf_patch['state_time'].value = time.time()
                            fully_conf_patch['need_print'].value = True
                            fully_conf_patch['machine_ip'] = machine.ip
            except:
                print(str(datetime.now()).split('.')[0], 'reload_running_exps ERROR1!', machine.ip)
            finally:
                ssh.close()
        else:
            print(str(datetime.now()).split('.')[0], 'reload_running_exps ERROR2!', machine.ip)


def find_running_exp_info(machine):
    exp_num = 0
    process_info_list = []
    connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
    if connect_flag:
        try:
            stdin, stdout, stderr = ssh.exec_command(
                "ps -ef --sort=start_time | grep \" " + machine.python_root_path +
                " \" | grep \" " + "--param_name param_name___" + "\" | grep -v grep")
            out, err = stdout.read().decode(), stderr.read().decode()
            process_info_list = out.split('\n')[:-1]
            exp_num = len(process_info_list)
            print(machine.ip, 'exp_num:', exp_num)
            if exp_num > 0:
                for process_info in process_info_list:
                    print(process_info)
        except:
            print(str(datetime.now()).split('.')[0], 'find_running_exp_info ERROR1!', machine.ip)
        finally:
            ssh.close()
    else:
        print(str(datetime.now()).split('.')[0], 'find_running_exp_info ERROR2!', machine.ip)
    return exp_num, process_info_list


def find_running_process_info(machine):
    exp_num = 0
    process_info_list = []
    connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
    if connect_flag:
        try:
            stdin, stdout, stderr = ssh.exec_command(
                "ps -ef --sort=start_time | grep \" " + machine.python_root_path + " \" | grep -v grep")
            out, err = stdout.read().decode(), stderr.read().decode()
            process_info_list = out.split('\n')[:-1]
            process_num = len(process_info_list)
            print(machine.ip, 'process_num:', process_num)
            if process_num > 0:
                for process_info in process_info_list:
                    print(process_info)
        except:
            print(str(datetime.now()).split('.')[0], 'find_running_process_info ERROR1!', machine.ip)
        finally:
            ssh.close()
    else:
        print(str(datetime.now()).split('.')[0], 'find_running_process_info ERROR2!', machine.ip)
    return exp_num, process_info_list


def kill_exp_by_pid(machine, pid):
    connect_flag, machine, ssh, sftp = machine_ip2ssh(machine.ip)
    if connect_flag:
        try:
            stdin, stdout, stderr = ssh.exec_command(
                "ps -ef --sort=start_time | grep \" " + str(pid) +
                " \" | grep \" " + machine.python_root_path + " \" | grep -v grep")
            out, err = stdout.read().decode(), stderr.read().decode()
            process_info_list = out.split('\n')[:-1]
            for process_info in process_info_list:
                pid = int(process_info.split()[1])
                ssh.exec_command(
                    "ps -ef | grep \" " + machine.python_root_path + " \" | grep \" " + str(
                        pid) + " \" | grep -v grep | awk '{print $2}' | xargs kill -9")
        except:
            print(str(datetime.now()).split('.')[0], 'kill_exp_by_pid ERROR1!', machine.ip, pid)
        finally:
            ssh.close()
    else:
        print(str(datetime.now()).split('.')[0], 'kill_exp_by_pid ERROR2!', machine.ip)

