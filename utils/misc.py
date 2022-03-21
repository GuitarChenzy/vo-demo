#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2017-12-08 10:06:43
Program: 
Description: 
"""
import torch
import shutil
import os
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import init


def pre_create_file_train(model_dir, log_dir, args):
    """
    预先创建模型的训练文件
    :param model_dir: 模型目录
    :param log_dir: 日志目录
    :param args: 参数
    :return:
    """

    dir_models = 'F:\Code\pyProject\deepvo\\' + model_dir + '/' + args.net_name
    dir_logs = log_dir + '/' + args.net_name
    dir_model = dir_models + '/' + args.dir0
    dir_log = dir_logs + '/' + args.dir0
    if not os.path.exists(dir_models):
        os.mkdir(dir_models)
    if not os.path.exists(dir_logs):
        os.mkdir(dir_logs)
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    if os.path.exists(dir_log):
        shutil.rmtree(dir_log)
    os.mkdir(dir_log)
    return dir_model, dir_log


def pre_create_file_test(args):
    """
    预先创建测试文件
    :param args: 参数
    :return:
    """
    dir_net = 'F:\Code\pyProject\deepvo\\test/' + args.net_restore
    dir_time = dir_net + '/' + args.dir_restore + '_' + args.model_restore
    if not os.path.exists(dir_net):
        os.mkdir(dir_net)
    if not os.path.exists(dir_time):
        os.mkdir(dir_time)
    return dir_time


def to_var(x):
    """
    将X放在CPU或者GPU
    :param x: 输入数据
    :return:
    """
    if torch.cuda.is_available():
        return Variable(x).cuda()
    else:
        return Variable(x)


def init_xavier(m):
    """
    初始化参数矩阵
    :param m: 参数矩阵
    :return:
    """
    if isinstance(m, torch.nn.Conv2d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0.0)
    if isinstance(m, torch.nn.Linear):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0.0)
    if isinstance(m, torch.nn.BatchNorm2d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0.0)


def adjust_learning_rate(optimizer, epoch, lr_base, gamma=0.316, epoch_lr_decay=25):
    """
    自动调整学习率
    :param optimizer: 优化方法
    :param epoch: 迭代次数
    :param lr_base: 基础学习率
    :param gamma: gamma值
    :param epoch_lr_decay: 速率衰减次数
    :return:
    """
    """
        epoch       lr
        000-025     1e-4
        025-050     3e-5
        050-075     1e-5
        075-100     3e-6
        100-125     1e-6
        125-150     3e-7
    """

    exp = int(math.floor(epoch / epoch_lr_decay))
    lr_decay = gamma ** exp
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_decay * lr_base


def display_loss_tb(hour_per_epoch, epoch, args, step, step_per_epoch, optimizer, loss, loss1, loss2,
                    loss_list, loss1_list, loss2_list, writer, step_global):
    """
    损失函数输出
    :param hour_per_epoch: 一小时迭代次数
    :param epoch: 迭代次数
    :param args: 参数
    :param step: 步长
    :param step_per_epoch: 每一代的步长
    :param optimizer: 优化器
    :param loss: 总损失
    :param loss1: 位移损失
    :param loss2: 旋转损失
    :param loss_list: 总损失
    :param loss1_list: 位移损失
    :param loss2_list: 旋转损失
    :param writer: 记录损失
    :param step_global:
    :return:
    """
    print('\n{:.3f} [{:03d}/{:03d}] [{:03d}/{:03d}] lr {:.7f}: {:.4f}({:.4f})={:.4f}({:.4f})+{:d}'
          '*{:.4f}({:.4f})'.format(hour_per_epoch, epoch + 1, args.epoch_max, step + 1,
                                   step_per_epoch,
                                   optimizer.param_groups[0]['lr'], loss, np.mean(loss_list), loss1,
                                   np.mean(loss1_list), args.beta, loss2, np.mean(loss2_list)))
    writer.add_scalars('./train-val',
                       {'loss_t': loss, 'loss1_t': loss1, 'loss2_t': loss2},
                       step_global)


def display_loss_tb_val(batch_v, loss_v, loss1_v, loss2_v, args, writer, step_global):
    print('\n{:d} batches: L {:.4f}={:.4f}+{:d}*{:.4f}'.format(batch_v, loss_v, loss1_v, args.beta, loss2_v))
    writer.add_scalars('./train-val', {'loss_v': loss_v, 'loss1_v': loss1_v, 'loss2_v': loss2_v}, step_global)
