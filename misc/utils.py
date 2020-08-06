# coding:utf-8
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time

import numpy as np
import pdb

import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

import glob
import shutil
import os
import collections
import colorlog

import random

GLOBAL_SEED = 1
GLOBAL_WORKER_ID = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if key not in dict_to.keys():
            assert AssertionError('key mismatching')
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def add_picks_histgram(picks, pvideo_len, video_len, tf_writer, iteration):
    pvideo_mask = pvideo_len.new_zeros(len(pvideo_len), pvideo_len.max())
    for i, l in enumerate(pvideo_len):
        pvideo_mask[i, :l] = 1
    h_picks = picks.masked_select(pvideo_mask.byte()).cpu().numpy()
    h_interval = (picks[:, 1:] - picks[:, :-1]).masked_select(pvideo_mask[:, 1:].byte()).cpu().numpy()
    h_number = pvideo_len.cpu().numpy()

    video_len, _ = video_len.float().chunk(2, dim=1)
    picks = picks.float()

    h_picks_ratio = (picks / video_len).masked_select(pvideo_mask.byte()).cpu().numpy()
    h_interval_max_ratio = ((picks[:, 1:] - picks[:, :-1]) / video_len).masked_select(
        pvideo_mask[:, 1:].byte()).cpu().numpy()
    h_number_ratio = (pvideo_len.float() / video_len.squeeze(1)).cpu().numpy()
    names = ["picks_position", "picks_interval", "pick_num", "picks_posit_relative", "picks_interval_relative",
             "pick_num_ratio"]
    data = [h_picks, h_interval, h_number, h_picks_ratio, h_interval_max_ratio, h_number_ratio]
    for name, d in zip(*(names, data)):
        tf_writer.add_histogram(name, d, iteration)



def print_opt(opt, model, logger):
    print_alert_message('All args:', logger)
    for key, item in opt._get_kwargs():
        logger.info('{} = {}'.format(key, item))
    print_alert_message('Model structure:', logger)
    logger.info(model)


def build_floder(opt):
    if opt.start_from:
        print('Start training from id:{}'.format(opt.start_from))
        save_folder = os.path.join(opt.save_dir, opt.start_from)
        assert os.path.exists(save_folder)
    else:
        if not os.path.exists(opt.save_dir):
            os.mkdir(opt.save_dir)
        save_folder = os.path.join(opt.save_dir, opt.id)
        if os.path.exists(save_folder):
            wait_flag = input('Warning! Folder {} exists, rename it? (Y/N) : '.format(save_folder))
            if wait_flag in ['Y', 'y']:
                opt.id = opt.id + '_v_{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
                save_folder = os.path.join(opt.save_dir, opt.id)
                print('Rename opt.id to "{}".'.format(opt.id))
            else:
                assert False, 'Parameter id error, folder {} exists'.format(save_folder)
        print('Results folder "{}" does not exist, creating folder...'.format(save_folder))
        os.mkdir(save_folder)
        os.mkdir(os.path.join(save_folder, 'prediction'))
    return save_folder


def backup_envir(save_folder):
    backup_folders = ['cfgs', 'misc', 'models']
    # backup_folders = ['models']
    backup_files = glob.glob('./*.py')
    for folder in backup_folders:
        shutil.copytree(folder, os.path.join(save_folder, 'backup', folder))
    for file in backup_files:
        shutil.copyfile(file, os.path.join(save_folder, 'backup', file))


def create_logger(folder, filename):
    log_colors = {
        'DEBUG': 'blue',
        'INFO': 'white',
        'WARNING': 'green',
        'ERROR': 'red',
        'CRITICAL': 'yellow',
    }

    import logging
    logger = logging.getLogger('DVC')
    # %(filename)s$RESET:%(lineno)d
    # LOGFORMAT = "%(log_color)s%(asctime)s [%(log_color)s%(filename)s:%(lineno)d] | %(log_color)s%(message)s%(reset)s |"
    LOGFORMAT = ""
    LOG_LEVEL = logging.DEBUG
    logging.root.setLevel(LOG_LEVEL)
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(colorlog.ColoredFormatter(LOGFORMAT, datefmt='%d %H:%M', log_colors=log_colors))

    # print to log file
    hdlr = logging.FileHandler(os.path.join(folder, filename))
    hdlr.setLevel(LOG_LEVEL)
    # hdlr.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    hdlr.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(hdlr)
    logger.addHandler(stream)
    return logger


def print_alert_message(str, logger=None):
    msg = '*' * 20 + ' ' + str + ' ' + '*' * (58 - len(str))
    if logger:
        logger.info('\n\n'+msg)
    else:
        print(msg)

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for i, param in enumerate(group['params']):
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def fix_model_parameters(model):
    for para in model.parameters():
        para.requires_grad = False


def unfix_model_parameters(model):
    for para in model.parameters():
        para.requires_grad = True


if __name__ == '__main__':
    # import opts
    #
    # info = {'opt': vars(opts.parse_opts()),
    #         'loss': {'tap_loss': 0, 'tap_reg_loss': 0, 'tap_conf_loss': 0, 'lm_loss': 0}}
    # record_this_run_to_csv(info, 'save/results_all_runs.csv')

    logger = create_logger('./', 'mylogger.log')
    logger.info('test')
    logger.info('test2')
