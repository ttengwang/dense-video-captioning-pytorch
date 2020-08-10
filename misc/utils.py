# coding:utf-8
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import torch
import glob
import shutil
import os
import colorlog
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if key not in dict_to.keys():
            raise AssertionError('key mismatching: {}'.format(key))
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]

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
            wait_flag = input('Warning! ID {} already exists, rename it? (Y/N) : '.format(opt.id))
            if wait_flag in ['Y', 'y']:
                opt.id = opt.id + '_v_{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
                save_folder = os.path.join(opt.save_dir, opt.id)
                print('Rename opt.id as "{}".'.format(opt.id))
            else:
                raise AssertionError('ID already exists, folder {} exists'.format(save_folder))
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
        logger.info('\n\n' + msg)
    else:
        print(msg)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for i, param in enumerate(group['params']):
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

if __name__ == '__main__':
    # import opts
    #
    # info = {'opt': vars(opts.parse_opts()),
    #         'loss': {'tap_loss': 0, 'tap_reg_loss': 0, 'tap_conf_loss': 0, 'lm_loss': 0}}
    # record_this_run_to_csv(info, 'save/results_all_runs.csv')

    logger = create_logger('./', 'mylogger.log')
    logger.info('debug')
    logger.info('test2')
