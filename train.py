# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import torch
import os
import collections

import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from eval_utils import evaluate

import opts
import misc.utils as utils
from tensorboardX import SummaryWriter
from models.EncoderDecoder import EncoderDecoder
from misc.utils import print_alert_message, build_floder, create_logger, backup_envir, print_opt, set_seed
from dataset import PropSeqDataset, collate_fn


def train(opt):
    set_seed(opt.seed)
    save_folder = build_floder(opt)
    logger = create_logger(save_folder, 'train.log')
    tf_writer = SummaryWriter(os.path.join(save_folder, 'tf_summary'))

    if not opt.start_from:
        backup_envir(save_folder)
        logger.info('backup evironment completed !')

    saved_info = {'best': {}, 'last': {}, 'history': {}, 'eval_history': {}}

    # continue training
    if opt.start_from:
        opt.pretrain = False
        infos_path = os.path.join(save_folder, 'info.json')
        with open(infos_path) as f:
            logger.info('Load info from {}'.format(infos_path))
            saved_info = json.load(f)
            prev_opt = saved_info[opt.start_from_mode[:4]]['opt']

            exclude_opt = ['start_from', 'start_from_mode', 'pretrain']
            for opt_name in prev_opt.keys():
                if opt_name not in exclude_opt:
                    vars(opt).update({opt_name: prev_opt.get(opt_name)})
                if prev_opt.get(opt_name) != vars(opt).get(opt_name):
                    logger.info('Change opt {} : {} --> {}'.format(opt_name, prev_opt.get(opt_name),
                                                                   vars(opt).get(opt_name)))
        opt.feature_dim = opt.raw_feature_dim

    train_dataset = PropSeqDataset(opt.train_caption_file,
                                   opt.visual_feature_folder,
                                   True, opt.train_proposal_type,
                                   logger, opt)

    val_dataset = PropSeqDataset(opt.val_caption_file,
                                 opt.visual_feature_folder,
                                 False, 'gt',
                                 logger, opt)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.nthreads, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn)

    epoch = saved_info[opt.start_from_mode[:4]].get('epoch', 0)
    iteration = saved_info[opt.start_from_mode[:4]].get('iter', 0)
    best_val_score = saved_info[opt.start_from_mode[:4]].get('best_val_score', -1e5)
    val_result_history = saved_info['history'].get('val_result_history', {})
    loss_history = saved_info['history'].get('loss_history', {})
    lr_history = saved_info['history'].get('lr_history', {})
    opt.current_lr = vars(opt).get('current_lr', opt.lr)

    # Build model
    model = EncoderDecoder(opt)
    model.train()

    # Recover the parameters
    if opt.start_from and (not opt.pretrain):
        if opt.start_from_mode == 'best':
            model_pth = torch.load(os.path.join(save_folder, 'model-best-CE.pth'))
        elif opt.start_from_mode == 'last':
            model_pth = torch.load(os.path.join(save_folder, 'model-last.pth'))
        logger.info('Loading pth from {}, iteration:{}'.format(save_folder, iteration))
        model.load_state_dict(model_pth['model'])

    # Load the pre-trained model
    if opt.pretrain and (not opt.start_from):
        logger.info('Load pre-trained parameters from {}'.format(opt.pretrain_path))
        if torch.cuda.is_available():
            model_pth = torch.load(opt.pretrain_path)
        else:
            model_pth = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_pth['model'])

    if torch.cuda.is_available():
        model.cuda()

    if opt.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    if opt.start_from:
        optimizer.load_state_dict(model_pth['optimizer'])

    # print the args for debugging
    print_opt(opt, model, logger)
    print_alert_message('Strat training !', logger)

    loss_sum = np.zeros(3)
    bad_video_num = 0
    start = time.time()

    # Epoch-level iteration
    while True:
        if True:
            # lr decay
            if epoch > opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.lr * decay_factor
            else:
                opt.current_lr = opt.lr
            utils.set_lr(optimizer, opt.current_lr)

            # scheduled sampling rate update
            if epoch > opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.basic_ss_prob + opt.scheduled_sampling_increase_prob * frac,
                                  opt.scheduled_sampling_max_prob)
                model.decoder.ss_prob = opt.ss_prob

        # Batch-level iteration
        for dt in tqdm(train_loader):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if opt.debug:
                # each epoch contains less mini-batches for debugging
                if (iteration + 1) % 5 == 0:
                    iteration += 1
                    break
            elif epoch == 0:
                break
            iteration += 1

            if torch.cuda.is_available():
                optimizer.zero_grad()
                dt = {key: _.cuda() if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}

            dt = collections.defaultdict(lambda: None, dt)

            if True:
                train_mode = 'train'

                loss = model(dt, mode=train_mode)
                loss_sum[0] = loss_sum[0] + loss.item()

                loss.backward()
                utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            losses_log_every = int(len(train_loader) / 5)

            if iteration % losses_log_every == 0:
                end = time.time()
                losses = np.round(loss_sum / losses_log_every, 3)
                logger.info(
                    "ID {} iter {} (epoch {}, lr {}), avg_iter_loss = {}, time/iter = {:.3f}, bad_vid = {:.3f}"
                        .format(opt.id, iteration, epoch, opt.current_lr, losses,
                                (end - start) / losses_log_every, bad_video_num))

                tf_writer.add_scalar('lr', opt.current_lr, iteration)
                tf_writer.add_scalar('ss_prob', model.decoder.ss_prob, iteration)
                tf_writer.add_scalar('train_caption_loss', losses[0].item(), iteration)

                loss_history[iteration] = losses.tolist()
                lr_history[iteration] = opt.current_lr
                loss_sum = 0 * loss_sum
                start = time.time()
                bad_video_num = 0
                torch.cuda.empty_cache()

        # evaluation
        if (epoch % opt.save_checkpoint_every == 0) and (epoch >= opt.min_epoch_when_save) and (epoch != 0):
            model.eval()

            result_json_path = os.path.join(save_folder, 'prediction',
                                         'num{}_epoch{}_score{}_nms{}_top{}.json'.format(
                                             len(val_dataset), epoch, opt.eval_score_threshold,
                                             opt.eval_nms_threshold, opt.eval_top_n))
            eval_score = evaluate(model, val_loader, result_json_path,
                                  opt.eval_score_threshold, opt.eval_nms_threshold,
                                  opt.eval_top_n, False, 1, logger=logger)
            current_score = np.array(eval_score['f1']).mean()

            # add to tf summary
            for key in eval_score.keys():
                tf_writer.add_scalar(key, np.array(eval_score[key]).mean(), iteration)
            _ = [item.append(np.array(item).mean()) for item in eval_score.values() if isinstance(item, list)]
            print_info = '\n'.join([key + ":" + str(eval_score[key]) for key in eval_score.keys()])
            logger.info('\nValidation results of iter {}:\n'.format(iteration) + print_info)
            val_result_history[epoch] = {'eval_score': eval_score}

            # Save model
            saved_pth = {'epoch': epoch,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(), }

            if opt.save_all_checkpoint:
                checkpoint_path = os.path.join(save_folder, 'model_iter_{}.pth'.format(iteration))
            else:
                checkpoint_path = os.path.join(save_folder, 'model-last.pth')

            torch.save(saved_pth, checkpoint_path)
            logger.info('Save model at iter {} to {}.'.format(iteration, checkpoint_path))

            # save the model parameter and  of best epoch
            if current_score > best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                saved_info['best'] = {'opt': vars(opt),
                                      'iter': iteration,
                                      'epoch': best_epoch,
                                      'best_val_score': best_val_score,
                                      'result_json_path': result_json_path,
                                      'avg_proposal_num': eval_score['avg_proposal_number'],
                                      'Precision': eval_score['Precision'],
                                      'Recall': eval_score['Recall']
                                      }

                # suffix = "RL" if sc_flag else "CE"
                torch.save(saved_pth, os.path.join(save_folder, 'model-best.pth'))
                logger.info('Save Best-model at iter {} to checkpoint file.'.format(iteration))

            saved_info['last'] = {'opt': vars(opt),
                                  'iter': iteration,
                                  'epoch': epoch,
                                  'best_val_score': best_val_score,
                                  }
            saved_info['history'] = {'val_result_history': val_result_history,
                                     'loss_history': loss_history,
                                     'lr_history': lr_history,
                                     }
            with open(os.path.join(save_folder, 'info.json'), 'w') as f:
                json.dump(saved_info, f)
            logger.info('Save info to info.json')

            model.train()

        epoch += 1
        torch.cuda.empty_cache()
        # Stop criterion
        if epoch >= opt.epoch:
            tf_writer.close()
            break

    return saved_info


if __name__ == '__main__':
    opt = opts.parse_opts()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    train(opt)
