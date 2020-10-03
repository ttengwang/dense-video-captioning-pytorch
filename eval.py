from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import argparse
import torch
import time

from eval_utils import evaluate
from models.EncoderDecoder import EncoderDecoder
from misc.utils import create_logger
from dataset import PropSeqDataset, collate_fn
from torch.utils.data import DataLoader


def main(opt):
    folder_path = os.path.join(opt.eval_save_dir, opt.eval_folder)
    infos_path = os.path.join(folder_path, 'info.json')
    logger = create_logger(folder_path, 'val.log')
    logger.info(vars(opt))

    with open(infos_path, 'rb') as f:
        logger.info('load info from {}'.format(infos_path))
        old_opt = json.load(f)['best']['opt']

    for k, v in old_opt.items():
        if k[:4] != 'eval':
            vars(opt).update({k: v})
    opt.feature_dim = opt.raw_feature_dim

    # Create the Data Loader instance
    val_dataset = PropSeqDataset(opt.eval_caption_file,
                                 opt.visual_feature_folder,
                                 False, 'gt',
                                 logger, opt)
    loader = DataLoader(val_dataset, batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn)

    model = EncoderDecoder(opt)

    if opt.eval_model_path:
        model_path = opt.eval_model_path
    else:
        model_path = os.path.join(folder_path, 'model-best.pth')

    while not os.path.exists(model_path):
        raise AssertionError('File {} does not exist'.format(model_path)) #TODO

    logger.debug('Loading model from {}'.format(model_path))

    loaded_pth = torch.load(model_path)
    epoch = loaded_pth['epoch']

    model.load_state_dict(loaded_pth['model'])
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    out_json_path = os.path.join(folder_path, '{}_epoch{}_num{}_score{}_nms{}_top{}.json'.format(
        time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime()) + str(opt.id), epoch, len(loader.dataset),
        opt.eval_score_threshold, opt.eval_nms_threshold, opt.eval_top_n))

    caption_scores = evaluate(model, loader, out_json_path,
                              opt.eval_score_threshold, opt.eval_nms_threshold,
                              opt.eval_top_n, opt.eval_esgn_rerank, opt.eval_esgn_topN, logger)

    avg_eval_score = {key: np.array(value).mean() for key, value in caption_scores.items() if key !='tiou'}
    avg_eval_score2 = {key: np.array(value).mean() * 4917 / len(loader.dataset) for key, value in caption_scores.items() if key != 'tiou'}

    logger.info(
        '\nValidation result based on all 4917 val videos:\n {}\n avg_score:\n{}'.format(
                                                                                   caption_scores.items(),
                                                                                   avg_eval_score))

    logger.info(
            '\nValidation result based on {} available val videos:\n avg_score:\n{}'.format(len(loader.dataset),
                                                                                       avg_eval_score2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_save_dir', type=str, default='save')
    parser.add_argument('--eval_folder', type=str, default='run0')
    parser.add_argument('--eval_model_path', type=str, default='')
    parser.add_argument('--eval_score_threshold', type=float, default=0.)
    parser.add_argument('--eval_nms_threshold', type=float, default=1.01)
    parser.add_argument('--eval_top_n', type=int, default=100)
    parser.add_argument('--eval_caption_file', type=str, default='data/captiondata/val_1.json')
    parser.add_argument('--eval_proposal_file', type=str, default='data/generated_proposals/dbg_trainval_top100.json')
    parser.add_argument('--eval_esgn_rerank', action='store_true')
    parser.add_argument('--eval_esgn_topN', type=int, default=1)
    parser.add_argument('--gpu_id', type=str, nargs='+', default=['0'])
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])

    if True:
        torch.backends.cudnn.enabled = False
    main(opt)
