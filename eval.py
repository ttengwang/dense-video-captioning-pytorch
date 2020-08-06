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
                                 opt.dict_file, False, 'gt',
                                 logger, opt)
    loader = DataLoader(val_dataset, batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn)

    opt.vocab_size = val_dataset.vocab_size
    model = EncoderDecoder(opt)

    if opt.eval_model_path:
        model_path = opt.eval_model_path
    else:
        model_path = os.path.join(folder_path, 'model-best-CE.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(folder_path, 'model-best-RL.pth')

    while not os.path.exists(model_path):
        assert AssertionError('File {} does not exist'.format(model_path))

    logger.debug('Loading model from {}'.format(model_path))

    loaded_pth = torch.load(model_path)
    epoch = loaded_pth['epoch']

    model.load_state_dict(loaded_pth['model'])
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    dvc_json_path = os.path.join(folder_path, '{}_epoch{}_num{}_eval{}_score{}_nms{}_top{}.json'.format(
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + str(opt.id), epoch, len(loader.dataset), opt.eval_type,
        opt.eval_score_threshold, opt.eval_nms_threshold, opt.eval_top_n))

    caption_scores = evaluate(model, loader, dvc_json_path, opt.load_tap_json,
                                    opt.eval_score_threshold, opt.eval_nms_threshold,
                                    opt.eval_top_n, logger)

    avg_eval_score = {key: np.array(value).mean() for key, value in caption_scores.items()}
    logger.info(
        'Validation result: {}\n avg_score:\n{}\n\n'.format(
            epoch, len(loader.dataset), opt.eval_top_n, opt.eval_score_threshold, opt.eval_nms_threshold,
            caption_scores, avg_eval_score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_save_dir', type=str, default='save')
    parser.add_argument('--eval_folder', type=str, default='default')
    parser.add_argument('--eval_model_path', type=str, default='')
    parser.add_argument('--eval_score_threshold', type=float, default=0.)
    parser.add_argument('--eval_nms_threshold', type=float, default=1.01)
    parser.add_argument('--eval_top_n', type=int, default=100)
    parser.add_argument('--load_tap_json', type=str, default='data/captiondata/val_1_for_tap.json')
    parser.add_argument('--eval_caption_file', type=str, default='data/captiondata/val_1.json')
    parser.add_argument('--gpu_id', type=str, nargs='+', default=['0'])
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])

    if True:
        torch.backends.cudnn.enabled = False
    main(opt)
