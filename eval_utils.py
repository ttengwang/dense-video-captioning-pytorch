from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os
import pdb
from itertools import chain
import torch
import numpy as np
import json
import sys
import misc.utils as utils
from collections import OrderedDict
from tqdm import tqdm
from densevid_eval3 import evaluate3 as eval_dvc

def calculate_avg_proposal_num(json_path):
    data = json.load(open(json_path))
    return np.array([len(v) for v in data['results'].values()]).mean()

def convert_tapjson_to_dvcjson(tap_json, dvc_json):
    data = json.load(open(tap_json, 'r'))
    data['version'] = "VERSION 1.0"
    data['external_data'] = {'used:': True, 'details': "C3D pretrained on Sports-1M"}

    all_names = list(data['results'].keys())
    for video_name in all_names:
        for p_info in data['results'][video_name]:
            p_info['timestamp'] = p_info.pop('segment')
            p_info['proposal_score'] = p_info.pop('score')
        data['results']["v_" + video_name] = data['results'].pop(video_name)
    json.dump(data, open(dvc_json, 'w'))


def convert_dvcjson_to_tapjson(dvc_json, tap_json):
    data = json.load(open(dvc_json, 'r'))['results']
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}

    all_names = list(data.keys())
    for video_name in all_names:
        video_info = []
        event_num = len(data[video_name])
        timestamps = [data[video_name][i]['timestamp'] for i in range(event_num)]
        sentences = [data[video_name][i]['sentence'] for i in range(event_num)]
        for i, timestamp in enumerate(timestamps):
            video_info.append({'segment': timestamp, 'score': 1.})
        out['results'][video_name[2:]] = video_info
    json.dump(out, open(tap_json, 'w'))

def convert_gtjson_to_tapjson(gt_json, tap_json):
    data = json.load(open(gt_json, 'r'))
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}

    all_names = list(data.keys())
    for video_name in all_names:
        video_info = []
        timestamps = data[video_name]['timestamps']
        sentences = data[video_name]['sentences']
        for i, timestamp in enumerate(timestamps):
            video_info.append({'segment': timestamp, 'score': 1.})
        out['results'][video_name[2:]] = video_info
    json.dump(out, open(tap_json, 'w'))

def eval_meteor(dvc_filename):
    score = collections.defaultdict(lambda: -1)
    dvc_score = eval_dvc.eval_score(dvc_filename, onlyMeteor=1, onlyRecallPrec=0, topN=1000)
    for key in dvc_score:
        score[key] = dvc_score[key]
    return score

def evaluate(model, loader, dvc_json_path, tap_json_path, score_threshold=0.1, nms_threshold=0.8, top_n=100, logger=None):
    out_json = {'results': {},
                'version': "VERSION 1.0",
                'external_data': {'used:': True, 'details': "C3D pretrained on Sports-1M"}}
    opt = loader.dataset.opt

    if tap_json_path:
        with open(tap_json_path, 'r') as f:
            tap_json = json.load(f)['results']
            tap_keys = ['v_'+key for key in tap_json.keys()]
            loader.dataset.keys = list(set(loader.dataset.keys) & set(tap_keys))

    with torch.set_grad_enabled(False):
        for dt in tqdm(loader):
            valid_keys = ["video_tensor", "video_length", "video_mask", "video_key"]
            dt = {key: value for key, value in dt.items() if key in valid_keys}
            if torch.cuda.is_available():
                dt = {key: _.cuda() if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt = collections.defaultdict(lambda: None, dt)

            if tap_json_path:
                batch_json = OrderedDict([(video_name, tap_json[video_name[2:]]) for video_name in dt['video_key']])
                # ranking events
                for vid in batch_json.keys():
                    v_data = batch_json[vid]
                    tmp = sorted(v_data, key=lambda x: x['segment'])
                    batch_json[vid] = tmp
            else:
                raise ValueError('load_tap_json must have a value')

            raw_timestamps = [[p['segment'] for p in info] for video_name, info in batch_json.items()]
            caption_nums = [len(info) for video_name, info in batch_json.items()]
            gather_idx = np.array(list(
                chain(*[[(0, dt['video_key'].index(video_name), 0) for p in info] for i, (video_name, info) in
                        enumerate(batch_json.items())])))
            feat_len, raw_len = np.split(dt['video_length'].cpu().numpy()[gather_idx[:, 1]], 2, 1)
            dt['lnt_featstamps'] = loader.dataset.process_time_step(raw_len, list(chain(*raw_timestamps)), feat_len)
            dt['lnt_timestamp'] = raw_timestamps

            if 'hrnn' in opt.caption_decoder_type:
                assert opt.batch_size == 1
                dt['lnt_event_seq_idx'] = [np.arange(caption_nums[i])[np.newaxis, :] for i in range(len(caption_nums))]
                dt['lnt_gt_idx'] = gather_idx
                FIRST_DIM = 0
                seq, cap_prob = model.forward_hrnn(dt, mode='eval')
                seq = seq[FIRST_DIM]
                cap_prob = cap_prob[FIRST_DIM]
            else:
                dt['lnt_gt_idx'] = gather_idx
                dt['lnt_event_seq_idx'] = [np.arange(caption_nums[i])[np.newaxis, :] for i in range(len(caption_nums))]
                seq, cap_prob = model.forward_rnn(dt, mode='eval')

            if len(seq):
                mask = (seq > 0).float()
                cap_score = (mask * cap_prob).sum(1).cpu().numpy().astype('float')
                seq = seq.detach().cpu().numpy().astype('int')  # (eseq_batch_size, eseq_len, cap_len)
                pred_caption = [loader.dataset.rtranslate(s) for s in seq]
            else:
                cap_score = [-1e5] * len(gather_idx)
                pred_caption = [''] * len(gather_idx)

            # construct tap+caption json
            idx = 0
            for video_name, info in batch_json.items():
                for i, p in enumerate(info):
                    p['timestamp'] = p.pop('segment')
                    p['proposal_score'] = p.pop('score')
                    p['proposal_id'] = [i, len(info)]
                    p['sentence'] = pred_caption[idx]
                    p['sentence_score'] = cap_score[idx]
                    idx += 1
            batch_json = nms(batch_json, score_threshold, nms_threshold, top_n)
            out_json['results'].update(batch_json)

    out_json['valid_video_num'] = len(out_json['results'])
    out_json['avg_proposal_num'] = np.array([len(v) for v in out_json['results'].values()]).mean().item()
    out_json['tap_json'] = tap_json_path

    with open(dvc_json_path, 'w') as f:
        json.dump(out_json, f)

    caption_scores = eval_meteor(dvc_json_path)
    out_json.update(caption_scores)
    with open(dvc_json_path, 'w') as f:
        logger.info('\nsaving json file to {}'.format(dvc_json_path))
        json.dump(out_json, f)

    sample_vid = video_name
    logger.debug('\nSamples of generated results : vid: {}, info: {}'.format(sample_vid, out_json['results'][sample_vid][:10]))

    return caption_scores


def tap_nms(tap_json, score_threshold, nms_threshold, top_n):
    for video_name in tap_json.keys():
        v_prop_timestamp = [prop['segment'] for prop in tap_json[video_name]]
        score = [prop['score'] for prop in tap_json[video_name]]
        start, end = list(zip(*v_prop_timestamp))
        remain_id = _nms(start, end, score, score_threshold=score_threshold, overlap=nms_threshold, top_n=top_n)
        v_info = [item for i, item in enumerate(tap_json[video_name]) if i in remain_id]
        tap_json[video_name] = v_info
    return tap_json


def nms(caption_json, score_threshold, nms_threshold, top_n):
    for video_name in caption_json.keys():
        v_prop_timestamp = [prop['timestamp'] for prop in caption_json[video_name]]
        score = [prop['proposal_score'] for prop in caption_json[video_name]]
        start, end = list(zip(*v_prop_timestamp))
        remain_id = _nms(start, end, score, score_threshold=score_threshold, overlap=nms_threshold, top_n=top_n)
        v_info = [item for i, item in enumerate(caption_json[video_name]) if i in remain_id]
        caption_json[video_name] = v_info
    return caption_json


def _nms(start, end, scores, score_threshold, overlap=0.8, top_n=100):
    if len(start) == 0:
        return []
    start, end = map(np.array, (start, end))
    ind = np.argsort(scores)
    ind = np.array([k for k in ind if scores[k] > score_threshold])

    area = end - start
    pick = []
    while len(ind) > 0 and len(pick) < top_n:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        if overlap >= 1.:
            continue
        else:
            tt1 = np.maximum(start[i], start[ind])
            tt2 = np.minimum(end[i], end[ind])
            wh = np.maximum(0., tt2 - tt1)
            o = wh / (area[i] + area[ind] - wh + 1e-5)
            ind = ind[np.nonzero(o <= overlap)[0]]
    return pick


if __name__ == '__main__':
    pass