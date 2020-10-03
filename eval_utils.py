from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pdb
from itertools import chain
import torch
import numpy as np
import json
import os
from collections import OrderedDict
from tqdm import tqdm
import densevid_eval3.evaluate3 as eval_dvc


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
        # sentences = [data[video_name][i]['sentence'] for i in range(event_num)]
        for i, timestamp in enumerate(timestamps):
            score = data[video_name][i].get('proposal_score', 1.0)
            video_info.append({'segment': timestamp, 'score': score})
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
        # sentences = data[video_name]['sentences']
        for i, timestamp in enumerate(timestamps):
            video_info.append({'segment': timestamp, 'score': 1.})
        out['results'][video_name[2:]] = video_info
    json.dump(out, open(tap_json, 'w'))


def eval_mIOU_PR(tap_filename):
    score = collections.defaultdict(lambda: -1)
    if not os.path.exists(tap_filename + '.tmp'):
        convert_tapjson_to_dvcjson(tap_filename, tap_filename + '.tmp')
    dvc_score = eval_dvc.eval_score(tap_filename + '.tmp', onlyMeteor=0, onlyRecallPrec=1, topN=1000)
    score['avg_precison'] = dvc_score['Precision']
    score['avg_recall'] = dvc_score['Recall']
    f1 = 2./(1./ np.array(dvc_score['Precision']) + 1. / np.array(dvc_score['Recall']))
    score['f1'] = f1.tolist()
    return score


def eval_meteor(dvc_filename):
    score = collections.defaultdict(lambda: -1)
    dvc_score = eval_dvc.eval_score(dvc_filename, onlyMeteor=1, onlyRecallPrec=0, topN=1000)
    for key in dvc_score:
        score[key] = dvc_score[key]
    return score


def esgn_reranking(esgn_score, prop_score, topN=10):
    seq = []
    prob = []

    for vid in range(esgn_score.shape[0]):
        sg_seq = []
        sg_prob = []
        for i in range(esgn_score.shape[1]):
            if np.argmax(esgn_score[vid][i]) == 0:
                break
            # attention here: we can use the unbalanced weights for esgn_score and proposal_score
            s = esgn_score[vid][i] + (prop_score) * 0.8
            ids = np.argsort(-s)[:topN]
            sg_seq.extend(ids.tolist())
            sg_prob.extend(s[ids].tolist())
        seq.append(sg_seq)
        prob.append(sg_prob)
    return seq, np.array(prob)


def evaluate(model, loader, tap_json_path, score_threshold=0.1, nms_threshold=0.8, top_n=100, esgn_rerank=False,
             esgn_topN=1, logger=None):
    out_json = {'results': {},
                'version': "VERSION 1.0",
                'external_data': {'used:': True, 'details': None}}
    opt = loader.dataset.opt

    with torch.set_grad_enabled(False):
        for dt in tqdm(loader):
            valid_keys = ["video_tensor", "video_length", "video_mask", "video_key", "lnt_timestamp", "lnt_gt_idx",
                          "lnt_featstamps", "lnt_prop_score"]
            dt = {key: value for key, value in dt.items() if key in valid_keys}
            if torch.cuda.is_available():
                dt = {key: _.cuda() if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt = collections.defaultdict(lambda: None, dt)

            if esgn_rerank:
                seq, sg_prob, weights = model(dt, mode='eval_rerank')
                if len(weights):
                    seq, prob = esgn_reranking(weights.detach().cpu().numpy(),
                                            dt['lnt_prop_score'].detach().cpu().numpy(), topN=esgn_topN)
            else:
                seq, prob = model(dt, mode='eval')

            if len(seq) == 0:
                continue

            batch_json = {}
            for idx, video_name in enumerate(dt['video_key']):
                batch_json[video_name[2:]] = [
                    {
                        "segment":
                            dt['lnt_timestamp'][idx][seq[idx][pid]],
                        "esgn_score":
                            prob[idx][pid].item(),
                        "score":
                            dt['lnt_prop_score'][seq[idx][pid]].item()
                    }
                    for pid in range(len(seq[idx]))]
            batch_json = tap_nms(batch_json, score_threshold, nms_threshold, top_n)
            out_json['results'].update(batch_json)

    out_json['valid_video_num'] = len(out_json['results'])
    out_json['avg_proposal_num'] = np.array([len(v) for v in out_json['results'].values()]).mean().item()
    with open(tap_json_path, 'w') as f:
        json.dump(out_json, f)

    scores = eval_mIOU_PR(tap_json_path)
    out_json.update(scores)

    with open(tap_json_path, 'w') as f:
        json.dump(out_json, f)
    return scores


def tap_nms(tap_json, score_threshold, nms_threshold, top_n):
    for video_name in tap_json.keys():
        v_prop_timestamp = [prop['segment'] for prop in tap_json[video_name]]
        score = [prop['score'] for prop in tap_json[video_name]]
        start, end = list(zip(*v_prop_timestamp))
        remain_id = _nms(start, end, score, score_threshold=score_threshold, overlap=nms_threshold, top_n=top_n)
        v_info = [item for i, item in enumerate(tap_json[video_name]) if i in remain_id]
        tap_json[video_name] = v_info
    return tap_json


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
