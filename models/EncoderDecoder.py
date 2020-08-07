import pdb
import sys
import torch
from torch import nn
import models
import numpy as np
from collections import OrderedDict
from itertools import chain

sys.path.append("densevid_eval3/coco-caption3")

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

Meteor_scorer = None


# Cider_scorer = None

class EncoderDecoder(nn.Module):
    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()
        self.opt = opt
        if opt.feature_dim > 1024:
            self.frame_reduce_dim_layer = nn.Sequential(nn.Linear(opt.feature_dim, self.opt.hidden_dim),
                                                        nn.ReLU())
            self.opt.raw_feature_dim = self.opt.feature_dim
            self.opt.feature_dim = self.opt.hidden_dim

        self.event_encoder_type = opt.event_encoder_type
        self.event_encoder = models.setup_event_encoder(opt)
        self.caption_decoder = models.setup_caption_decoder(opt)

    def forward(self, dt, mode, loader=None):

        if 'hrnn' in self.opt.caption_decoder_type:
            return self.forward_hrnn(dt, mode, loader)
        else:
            return self.forward_rnn(dt, mode, loader)

    def get_features(self, dt, soi_select_list):
        # assert type(soi_select_list) == list
        soi_select_list = np.array(soi_select_list)
        event, clip, clip_mask = self.event_encoder(dt['video_tensor'],
                                                    dt['lnt_gt_idx'][:, 1], soi_select_list, dt['lnt_event_seq_idx'],
                                                    list(chain(*dt['lnt_timestamp'])), dt['video_length'][:, 1])
        return event, clip, clip_mask

    def forward_hrnn(self, dt, mode='train', loader=None):
        '''
        Support caption model with hierarchical RNN, note that batch_size must be 1 (one video)
        '''
        assert self.opt.batch_size == 1
        assert self.opt.train_proposal_type in ['learnt_seq', 'gt']

        FIRST_DIM = 0
        event_seq_idx = dt['lnt_event_seq_idx'][FIRST_DIM]
        if mode == 'train' or mode == 'train_rl':
            seq_gt_idx = dt['lnt_seq_gt_idx'][FIRST_DIM]
            cap_raw = dt['cap_raw'][FIRST_DIM]
            cap_big_ids = dt['lnt_gt_idx'][:, 0]

        if hasattr(self, 'frame_reduce_dim_layer'):
            vid_num, vid_len, _ = dt['video_tensor'].shape
            tmp = dt['video_tensor'].reshape(vid_num * vid_len, -1)
            dt['video_tensor'] = self.frame_reduce_dim_layer(tmp).reshape(vid_num, vid_len, -1)

        event, clip, clip_mask = self.get_features(dt, dt['lnt_featstamps'])

        event_feat_expand_flag = self.event_encoder_type in ['tsrm']

        if mode == 'train':
            cap_prob = self.caption_decoder(event, clip, clip_mask, dt['cap_tensor'][cap_big_ids],
                                           event_seq_idx, event_feat_expand_flag)
            cap_prob = cap_prob.reshape(-1, cap_prob.shape[-2], cap_prob.shape[-1])
            caption_tensor = dt['cap_tensor'][:, 1:][seq_gt_idx.reshape(-1)]
            caption_mask = dt['cap_mask'][:, 1:][seq_gt_idx.reshape(-1)]
            loss = self.caption_decoder.build_loss(cap_prob, caption_tensor, caption_mask)
            return loss, torch.zeros(1), torch.zeros(1)

        elif mode == 'train_rl':
            # gen_result: (eseq_num, eseq_len, ~cap_len), sample_logprobs:(eseq_num, eseq_len, ~cap_len)
            gen_result, sample_logprobs = self.caption_decoder.sample(event, clip, clip_mask, event_seq_idx,
                                                                      event_feat_expand_flag, opt={'sample_max': 0})
            self.caption_decoder.eval()
            with torch.no_grad():
                # SCST scheme in paper "Self-critical Sequence Training for Image Captioning"
                greedy_res, _ = self.caption_decoder.sample(event, clip, clip_mask, event_seq_idx,
                                                            event_feat_expand_flag)
                # # RL scheme in paper "streamlined dense video captioning"
                # video_bl, event_bl, clip_bl, clip_mask_bl, _ = self.get_features(dt, dt['gt_featstamps'])
                # greedy_res, _ = self.caption_model.sample(video_bl, event_bl, clip_bl, clip_mask_bl, seq_gt_idx,
                #                                               event_feat_expand_flag)
            self.caption_decoder.train()
            gen_result = gen_result.reshape(-1, gen_result.shape[-1])
            greedy_res = greedy_res.reshape(-1, greedy_res.shape[-1])
            gt_caption = [loader.dataset.translate(cap, max_len=50) for cap in cap_raw]
            gt_caption = [gt_caption[i] for i in seq_gt_idx.reshape(-1)]
            reward, sample_meteor, greedy_meteor = get_caption_reward(greedy_res, gt_caption, gen_result, self.opt)
            reward = np.repeat(reward[:, np.newaxis], gen_result.size(1), 1)
            caption_loss = self.caption_decoder.build_rl_loss(sample_logprobs, gen_result.float(),
                                                              sample_logprobs.new_tensor(reward))
            return caption_loss, sample_meteor, greedy_meteor

        elif mode == 'eval':
            with torch.no_grad():
                seq, cap_prob = self.caption_decoder.sample(event, clip, clip_mask, event_seq_idx,
                                                           event_feat_expand_flag)
            return seq, cap_prob

        else:
            raise AssertionError

    def forward_rnn(self, dt, mode='train', loader=None):
        '''
        Support caption model with single-level RNN, batch_size can be larger than 1
        '''
        assert self.opt.train_proposal_type in ['learnt', 'gt']

        if hasattr(self, 'frame_reduce_dim_layer'):
            vid_num, vid_len, _ = dt['video_tensor'].shape
            dt['video_tensor'] = self.frame_reduce_dim_layer(dt['video_tensor'].reshape(vid_num * vid_len, -1)).reshape(
                vid_num, vid_len, -1)

        cap_bigids, cap_vid_ids, cap_event_ids = dt['lnt_gt_idx'][:, 0], dt['lnt_gt_idx'][:, 1], dt['lnt_gt_idx'][:, 2]
        event, clip, clip_mask = self.get_features(dt, dt['lnt_featstamps'])

        if mode == 'train':
            cap_prob = self.caption_decoder(event, clip, clip_mask, dt['cap_tensor'][cap_bigids])
            cap_prob = cap_prob.reshape(-1, cap_prob.shape[-2], cap_prob.shape[-1])
            caption_tensor = dt['cap_tensor'][:, 1:][cap_bigids]
            caption_mask = dt['cap_mask'][:, 1:][cap_bigids]
            loss = self.caption_decoder.build_loss(cap_prob, caption_tensor, caption_mask)
            return loss, torch.zeros(1), torch.zeros(1)

        elif mode == 'train_rl':
            # gen_result: (eseq_num, eseq_len, ~cap_len), sample_logprobs :(eseq_num, eseq_len, ~cap_len)
            gen_result, sample_logprobs = self.caption_decoder.sample(event, clip, clip_mask,
                                                                      opt={'sample_max': 0})
            self.caption_decoder.eval()
            with torch.no_grad():
                greedy_res, _ = self.caption_decoder.sample(event, clip, clip_mask)
                # video_bl, event_bl, clip_bl, clip_mask_bl, _ = self.get_features(dt, dt['gt_featstamps'])
                # greedy_res, _ = self.caption_model.sample(video_bl, event_bl, clip_bl, clip_mask_bl)
            self.caption_decoder.train()
            gen_result = gen_result.reshape(-1, gen_result.shape[-1])
            greedy_res = greedy_res.reshape(-1, greedy_res.shape[-1])

            if True:
                gt_caption = [[loader.dataset.translate(cap, max_len=50) for cap in caps] for caps in dt['cap_raw']]
                gt_caption = [gt_caption[cap_vid_ids[i]][cap_event_ids[i]] for i in range(len(cap_vid_ids))]
                reward, sample_meteor, greedy_meteor = get_caption_reward(greedy_res, gt_caption, gen_result, self.opt)
            reward = np.repeat(reward[:, np.newaxis], gen_result.size(1), 1)
            caption_loss = self.caption_decoder.build_rl_loss(sample_logprobs, gen_result.float(),
                                                              sample_logprobs.new_tensor(reward))
            return reward, caption_loss, sample_meteor, greedy_meteor

        elif mode == 'eval':
            with torch.no_grad():
                seq, cap_prob = self.caption_decoder.sample(event, clip, clip_mask)
            return seq, cap_prob
        else:
            raise AssertionError


def init_scorer():
    global Meteor_scorer
    Meteor_scorer = Meteor()
    # global Cider_scorer
    # Cider_scorer = Cider()


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_caption_reward(greedy_res, gt_captions, gen_result, opt):
    greedy_res = greedy_res.detach().cpu().numpy()
    gen_result = gen_result.detach().cpu().numpy()
    batch_size = len(gen_result)

    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(gt_captions)):
        gts[i] = [array_to_str(gt_captions[i][1:])]

    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}

    _, meteor_score = Meteor_scorer.compute_score(gts, res__)
    scores = np.array(meteor_score)
    rewards = scores[:batch_size] - scores[batch_size:]

    return rewards, scores[:batch_size], scores[batch_size:]
