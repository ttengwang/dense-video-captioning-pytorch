import pdb
import sys
import torch
from torch import nn
import models
import numpy as np
from collections import OrderedDict
from itertools import chain
from models.ESGN import ESGN


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
        self.decoder = ESGN(opt)

    def get_features(self, dt, soi_select_list):
        # assert type(soi_select_list) == list
        soi_select_list = np.array(soi_select_list)
        event, pos_feats, clip, clip_mask = self.event_encoder(dt['video_tensor'],
                                                    dt['lnt_gt_idx'][:, 1], soi_select_list, dt['lnt_event_seq_idx'],
                                                    dt['lnt_timestamp'], dt['video_length'][:, 1])
        return event, pos_feats, clip, clip_mask


    def forward(self, dt, mode='train', loader=None):
        '''
        decoder RNN, batch_size (the number of videos) must be 1
        '''
        vid_num, vid_len, _ = dt['video_tensor'].shape
        assert vid_num==1
        assert self.opt.train_proposal_type in ['learnt']

        if hasattr(self, 'frame_reduce_dim_layer'):
            dt['video_tensor'] = self.frame_reduce_dim_layer(dt['video_tensor'].reshape(vid_num * vid_len, -1)).reshape(
                vid_num, vid_len, -1)

        event, pos_feats, clip, clip_mask = self.get_features(dt, dt['lnt_featstamps'])
        FIRST_DIM = 0
        event = event.unsqueeze(FIRST_DIM) # only avalilable for video number = 1
        if mode == 'train':

            gt_seq = event.new_zeros((vid_num, 2 + len(dt['gt_featstamps'])), dtype=torch.int)
            _, gt_seq_mid = dt['lnt_iou_mat'][:, :-1, 2:].max(dim=2)
            gt_seq[:, 1:-1] = gt_seq_mid + 2
            gt_seq[:, 0] = 1
            prob = self.decoder(event, pos_feats, gt_seq)
            loss = self.decoder.build_loss(dt['lnt_iou_mat'][FIRST_DIM], prob[FIRST_DIM])
            return loss

        elif mode == 'eval':
            with torch.no_grad():
                seq, prob = self.decoder.sample(event, pos_feats)
            return seq, prob

        elif mode == 'eval_rerank':
            with torch.no_grad():
                seq, sg_prob, weights = self.decoder.sample_rerank(event, pos_feats, dt['lnt_prop_score'])
            return seq, sg_prob,weights
        else:
            raise AssertionError