import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from itertools import chain


class BasicEncoder(nn.Module):
    def __init__(self, opt):
        super(BasicEncoder, self).__init__()
        self.opt = opt
        self.hidden_dim = self.opt.hidden_dim
        opt.event_context_dim = self.opt.feature_dim
        opt.clip_context_dim = self.opt.feature_dim
        self.position_encoding_size = self.opt.position_encoding_size

    def forward(self, feats, vid_idx, featstamps, event_seq_idx=None, timestamps=None, vid_time_len=None):
        clip_feats, clip_mask = self.get_clip_level_feats(feats, vid_idx, featstamps)
        event_feats = (clip_feats * clip_mask.unsqueeze(2)).sum(1) / (clip_mask.sum(1, keepdims=True) + 1e-5)
        pos_feats = self.get_pos_embed(timestamps, vid_time_len, self.position_encoding_size)
        pos_feats = feats.new_tensor(pos_feats)
        return event_feats, pos_feats, clip_feats, clip_mask

    def get_clip_level_feats(self, feats, vid_idx, featstamps):
        max_att_len = max([(s[1] - s[0] + 1) for s in featstamps])
        clip_mask = feats.new(len(featstamps), max_att_len).zero_()
        clip_feats = feats.new(len(featstamps), max_att_len, feats.shape[-1]).zero_()
        for i, soi in enumerate(featstamps):
            v_idx = vid_idx[i]
            selected = feats[v_idx][soi[0]:soi[1] + 1].reshape(-1, feats.shape[-1])
            clip_feats[i, :len(selected), :] = selected
            clip_mask[i, :len(selected)] = 1
        return clip_feats, clip_mask

    def get_pos_embed(self, timestamps, durations, L=100):
        locs = []
        for vid, duration in enumerate(durations):
            p_num = len(timestamps[vid])
            loc = torch.zeros((p_num, L))
            for i, (ts, te) in enumerate(timestamps[vid]):
                rescale_ts = max(0, int(ts / duration * (L - 1)))
                rescale_te = min(L - 1, int(te / duration * (L - 1)))
                loc[i, rescale_ts: rescale_te + 1] = 1
            locs.append(loc)
        return torch.stack(locs, 0)


class RNNEncoder(BasicEncoder):
    def __init__(self, opt):
        super(RNNEncoder, self).__init__(opt)
        self.opt = opt
        self.hidden_dim = self.opt.hidden_dim
        self.frame_encoder = nn.GRU(self.opt.feature_dim, self.hidden_dim, num_layers=2, batch_first=True,
                                    dropout=0.5, bidirectional=False, )
        self.frame_encoder_drop = nn.Dropout(p=0.5)
        opt.event_context_dim = 2 * self.hidden_dim
        opt.clip_context_dim = None

    def forward(self, feats, vid_idx, featstamps, event_seq_idx=None, timestamps=None, vid_time_len=None):
        recur_feat, _ = self.frame_encoder(feats)
        recur_feat = self.frame_encoder_drop(recur_feat)  # [video_num, video_len, video_dim]

        event_feat_list = []
        for i, soi in enumerate(featstamps):
            v_idx = vid_idx[i]
            start_feat = recur_feat[v_idx, soi[0]]
            end_feat = recur_feat[v_idx, soi[1]]
            event_feat_list.append(torch.cat((start_feat, end_feat), 0))
        event_feat = torch.stack(event_feat_list, 0)
        pos_feats = self.get_pos_embed(timestamps, vid_time_len, self.position_encoding_size)
        pos_feats = feats.new_tensor(pos_feats)
        return event_feat, pos_feats, _, _


class BRNNEncoder(BasicEncoder):
    def __init__(self, opt):
        super(BRNNEncoder, self).__init__(opt)
        self.opt = opt
        self.hidden_dim = self.opt.hidden_dim
        self.frame_encoder = nn.LSTM(self.opt.feature_dim, self.hidden_dim, num_layers=1, batch_first=True,
                                     dropout=0.5, bidirectional=True)
        self.frame_encoder_drop = nn.Dropout(p=0.5)

        opt.event_context_dim = 2 * self.hidden_dim
        opt.clip_context_dim = None

    def forward(self, feats, vid_idx, featstamps, event_seq_idx=None, timestamps=None, vid_time_len=None):
        video_num, video_len, video_dim = feats.shape
        recur_feat, _ = self.frame_encoder(feats)
        recur_feat = self.frame_encoder_drop(recur_feat)  # [video_num, video_len, video_dim]
        fpass_feat, bpass_feat = recur_feat.reshape(video_num, video_len, 2, self.hidden_dim).split(dim=2, split_size=1)

        fpass_feat = fpass_feat.squeeze(2)
        bpass_feat = bpass_feat.squeeze(2)

        event_feat_list = []
        for i, soi in enumerate(featstamps):
            v_idx = vid_idx[i]
            start_feat = fpass_feat[v_idx, soi[1]]
            end_feat = bpass_feat[v_idx, soi[0]]
            event_feat_list.append(torch.cat((start_feat, end_feat), 0))
        event_feat = torch.stack(event_feat_list, 0)
        pos_feats = self.get_pos_embed(timestamps, vid_time_len, self.position_encoding_size)
        pos_feats = feats.new_tensor(pos_feats)
        return event_feat, pos_feats, _, _


if __name__ == '__main__':
    pass
