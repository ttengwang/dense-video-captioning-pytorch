from collections import defaultdict
from itertools import chain
import json

import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

C3D_MEAN = -0.001915027447565527
C3D_VAR = 1.9239444588254049
RESNET_MEAN = 0.41634243404998694
RESNET_VAR = 0.2569392081183313
BN_MEAN = 0.8945046635916155
BN_VAR = 3.6579982046018844


def collate_fn(batch):
    batch_size = len(batch)
    feature_size = batch[0][0].shape[1]
    feature_list, timestamps_list, gt_timestamps_list, event_seq_idx, seq_gt_idx, gt_idx, caption_list, raw_timestamp, gt_raw_timestamp, raw_duration, raw_caption, key = zip( *batch)

    max_video_length = max([x.shape[0] for x in feature_list])
    max_caption_length = max(chain(*[[len(caption) for caption in captions] for captions in caption_list]))
    total_caption_num = sum(chain([len(captions) for captions in caption_list]))
    total_proposal_num = sum(chain([len(timestamp) for timestamp in timestamps_list]))

    video_tensor = torch.FloatTensor(batch_size, max_video_length, feature_size).zero_()
    video_length = torch.FloatTensor(batch_size, 2).zero_()  # true length, sequence length
    video_mask = torch.FloatTensor(batch_size, max_video_length, 1).zero_()

    caption_tensor = torch.LongTensor(total_caption_num, max_caption_length).zero_()

    caption_length = torch.LongTensor(total_caption_num).zero_()
    caption_mask = torch.FloatTensor(total_caption_num, max_caption_length, 1).zero_()
    caption_gather_idx = torch.LongTensor(total_caption_num).zero_()
    proposal_gather_idx = torch.LongTensor(total_proposal_num).zero_()

    # index information for finding corresponding gt captions
    gt_idx_tensor = torch.LongTensor(total_proposal_num,3).zero_()

    total_caption_idx = 0
    total_proposal_idx = 0

    for idx in range(batch_size):
        video_len = feature_list[idx].shape[0]

        video_tensor[idx, :video_len, :] = torch.from_numpy(feature_list[idx])
        video_length[idx, 0] = float(video_len)
        video_length[idx, 1] = raw_duration[idx]
        video_mask[idx, :video_len, 0] = 1

        proposal_length = len(timestamps_list[idx])
        timestamps = list(chain(*timestamps_list))
        proposal_gather_idx[total_proposal_idx:total_proposal_idx + proposal_length] = idx
        gt_idx_tensor[total_proposal_idx: total_proposal_idx + proposal_length, 0] = torch.from_numpy(
            total_caption_idx + gt_idx[idx])
        gt_idx_tensor[total_proposal_idx: total_proposal_idx + proposal_length, 1] = idx
        gt_idx_tensor[total_proposal_idx: total_proposal_idx + proposal_length, 2] = torch.from_numpy(gt_idx[idx])

        gt_proposal_length = len(gt_timestamps_list[idx])
        gt_timestamps = list(chain(*gt_timestamps_list))

        caption_gather_idx[total_caption_idx:total_caption_idx + gt_proposal_length] = idx

        for iidx, captioning in enumerate(caption_list[idx]):
            _caption_len = len(captioning)
            caption_length[total_caption_idx + iidx] = _caption_len
            caption_tensor[total_caption_idx + iidx, :_caption_len] = torch.from_numpy(captioning)
            caption_mask[total_caption_idx + iidx, :_caption_len, 0] = 1
        total_caption_idx += gt_proposal_length
        total_proposal_idx += proposal_length

    dt = {
        "video":
            {
                "tensor": video_tensor,  # tensor,      (video_num, video_len, video_dim)
                "length": video_length,
                # tensor,      (video_num, 2), the first row is feature length, the second is time length
                "mask": video_mask,  # tensor,      (video_num, video_len,)
                "key": list(key),  # list,        (video_num)
            },

        "lnt":
            {
                "featstamps": timestamps,  # list,        (lnt_all_event_num, 2)
                "timestamp": list(raw_timestamp),  # list (len: video_num) of tensors (shape: (~lnt_event_num, 2))
                "gather_idx": proposal_gather_idx,  # tensor, (lnt_all_event_num)
                "gt_idx": gt_idx_tensor,  # tensor,      (lnt_all_event_num, 3)

                # only available when video_num = 1
                "event_seq_idx": event_seq_idx,
                # list (len: video_num) of tensors (shape: (eseq_num, eseq_len)), eseq_len = 1 means we do not use event sequence
                "seq_gt_idx": seq_gt_idx,
                # list (len: video_num) of tensors(shape: (eseq_num, eseq_len)), eseq_len = 1 means we do not use event sequence
            },

        "gt":
            {
                "featstamps": gt_timestamps,  # list,        (gt_all_event_num, 2)
                "timestamp": list(gt_raw_timestamp),  # list (len: video_num) of tensors (shape: (gt_event_num, 2))
                "gather_idx": caption_gather_idx,  # tensor,      (gt_all_event_num)
            },

        "cap":
            {
                "tensor": caption_tensor,  # tensor,      (gt_all_event_num, cap_len)
                "length": caption_length,  # tensor,      (gt_all_event_num)
                "mask": caption_mask,  # tensor,      (gt_all_event_num, cap_len, 1)
                "raw": list(raw_caption),  # list,        (video_num, ~gt_event_num, ~~caption_len)
            }
    }
    dt = {k1 + '_' + k2: v2 for k1, v1 in dt.items() for k2, v2 in v1.items()}
    return dt


class EDVCdataset(Dataset):

    def __init__(self, anno_file, feature_folder, translator_json, is_training, proposal_type, logger,
                 opt):

        super(EDVCdataset, self).__init__()
        self.anno = json.load(open(anno_file, 'r'))

        self.translator = json.load(open(translator_json, 'r'))
        self.vocab_size = len(self.translator['word_to_ix'].keys())

        self.translator['word_to_ix'] = defaultdict(lambda: self.vocab_size,
                                                    self.translator['word_to_ix'])
        self.translator['ix_to_word'] = defaultdict(lambda: self.vocab_size,
                                                    self.translator['ix_to_word'])
        self.max_caption_len = opt.max_caption_len
        logger.info('load translator, total_vocab: %d', len(self.translator['ix_to_word']))

        self.keys = self.anno.keys()
        if opt.invalid_video_json:
            invalid_videos = json.load(open(opt.invalid_video_json))
            self.keys = [k for k in self.keys if k[:13] not in invalid_videos]
        logger.info('load captioning file, %d captioning loaded', len(self.keys))

        self.feature_folder = feature_folder
        self.feature_sample_rate = opt.feature_sample_rate
        self.opt = opt

        self.proposal_type = proposal_type
        self.is_training = is_training
        self.train_proposal_sample_num = opt.train_proposal_sample_num

        self.feature_dim = self.opt.feature_dim

        if self.is_training and opt.train_proposal_file:
            self.train_proposal_file = json.load(open(opt.train_proposal_file))['results']
            for vid in self.train_proposal_file.keys():
                v_data = self.train_proposal_file[vid]
                v_data = [p for p in v_data if p['score'] > 0]
                tmp = sorted(v_data, key=lambda x: x['segment'])
                self.train_proposal_file[vid] = tmp
            tp_keys = set(self.train_proposal_file.keys())
            self.keys = [k for k in self.keys if k[2:13] in tp_keys]

    def __len__(self):
        return len(self.keys)


    def translate(self, sentence, max_len):
        tokens = [',', ':', '!', '_', ';', '-', '.', '?', '/', '"', '\\n', '\\', '.']
        for token in tokens:
            sentence = sentence.replace(token, ' ')
        sentence_split = sentence.replace('.', ' . ').replace(',', ' , ').lower().split()
        res = np.array(
            [0] + [self.translator['word_to_ix'][word] for word in sentence_split][:max_len - 2] + [0])
        return res

    def rtranslate(self, sent_ids):
        for i in range(len(sent_ids)):
            if sent_ids[i] == 0:
                sent_ids = sent_ids[:i]
                break
        if len(sent_ids):
            return ' '.join([self.translator['ix_to_word'][str(idx)] for idx in sent_ids]) + '.'
        else:
            return ''

    def process_time_step(self, duration, timestamps_list, feature_length):
        duration = np.array(duration)
        timestamps = np.array(timestamps_list)
        feature_length = np.array(feature_length)
        featstamps = feature_length * timestamps / duration
        featstamps = np.minimum(featstamps, feature_length - 1).astype('int')
        featstamps = np.maximum(featstamps, 0).astype('int')
        return featstamps.tolist()

    def __getitem__(self, idx):
        raise NotImplementedError()


class PropSeqDataset(EDVCdataset):

    def __init__(self, anno_file, feature_folder, translator_pickle, is_training, proposal_type,
                 logger,
                 opt):
        super(PropSeqDataset, self).__init__(anno_file,
                                             feature_folder, translator_pickle, is_training, proposal_type,
                                             logger, opt)

    def sample_proposal_seq(self, iou_mat, sample_num, iou_thres=0):

        gt_num, lnt_num = iou_mat.shape

        lnt_max_ids = np.argmax(iou_mat, 0)
        gt_max_ids = np.argmax(iou_mat, 1)

        for i in range(gt_num):
            if iou_mat[i, gt_max_ids[i]] > 0:
                lnt_max_ids[gt_max_ids[i]] = i  # assure each gt proposal matches at last one learnt proposal
        gt2lnt = {}

        # print(lnt_max_ids)
        for j in range(lnt_num):
            if np.max(iou_mat[:, j]) > iou_thres:
                gt2lnt[lnt_max_ids[j]] = gt2lnt.get(lnt_max_ids[j], [])
                gt2lnt[lnt_max_ids[j]].append(j)

        # print(gt2lnt)
        valid_gt_num = len(gt2lnt)
        event_seq_idx = np.zeros((sample_num, valid_gt_num)).astype('int')
        seq_gt_idx = np.zeros((sample_num, valid_gt_num)).astype('int')

        for i in range(sample_num):
            j = 0
            col = 0
            while (j < gt_num):
                if j not in gt2lnt.keys():
                    j += 1
                    continue
                id = random.choice(gt2lnt[j])
                event_seq_idx[i][col] = id
                seq_gt_idx[i][col] = j
                col += 1
                j += 1

        return event_seq_idx, seq_gt_idx, lnt_max_ids

    def sample_proposal(self, iou_mat, sample_num, sample_len, iou_thres=0):
        gt_num, lnt_num = iou_mat.shape
        lnt_max_ids = np.argmax(iou_mat, 0)
        gt_max_ids = np.argmax(iou_mat, 1)

        event_seq_idx = [random.sample(range(lnt_num), sample_len) for j in range(sample_num)]
        event_seq_idx = np.sort(event_seq_idx, axis=1)
        for i in range(gt_num):
            if iou_mat[i, gt_max_ids[i]] > 0:
                lnt_max_ids[gt_max_ids[i]] = i  # assure that each GT proposal matches at last 1 lnt proposal
        seq_gt_idx = lnt_max_ids[event_seq_idx]

        return event_seq_idx.astype('int'), seq_gt_idx.astype('int'), lnt_max_ids

    def load_feats(self, key):
        if self.opt.visual_feature_type == 'c3d':
            feats = np.load(os.path.join(self.feature_folder, key[0:13] + '.npy'))
            feats = (feats - C3D_MEAN) / np.sqrt(C3D_VAR)

        elif self.opt.visual_feature_type == 'resnet':
            feats = np.load(os.path.join(self.feature_folder, key[2:13] + '_resnet.npy'))
            feats = (feats - RESNET_MEAN) / np.sqrt(RESNET_VAR)

        elif self.opt.visual_feature_type == 'resnet_bn':
            feature_obj1 = np.load(os.path.join(self.feature_folder, key[2:13] + '_resnet.npy'))
            feature_obj1 = (feature_obj1 - RESNET_MEAN) / np.sqrt(RESNET_VAR)
            feature_obj2 = np.load(os.path.join(self.feature_folder, key[2:13] + '_bn.npy'))
            feature_obj2 = (feature_obj2 - BN_MEAN) / np.sqrt(BN_VAR)
            feats = np.concatenate((feature_obj1, feature_obj2), 1)
        else:
            raise AssertionError('feature type error')
        return feats

    def __getitem__(self, idx):

        key = str(self.keys[idx])
        feats = self.load_feats(key)
        feats = feats[::self.feature_sample_rate, :]
        duration = self.anno[key]['duration']
        captions = self.anno[key]['sentences']
        gt_timestamps = self.anno[key]['timestamps']  # [gt_num, 2]

        caption_label = [np.array(self.translate(sent, self.max_caption_len)) for sent in captions]
        gt_featstamps = self.process_time_step(duration, gt_timestamps, feats.shape[0])

        if self.proposal_type == 'learnt_seq':
            lnt_timestamps = [p['segment'] for p in self.train_proposal_file[key[2:13]]]  # [p_num ,2]
            lnt_featstamps = self.process_time_step(duration, lnt_timestamps, feats.shape[0])
            iou_mat = iou(gt_timestamps, lnt_timestamps)
            sample_num = min(int(500 / len(gt_timestamps)), self.train_proposal_sample_num)  # for GPU memory limitation
            event_seq_idx, seq_gt_idx, gt_idx = self.sample_proposal_seq(iou_mat, sample_num, iou_thres=0)

        elif self.proposal_type == 'learnt':
            lnt_timestamps = [p['segment'] for p in self.train_proposal_file[key[2:13]]]  # [p_num ,2]
            train_sample_num = len(lnt_timestamps) if (
                    len(lnt_timestamps) < self.train_proposal_sample_num) else self.train_proposal_sample_num
            random_ids = np.random.choice(list(range(len(lnt_timestamps))), train_sample_num, replace=False)
            lnt_timestamps = [lnt_timestamps[_] for _ in range(len(lnt_timestamps)) if _ in random_ids]

            lnt_featstamps = self.process_time_step(duration, lnt_timestamps, feats.shape[0])
            iou_mat = iou(gt_timestamps, lnt_timestamps)
            event_seq_idx, seq_gt_idx, gt_idx = self.sample_proposal(iou_mat, 1, train_sample_num)

        elif self.proposal_type == 'gt':
            lnt_timestamps = gt_timestamps
            lnt_featstamps = gt_featstamps
            gt_idx = np.arange(len(gt_timestamps))
            event_seq_idx = seq_gt_idx = np.expand_dims(gt_idx, 0)

        else:
            raise AssertionError('proposal type error')

        return feats, lnt_featstamps, gt_featstamps, event_seq_idx, seq_gt_idx, gt_idx, caption_label, \
               lnt_timestamps, gt_timestamps, duration, captions, key


def iou(interval_1, interval_2):
    interval_1, interval_2 = map(np.array, (interval_1, interval_2))
    start, end = np.expand_dims(interval_2[:, 0], 0), np.expand_dims(interval_2[:, 1], 0)
    start_i, end_i = np.expand_dims(interval_1[:, 0], 1), np.expand_dims(interval_1[:, 1], 1)

    intersection = np.maximum(0, np.minimum(end, end_i) - np.maximum(start, start_i))
    union = np.minimum(np.maximum(end, end_i) - np.minimum(start, start_i), end - start + end_i - start_i)
    iou = intersection / (union + 1e-8)
    return iou


if __name__=="__main__":
    import opts
    from tqdm import tqdm
    from misc.utils import build_floder, create_logger

    opt = opts.parse_opts()
    save_folder = build_floder(opt)
    logger = create_logger(save_folder, 'train.log')
    train_dataset = PropSeqDataset(opt.train_caption_file,
                                   opt.visual_feature_folder,
                                   opt.dict_file, True, opt.train_proposal_type,
                                   logger, opt)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.nthreads, collate_fn=collate_fn)
    for dt in tqdm(train_loader):
        pass
    print('end')
