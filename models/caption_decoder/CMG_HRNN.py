import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

class CMG_HRNN(nn.Module):
    def __init__(self, opt):
        super(CMG_HRNN, self).__init__()

        self.opt = opt
        assert opt.batch_size == 1

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob
        self.max_caption_len = opt.max_caption_len
        self.ss_prob = 0.0
        self.sent_rnn_size = self.rnn_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        self.sent_rnn = nn.LSTM(opt.hidden_dim + opt.hidden_dim,
                                self.sent_rnn_size, 1, bias=False,
                                dropout=self.drop_prob_lm)

        self.gate_layer = nn.Sequential(nn.Linear(2 * opt.hidden_dim + self.rnn_size, opt.hidden_dim),
                                        nn.Sigmoid())
        self.global_proj = nn.Sequential(nn.Linear(self.sent_rnn_size, opt.hidden_dim),
                                         nn.Tanh())
        self.local_proj = nn.Sequential(nn.Linear(opt.event_context_dim, opt.hidden_dim),
                                        nn.Tanh())
        self.para_transfer_layer = nn.Linear(self.sent_rnn_size, self.rnn_size * self.num_layers)

        self.gate_drop = nn.Dropout(p=opt.drop_prob)

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.rnn_size).zero_(),
                weight.new(self.num_layers, batch_size, self.rnn_size).zero_())  # (h0, c0)

    def build_loss(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = input.reshape(-1, input.size(2))
        target = target.reshape(-1, 1)
        mask = mask.reshape(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / (torch.sum(mask) + 1e-6)
        return output

    def build_rl_loss(self, input, seq, reward):
        input = (input).reshape(-1)
        reward = (reward).reshape(-1)
        mask = (seq > 0).float()
        mask = (torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / (torch.sum(mask) + 1e-6)
        return output

    def forward(self, event, clip, clip_mask, seq, event_seq_idx, event_feat_expand=False):
        # TODO: annotation
        eseq_num, eseq_len = event_seq_idx.shape
        para_state = self.init_hidden(eseq_num)  # return Zero hidden state
        last_sent_state = clip.new_zeros(eseq_num, self.rnn_size)  # return Zero hidden state
        para_outputs = []
        seq = seq.long()

        if event is None:
            event = (clip * clip_mask.unsqueeze(2)).sum(1) / (clip_mask.sum(1, keepdims=True) + 1e-5)

        if not event_feat_expand:
            assert len(event.shape) == 2
        else:
            event = event.reshape(eseq_num, eseq_len, event.shape[-1])

        for idx in range(eseq_len):

            event_idx = event[event_seq_idx[:, idx]] if not event_feat_expand else event[:, idx]
            clip_idx = clip[event_seq_idx[:, idx]]
            clip_mask_idx = clip_mask[event_seq_idx[:, idx]]
            seq_idx = seq[event_seq_idx[:, idx]]

            # cross-modal fusion
            # last_sent_state: linguistic information of previous events
            # event_idx: visual information of previous events
            prev_state_proj = self.global_proj(last_sent_state)
            event_proj = self.local_proj(event_idx)
            gate_input = torch.cat((para_state[0][-1], prev_state_proj, event_proj), 1)
            gate = self.gate_layer(self.gate_drop(gate_input))
            gate = torch.cat((gate, 1 - gate), dim=1)
            sent_rnn_input = torch.cat((prev_state_proj, event_proj), dim=1)
            sent_rnn_input = sent_rnn_input * gate
            _, para_state = self.sent_rnn(sent_rnn_input.unsqueeze(0), para_state)

            para_c, para_h = para_state
            num_layers, batch_size, para_dim = para_h.size()
            init_h = self.para_transfer_layer(para_h[-1]).reshape(self.num_layers, batch_size, para_dim)
            state = (init_h, init_h)

            outputs = []
            seq_len_idx = (seq_idx > 0).sum(1) + 2

            last_sent_state = clip.new_zeros(eseq_num, self.rnn_size)

            for i in range(seq_idx.size(1) - 1):
                if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                    sample_prob = clip.new(eseq_num).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq_idx[:, i].clone().long()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq_idx[:, i].clone().long()
                        prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind,
                                       torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq_idx[:, i].clone().long()
                    # break if all the sequences end
                if i >= 1 and seq_idx[:, i].data.sum() == 0:
                    break

                output, state = self.get_logprobs_state(it, event_idx, clip_idx, clip_mask_idx, state)

                interest = (seq_len_idx == i + 2)

                if interest.sum() > 0:
                    end_id = interest.nonzero().squeeze(1)
                    last_sent_state[end_id] = state[0][
                        -1, end_id]  # state[0].shape = (num_layer, batch_size, rnn_size))
                outputs.append(output)  # output: (batch, vocab_size+1); outputs: ( cap_seq_len, batch, vocab_size+1 )
            para_outputs.append(
                torch.stack(outputs, 0))  # para_outputs: ( esqn_len, cap_seq_len, batch, self.vocab_size + 1 )

        para_output_tensor = clip.new_zeros(self.opt.batch_size, eseq_len, seq.size(1) - 1, eseq_num,
                                            self.vocab_size + 1)
        para_output_tensor = para_output_tensor.squeeze(0)

        for i in range(para_output_tensor.shape[0]):
            para_output_tensor[i, :len(para_outputs[i])] = para_outputs[i]
        para_output_tensor = para_output_tensor.permute(2, 0, 1,
                                                        3)  # (batchsize, esqn_len, cap_seq_len, self.vocab_size+1)

        return para_output_tensor

    def get_logprobs_state(self, it, event, clip, clip_mask, state):
        xt = self.embed(it)
        output, state = self.core(xt, event, clip, clip_mask, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)), dim=1)
        return logprobs, state

    def sample(self, event, clip, clip_mask, event_seq_idx, event_feat_expand=False, opt={}):

        sample_max = opt.get('sample_max', 1)
        temperature = opt.get('temperature', 1.0)

        eseq_num, eseq_len = event_seq_idx.shape

        para_state = self.init_hidden(eseq_num)
        last_sent_state = clip.new_zeros(eseq_num, self.rnn_size)

        para_seqLogprobs = []
        para_seq = []

        if event is None:
            event = (clip * clip_mask.unsqueeze(2)).sum(1) / clip_mask.sum(1, keepdims=True)
        if not event_feat_expand:
            assert len(event.shape) == 2
        else:
            event = event.reshape(eseq_num, eseq_len, event.shape[-1])

        for idx in range(eseq_len):
            event_idx = event[event_seq_idx[:, idx]] if not event_feat_expand else event[:, idx]
            clip_idx = clip[event_seq_idx[:, idx]]
            clip_mask_idx = clip_mask[event_seq_idx[:, idx]]

            prev_state_proj = self.global_proj(last_sent_state)
            event_proj = self.local_proj(event_idx)
            gate_input = torch.cat((para_state[0][-1], prev_state_proj, event_proj), 1)
            gate = self.gate_layer(self.gate_drop(gate_input))
            gate = torch.cat((gate, 1 - gate), dim=1)
            sent_rnn_input = torch.cat((prev_state_proj, event_proj), dim=1)
            sent_rnn_input = sent_rnn_input * gate
            _, para_state = self.sent_rnn(sent_rnn_input.unsqueeze(0), para_state)

            para_c, para_h = para_state
            num_layers, batch_size, para_dim = para_h.size()
            init_h = self.para_transfer_layer(para_h[-1]).reshape(self.num_layers, batch_size, para_dim)
            state = (init_h, init_h)

            seq = []
            seqLogprobs = []
            last_sent_state = clip.new_zeros(eseq_num, self.rnn_size)

            for t in range(self.max_caption_len + 1):
                if t == 0:  # input <bos>
                    it = clip.new_zeros(eseq_num).long()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, Variable(it,
                                                                 requires_grad=False))  # gather the logprobs at sampled positions
                    it = it.view(-1).long()  # and flatten indices for downstream processing

                logprobs, state = self.get_logprobs_state(it, event_idx, clip_idx, clip_mask_idx, state)

                if t >= 1:
                    # stop when all finished
                    if t == 1:
                        unfinished = it > 0
                        interest = ~unfinished
                    else:
                        new_unfinished = unfinished & (it > 0)
                        interest = new_unfinished ^ unfinished
                        unfinished = new_unfinished
                    it = it * unfinished.type_as(it)
                    seq.append(it)  # seq[t] the input of t+2 time step
                    seqLogprobs.append(sampleLogprobs.view(-1))
                    if unfinished.sum() == 0:
                        break
                    if interest.sum() > 0:
                        end_id = interest.nonzero().squeeze(1)
                        last_sent_state[end_id] = state[0][-1, end_id]
            if len(seq) == 0:
                seq.append(clip.new_zeros(1).long())
                seqLogprobs.append(clip.new_zeros(1, requires_grad=True) - 10)

            para_seqLogprobs.append(torch.stack(seqLogprobs, 0))  # para_seqLogprobs: (eseq_len, seq_len, batch_size)
            para_seq.append(torch.stack(seq, 0))  # para_seq： (eseq_len, seq_len, batch_size)

        max_len = max([p.shape[0] for p in para_seqLogprobs])
        para_seqLogprobs_tensor = clip.new_zeros(eseq_len, max_len, eseq_num)
        para_seq_tensor = clip.new_zeros(eseq_len, max_len, eseq_num).int()

        for i in range(para_seq_tensor.shape[0]):
            para_seqLogprobs_tensor[i, :len(para_seqLogprobs[i])] = para_seqLogprobs[i]
            para_seq_tensor[i, :len(para_seq[i])] = para_seq[i]
        para_seqLogprobs_tensor = para_seqLogprobs_tensor.permute(2, 0, 1)
        para_seq_tensor = para_seq_tensor.permute(2, 0, 1)
        return para_seq_tensor, para_seqLogprobs_tensor


class ShowAttendTellCore(nn.Module):

    def __init__(self, opt):
        super(ShowAttendTellCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size

        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob
        # self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.att_hid_size

        self.opt = opt
        self.wordRNN_input_feats_type = opt.wordRNN_input_feats_type
        self.input_dim = self.decide_input_feats_dim()

        self.rnn = nn.LSTM(self.input_encoding_size + self.input_dim,
                                                      self.rnn_size, self.num_layers, bias=False,
                                                      dropout=self.drop_prob_lm)

        if self.att_hid_size > 0:
            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1)
        else:
            self.ctx2att = nn.Linear(self.att_feat_size, 1)
            self.h2att = nn.Linear(self.rnn_size, 1)

    def decide_input_feats_dim(self):
        dim = 0
        if 'E' in self.wordRNN_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.wordRNN_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim

    def get_input_feats(self, event, att_clip):
        input_feats = []
        if 'E' in self.wordRNN_input_feats_type:
            input_feats.append(event)
        if 'C' in self.wordRNN_input_feats_type:
            input_feats.append(att_clip)

        input_feats = torch.cat(input_feats, 1)
        return input_feats

    def forward(self, xt, event, clip, clip_mask, state):
        att_size = clip.numel() // clip.size(0) // self.opt.clip_context_dim
        att = clip.view(-1, self.opt.clip_context_dim)

        att = self.ctx2att(att)  # (batch * att_size) * att_hid_size
        att = att.view(-1, att_size, self.att_hid_size)  # batch * att_size * att_hid_size
        att_h = self.h2att(state[0][-1])  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)
        if clip_mask is not None:
            weight = weight * clip_mask.view(-1, att_size).float()
            weight = weight / (weight.sum(1, keepdim=True) + 1e-6)

        att_feats_ = clip.view(-1, att_size, self.att_feat_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        input_feats = self.get_input_feats(event, att_res)
        output, state = self.rnn(torch.cat([xt, input_feats], 1).unsqueeze(0), state)
        return output.squeeze(0), state



class ShowAttendTellModel(CMG_HRNN):
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__(opt)
        self.core = ShowAttendTellCore(opt)



if __name__ == '__main__':
    import opts

    opt = opts.parse_opts()
    opt.wordRNN_input_feats_type = 'C'
    opt.clip_context_type = 'CC+CH'
    opt.vocab_size = 5767
    opt.max_caption_len = 32
    opt.clip_context_dim = 1012

    model = ShowAttendTellModel(opt)

    video = torch.randn(3, 500)
    event = torch.randn(3, 512)
    clip = torch.randn(3, 20, 1012)
    clip_mask = torch.ones(3, 20)
    seq = torch.randint(0, 27, (3, 32))
    # out = model(video, event, clip, clip_mask, seq)
    out = model.sample(video, event, clip, clip_mask)
    pass
