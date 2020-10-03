import argparse
import time
import yaml
from misc import utils


def parse_opts():
    parser = argparse.ArgumentParser()

    # ID of this run
    parser.add_argument('--cfg_path', type=str, default='cfgs/esgn.yml', help='')
    parser.add_argument('--id', type=str, default='default', help='id of this run')
    parser.add_argument('--gpu_id', type=str, nargs='+', default=['0'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--disable_cudnn', type=int, default=1, help='disable cudnn may solve some unknown bugs')
    parser.add_argument('--debug', action='store_true', help='using mini-dataset for fast debugging')

    #  ***************************** INPUT DATA PATH *****************************
    parser.add_argument('--train_caption_file', type=str,
                        default='data/captiondata/train_modified.json', help='')
    parser.add_argument('--invalid_video_json', type=str, nargs='+', default=['data/DBG_invalid_videos.json'])
    parser.add_argument('--val_caption_file', type=str, default='data/captiondata/val_1.json')
    parser.add_argument('--visual_feature_folder', type=str, default='data/resnet_bn')
    parser.add_argument('--train_proposal_file', type=str, default='',
                        help='generated results on trainset by a Temporal Action Proposal model')
    parser.add_argument('--eval_proposal_file', type=str, default='',
                        help='generated results on valset by a Temporal Action Proposal model')
    parser.add_argument('--visual_feature_type', type=str, default='c3d', choices=['c3d', 'resnet_bn', 'resnet'])
    parser.add_argument('--feature_dim', type=int, default=500, help='dim of frame-level feature vector')
    # parser.add_argument('--dict_file', type=str, default='data/vocabulary_activitynet.json', help='')

    parser.add_argument('--start_from', type=str, default='', help='id of the run with incompleted training')
    parser.add_argument('--start_from_mode', type=str, choices=['best', 'last'], default="last")
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_path', type=str, default='', help='path of .pth')

    #  ***************************** DATALOADER OPTION *****************************
    parser.add_argument('--nthreads', type=int, default=4)
    parser.add_argument('--feature_sample_rate', type=int, default=1)
    parser.add_argument('--train_proposal_sample_num', type=int,
                        default=24,
                        help='number of sampled proposals (or proposal sequence), a bigger value may be better')
    parser.add_argument('--train_proposal_type', type=str, default='', help='gt, learnt_seq, learnt')

    # ***************************** Event ENCODER  *****************************
    parser.add_argument('--event_encoder_type', type=str, choices=['basic', 'TSRM'], default='basic')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden size of all fc layers')
    parser.add_argument('--position_encoding_size', type=int, default=100)

    #  ***************************** DECODER  *****************************
    parser.add_argument('--decoder_input_feats_type', type=str, default='C', choices=['C', 'E', 'C+E'],
                        help='C:clip-level features, E: event-level features, C+E: both')
    parser.add_argument('--decoder_type', type=str, default="show_attend_tell",
                        choices=['show_attend_tell', 'hrnn', 'cmg_hrnn'])
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the decoderRNN')

    parser.add_argument('--att_hid_size', type=int, default=512, help='the hidden size of the attention MLP')
    parser.add_argument('--drop_prob', type=float, default=0.5, help='strength of dropout in the Language Model RNN')
    parser.add_argument('--max_decoding_len', type=int, default=30, help='')

    #  ***************************** OPTIMIZER *****************************

    # optimizer
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size must be 1 when using hrnn')
    parser.add_argument('--grad_clip', type=float, default=100., help='clip gradients at this value')
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

    # lr
    parser.add_argument('--lr', type=float, default=1e-4, help='1e-4 for resnet feature and 5e-5 for C3D feature')
    parser.add_argument('--learning_rate_decay_start', type=float, default=8)
    parser.add_argument('--learning_rate_decay_every', type=float, default=3)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5)

    # scheduled sampling
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                        help='at what iteration to start decay gt probability')
    parser.add_argument('--basic_ss_prob', type=float, default=0, help='initial ss prob')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=1,
                        help='every how many epoch thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    #  ***************************** SAVING AND LOGGING *****************************
    parser.add_argument('--min_epoch_when_save', type=int, default=-1)
    parser.add_argument('--save_checkpoint_every', type=int, default=1)
    parser.add_argument('--save_all_checkpoint', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./save', help='directory to store checkpointed models')

    #  ***************************** Evaluation *************************************
    parser.add_argument('--eval_score_threshold', type=float, default=0.)
    parser.add_argument('--eval_nms_threshold', type=float, default=1)
    parser.add_argument('--eval_top_n', type=int, default=100)
    args = parser.parse_args()

    if args.cfg_path:
        with open(args.cfg_path, 'r') as handle:
            options_yaml = yaml.load(handle)
        utils.update_values(options_yaml, vars(args))

    args.raw_feature_dim = args.feature_dim

    if args.debug:
        args.id = 'debug_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.min_epoch_when_save = 0
        args.save_checkpoint_every = 1
        args.shuffle = 0
        args.tap_epochs = 10
        args.train_caption_file = 'data/captiondata/train_modified_small.json'
        args.val_caption_file = 'data/captiondata/val_1_small.json'

    return args


if __name__ == '__main__':
    opt = parse_opts()
    print(opt)
