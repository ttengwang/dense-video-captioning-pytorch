from models.caption_decoder.ShowAttendTell import ShowAttendTellModel
from models.caption_decoder.HRNN import ShowAttendTellModel as HRNN
from models.caption_decoder.CMG_HRNN import ShowAttendTellModel as CMG_HRNN
from models.EventEncoder import Basic_Encoder, TSRM_Encoder

def setup_caption_decoder(opt):
    if opt.caption_decoder_type == 'show_attend_tell':
        model = ShowAttendTellModel(opt)
    elif opt.caption_decoder_type == 'hrnn':
        model = HRNN(opt)
    elif opt.caption_decoder_type == 'cmg_hrnn':
        model = CMG_HRNN(opt)
    else:
        raise AssertionError('args error: caption_model')
    return model

def setup_event_encoder(opt):
    if opt.event_encoder_type == 'basic':
        model = Basic_Encoder(opt)
    elif opt.event_encoder_type == 'tsrm':
        model = TSRM_Encoder(opt)
    else:
        raise AssertionError('args error: event_encoder_type')
    return model