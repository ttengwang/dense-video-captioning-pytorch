from models.EventEncoder import BasicEncoder, RNNEncoder, BRNNEncoder

def setup_event_encoder(opt):
    if opt.event_encoder_type == 'basic':
        model = BasicEncoder(opt)
    elif opt.event_encoder_type == 'brnn':
        model = BRNNEncoder(opt)
    elif opt.event_encoder_type == 'rnn':
        model = RNNEncoder(opt)
    else:
        raise AssertionError('args error: event_encoder_type')
    return model