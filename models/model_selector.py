
def modelSR(sr_mode, scale):

    if sr_mode == 'SRCNN':
        from .Network_SRCNN import SRCNN
        sr_net = SRCNN()
    elif sr_mode == 'FSRCNN':
        from .Network_FSRCNN import FSRCNN
        sr_net = FSRCNN(scale)
    elif sr_mode == 'VDSR':
        from .Network_VDSR import VDSR
        sr_net = VDSR()
    elif sr_mode == 'EDSR':
        from .Network_EDSR import EDSR
        sr_net = EDSR(scale)

    return sr_net
