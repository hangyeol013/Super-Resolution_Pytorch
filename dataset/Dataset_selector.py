


def DatasetSR(sr_mode):

    if sr_mode == 'SRCNN':
        from .DatasetSRCNN import DatasetSRCNN
        DatasetSR = DatasetSRCNN
    elif sr_mode == 'FSRCNN':
        from .DatasetFSRCNN import DatasetFSRCNN
        DatasetSR = DatasetFSRCNN
    elif sr_mode == 'VDSR':
        from .DatasetVDSR import DatasetVDSR
        DatasetSR = DatasetVDSR
    elif sr_mode == 'EDSR':
        from .DatasetEDSR import DatasetEDSR
        DatasetSR = DatasetEDSR

    return DatasetSR

