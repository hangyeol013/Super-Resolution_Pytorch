from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import argparse
import utils_option
from .utils_data import load_images, rgbs_to_ycbcrs
from .utils_data import hrs_to_lrs, lrs_to_ins, ycbcrs_to_patches


class DatasetSRCNN(Dataset):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.phase = args['phase']
        self.dataset_name = args['dataset_name']
        self.scale = args['scale']
        self.stride = args['stride'] if self.phase == 'train' else None
        self.patch_size = args['patch_size'] if self.phase == 'train' else None
        self.H_images = load_images(self.phase, self.scale, self.dataset_name)
        self.hr_ycbcrs = rgbs_to_ycbcrs(self.H_images)
        self.lr_ycbcrs = hrs_to_lrs(self.hr_ycbcrs, self.scale)
        self.in_ycbcrs = lrs_to_ins(self.lr_ycbcrs, self.scale)
        self.in_patches, self.hr_patches = (self.in_ycbcrs, self.hr_ycbcrs) if self.phase=='test' \
            else ycbcrs_to_patches(self.hr_ycbcrs, self.in_ycbcrs, self.patch_size, self.stride, self.scale)

    def __getitem__(self, index):

        hr_patch = self.hr_patches[index]
        in_patch = self.in_patches[index]

        return {'in': in_patch, 'hr': hr_patch}

    def __len__(self):
        return len(self.in_patches)



def main(json_path = 'Implementation.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = utils_option.parse(parser.parse_args().opt)
    args = args['SRCNN']

    for phase, dataset_opt in args['dataset'].items():
        if phase == 'train':
            train_set = DatasetSRCNN(dataset_opt)
            train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
        elif phase == 'test':
            test_set = DatasetSRCNN(dataset_opt)
            test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)
        else:
            raise NotImplementedError("Phase [%s} is not recognized."%phase)

    for i, train_data in enumerate(train_dataloader):
        print(i, train_data['in'].shape)
    for j, test_data in enumerate(test_dataloader):
        print(j, test_data['in'].shape)
        print(j, test_data['hr'].shape)



if __name__ == '__main__':
    main()






