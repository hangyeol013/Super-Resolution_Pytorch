import numpy as np
import random
import os
import cv2
from collections import OrderedDict
import argparse

import torch
import utils_option
from utils_option import make_logger, parse
from dataset.Dataset_selector import DatasetSR
from models.model_selector import modelSR


def test(args):
    '''
    # ----------------------------------------
    # Seed & Settings
    # ----------------------------------------
    '''
    sr_mode = args['sr_mode']
    cuda = torch.cuda.is_available()

    logger_name = args['logger_name']
    logger_path = args['logger_path']
    logger = make_logger(file_path=logger_path, name=logger_name)

    seed = args['seed']
    if seed is None:
        seed = random.randint(1, 100)
    logger.info('Random Seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info(f'cuda: {cuda}')
    logger.info(f'SR mode: {sr_mode}')

    args = args[sr_mode]
    scale = args['dataset']['test']['scale']
    batch_size = args['test']['batch_size']
    epoch = args['test']['epoch']

    logger.info(f'scale: {scale}, epoch:{epoch}, batch size:{batch_size}')


    '''
    # ----------------------
    # Dataset
    # ----------------------
    '''
    Dataset_SR = DatasetSR(sr_mode)
    args_dataset = args['dataset']
    test_set = Dataset_SR(args_dataset['test'])
    logger.info(f'test_set: {len(test_set)}')

    test_base = args['dataset']['test']['base_path']
    testset_name = args['dataset']['test']['test_set']
    test_path = os.path.join(test_base, testset_name)

    E_path = f'results/{testset_name}'
    if not os.path.exists(E_path):
        os.makedirs(E_path)


    '''
    # ----------------------
    # Model & Settings
    # ----------------------
    '''

    base_path = 'model_zoo'
    model_name = f'{sr_mode}_s{scale}e{epoch}b{batch_size}.pth'
    model_path = os.path.join(base_path, model_name)

    model = modelSR(sr_mode, scale)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict = True)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
    if cuda:
        model = model.cuda()

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    result_name = testset_name + '_' + model_name

    logger.info(f' Model: {model_name}')


    '''
    # ----------------------
    # Test
    # ----------------------
    '''

    for idx in range(len(test_set)):

        img_name, ext = os.path.splitext(os.listdir(test_path)[idx])
        test_img = test_set[idx]
        img_I = test_img['in']
        img_H = test_img['hr']

        img_I = img_I.unsqueeze(0)
        img_I_save = np.uint8((img_I.squeeze()*255).round())
        img_I_save = np.transpose(img_I_save, (1,2,0))

        if cuda:
            img_I, img_H = img_I.cuda(), img_H.cuda()

        img_E = model(img_I, scale)
        img_E = img_E.data.squeeze().float().clamp_(0,1).cpu().numpy()
        img_E = np.transpose(img_E, (1,2,0))
        img_E = np.uint8((img_E * 255).round())

        img_H = np.uint8((img_H.squeeze() * 255).round())
        img_H = np.transpose(img_H, (1,2,0))


    '''
    # ----------------------
    # Calculate Metrics, Save images
    # ----------------------
    '''

    psnr = utils_option.calculate_psnr(img_E, img_H, border = args['test']['border'])
    ssim = utils_option.calculate_ssim(img_E, img_H, border = args['test']['border'])
    test_results['psnr'].append(psnr)
    test_results['ssim'].append(ssim)
    logger.info('{} - PSNR:  {:.2f} dB;  SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
    img_E_path = f'{img_name}_E_{scale}.png'
    img_I_path = f'{img_name}_I.png'
    cv2.imwrite(os.path.join(E_path, img_I_path), img_I_save)
    cv2.imwrite(os.path.join(E_path, img_E_path), img_E)

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])

    logger.info('Average PSNR/SSIM - {} \
                - PSNR: {:.2f} dB,  SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))




def main(json_path = 'Implementation.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args().opt)
    test(args)


if __name__ == '__main__':
    main()
