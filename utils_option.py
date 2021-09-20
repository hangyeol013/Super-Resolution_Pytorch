
import json
import logging
import numpy as np
import cv2
import math

import torch.nn as nn
import torch.optim as optim



'''
# ----------------------------
# Parse
# ----------------------------
'''
def parse(opt_path):
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str)
    return opt

'''
# ----------------------------
# Make logger
# ----------------------------
'''
def make_logger(file_path, name=None):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # file_handler = logging.FileHandler(filename=file_path, mode='w')
        # file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)

    return logger

'''
# ----------------------------------
# Loss function
# ----------------------------------
'''
def select_lossfn(opt, reduction='mean'):
    if opt == 'l1':
        lossfn = nn.L1Loss(reduction=reduction)
    elif opt == 'l2':
        lossfn = nn.MSELoss(reduction=reduction)
    return lossfn

'''
# ----------------------------------
# Optimizer
# ----------------------------------
'''
def select_optimizer(opt, lr, model):
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    return optimizer



'''
# ----------------------------
# Calculate PSNR
# ----------------------------
'''
def calculate_psnr(img1, img2, border = 0):

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


'''
# ----------------------------
# Calculate SSIM
# ----------------------------
'''
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, border = 0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions')