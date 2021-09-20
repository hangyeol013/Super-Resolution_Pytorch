import cv2
import os
import numpy as np




def _read_image_RGB(image_path, scale):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    h_new = (h//scale) * scale
    w_new = (w//scale) * scale
    image = image[:h_new, :w_new, ...]

    return image


def load_images(phase, scale, dataset_name):
    base_path = 'dataset/datasets/'
    if phase == 'train':
        path_dir = f'train_set/{dataset_name}/'
    elif phase == 'test':
        path_dir = f'test_set/{dataset_name}/'
    image_dir = base_path + path_dir
    images = []
    for fn in next(os.walk(image_dir))[2]:
        image = _read_image_RGB(image_dir + fn, scale)
        images.append(image)

    return images

# rgb -> ycbcr
# change ycbcr -> cbcry (to calculate cv2.resize)
def rgbs_to_ycbcrs(imgs):

    ycbcrs = []
    for img in imgs:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 91.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        img_ycbcr = np.array([y,cb,cr])
        ycbcrs.append(img_ycbcr)
    return ycbcrs

# hr_ycbcr to lr_ycbcr
# Caution: cv2.resize: dsize=(width,height)
def hrs_to_lrs(hr_imgs, scale):
    lr_imgs = []
    for hr in hr_imgs:
        hr_img = hr.transpose(1,2,0)
        hr_h = hr_img.shape[0]
        hr_w = hr_img.shape[1]
        lr_img = cv2.resize(hr_img, dsize=(hr_w//scale, hr_h//scale), interpolation=cv2.INTER_CUBIC)
        lr_imgs.append(lr_img.transpose(2,0,1))
    return lr_imgs


# lr_ycbcr to in_ycbcr
# Caution: cv2.resize: dsize=(width,height)
def lrs_to_ins(lr_imgs, scale):

    in_imgs = []
    for lr in lr_imgs:
        lr_img = lr.transpose(1,2,0)
        lr_h = lr_img.shape[0]
        lr_w = lr_img.shape[1]
        in_img = cv2.resize(lr_img, dsize=(lr_w*scale, lr_h*scale), interpolation=cv2.INTER_CUBIC)
        in_imgs.append(in_img.transpose(2,0,1))
    return in_imgs


# from hr_ycbcr, in_ycbcr to hr_patches, in_patches
def ycbcrs_to_patches(in_ycbcrs, hr_ycbcrs, patch_size, stride, scale):

    in_patches = []
    hr_patches = []

    for hr_ycbcr, in_ycbcr in zip(hr_ycbcrs, in_ycbcrs):
        for i in range(0, in_ycbcr.shape[1] - patch_size + 1, stride):
            for j in range(0, in_ycbcr.shape[2] - patch_size + 1, stride):

                if hr_ycbcr.shape == in_ycbcr.shape:
                    in_patch = in_ycbcr[:, i:i+patch_size, j:j+patch_size]
                    hr_patch = hr_ycbcr[:, i:i+patch_size, j:j+patch_size]
                else:
                    in_patch = in_ycbcr[:, i:i+patch_size, j:j+patch_size]
                    hr_patch = hr_ycbcr[:, i*scale:(i+patch_size)*scale, j*scale:(j+patch_size)*scale]

                in_patches.append(in_patch)
                hr_patches.append(hr_patch)

    in_patches = np.array(in_patches)
    hr_patches = np.array(hr_patches)

    return in_patches, hr_patches



def main():

    H_images = load_images('train', 2, 'T91')
    hr_ycbcrs = rgbs_to_ycbcrs(H_images)
    lr_ycbcrs = hrs_to_lrs(hr_ycbcrs, 2)
    in_ycbcrs = lrs_to_ins(lr_ycbcrs, 2)
    in_pat, hr_pat = ycbcrs_to_patches(hr_ycbcrs, in_ycbcrs, 64, 14, 2)



if __name__ == '__main__':
    main()


