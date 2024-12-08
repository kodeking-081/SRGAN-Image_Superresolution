import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

def main():
 parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
 parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
 parser.add_argument('--model_name', default='netG_epoch_4_50.pth', type=str, help='generator model epoch name')
 opt = parser.parse_args()

 UPSCALE_FACTOR = opt.upscale_factor
 MODEL_NAME = opt.model_name

 results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

 model = Generator(UPSCALE_FACTOR).eval()
 if torch.cuda.is_available():
    model = model.cuda()
 model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

 test_set = TestDatasetFromFolder('dataset/test', upscale_factor=UPSCALE_FACTOR)
 test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
 test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

 out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
 if not os.path.exists(out_path):
    os.makedirs(out_path)

 for image_name, lr_image, hr_restore_img, hr_image in test_bar:
    image_name = image_name[0]

    with torch.no_grad():
     lr_image = Variable(lr_image)
     hr_image = Variable(hr_image)

    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    sr_image = model(lr_image)

 # Check the shape of the images
    print(f"sr_image shape: {sr_image.shape}, hr_image shape: {hr_image.shape}")

    # Ensure both sr_image and hr_image have the same dimensions
    # Resize sr_image if necessary to match hr_image's size
    if sr_image.shape[2:] != hr_image.shape[2:]:
        sr_image_resized = F.interpolate(sr_image, size=hr_image.shape[2:], mode='bilinear', align_corners=False)
        print(f"Resized sr_image to: {sr_image_resized.shape}")
    else:
        sr_image_resized = sr_image

    # Calculate MSE
    mse = ((hr_image - sr_image_resized) ** 2).data.mean()
    psnr = 10 * log10(1 / mse)
    ssim = pytorch_ssim.ssim(sr_image_resized, hr_image).item()

    test_images = torch.stack(
        [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
         display_transform()(sr_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=3, padding=5)
    utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                     image_name.split('.')[-1], padding=5)

    # save psnr\ssim
   # Correct use of setdefault
    image_key = image_name.split('_')[0].split('.')[0]  # Remove extension if necessary
    results.setdefault(image_key, {'psnr': []})['psnr'].append(psnr)

    if image_key not in results:
     results[image_key] = {'ssim': []}  # Initialize if not present

     results[image_key]['ssim'].append(ssim)


 out_path = 'statistics/'
 saved_results = {'psnr': [], 'ssim': []}
 for item in results.values():
    # Check if 'ssim' and 'psnr' keys exist
    if 'psnr' not in item:
        item['psnr'] = []  # Initialize if missing
    if 'ssim' not in item:
        item['ssim'] = []  # Initialize if missing

    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])

    # Check if lists are empty
    if len(psnr) == 0 or len(ssim) == 0:
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()

 saved_results['psnr'].append(psnr)
 saved_results['ssim'].append(ssim)

 data_frame = pd.DataFrame(saved_results, results.keys())
 data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')


if __name__ == '__main__':
    main()
