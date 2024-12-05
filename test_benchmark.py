import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

if __name__ == '__main__':
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
    parser.add_argument('--upscale_factor', default=4, type=int, help='Super resolution upscale factor')
    parser.add_argument('--model_name', default='netGc_epoch_4_84.pth', type=str, help='Generator model epoch name')
    opt = parser.parse_args()

    # Parameters
    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = Generator(UPSCALE_FACTOR).eval()
    model.load_state_dict(torch.load(f'epochs/{MODEL_NAME}', map_location=device))
    model = model.to(device)

    # Prepare Dataset and Dataloader
    test_set = TestDatasetFromFolder('G:\\My Drive\\SR\\dataset', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[Testing Benchmark Datasets]')

    # Output Directories
    out_path = f'benchmark_results/SRF_{UPSCALE_FACTOR}/'
    os.makedirs(out_path, exist_ok=True)
    stat_path = 'statistics/'
    os.makedirs(stat_path, exist_ok=True)

    results = {'psnr': [], 'ssim': []}

    # Testing Loop
    with torch.no_grad():
        for image_name, lr_image, hr_restore_img, hr_image in test_bar:
            # Move tensors to device
            lr_image = lr_image.to(device).float()
            hr_image = hr_image.to(device).float()

            # Process SR Image
            sr_image = model(lr_image)

            # Calculate Metrics
            mse = ((hr_image - sr_image) ** 2).mean().item()
            psnr = 10 * log10(1 / mse) if mse > 0 else float('inf')
            ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

            # Save Results
            test_images = torch.stack([
                display_transform()(hr_restore_img.squeeze(0)),
                display_transform()(hr_image.cpu().squeeze(0)),
                display_transform()(sr_image.cpu().squeeze(0))
            ])
            image = utils.make_grid(test_images, nrow=3, padding=5)
            utils.save_image(image, f"{out_path}{image_name[0]}_psnr_{psnr:.4f}_ssim_{ssim:.4f}.png")

            results['psnr'].append(psnr)
            results['ssim'].append(ssim)

            # Clear memory
            del lr_image, hr_image, sr_image
            torch.cuda.empty_cache()

    # Save Metrics to CSV
    pd.DataFrame(results, columns=['psnr', 'ssim']).to_csv(f"{stat_path}srf_{UPSCALE_FACTOR}_test_results.csv", index_label='DataSet')

    # Debugging Memory Usage (Optional)
    import psutil
    print(f"Final Memory Usage: {psutil.virtual_memory().percent}%")
