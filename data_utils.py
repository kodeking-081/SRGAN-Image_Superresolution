import os
from math import log10
import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pytorch_ssim  # Ensure this library is installed

# Import your custom modules
from model import Generator  # Replace with the correct path to your model
#from data_utils import display_transform  # Replace with the correct path to your data utilities
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [os.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)




# Optimized TestDatasetFromFolder class
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = os.path.join(dataset_dir, f'SRF_{upscale_factor}', 'data')
        self.hr_path = os.path.join(dataset_dir, f'SRF_{upscale_factor}', 'target')
        self.upscale_factor = upscale_factor

        # Ensure the directories exist
        if not os.path.exists(self.lr_path) or not os.path.exists(self.hr_path):
            raise FileNotFoundError(f"Required directories not found: \n"
                                    f"LR Path: {self.lr_path}\n"
                                    f"HR Path: {self.hr_path}")

        # Collect file names
        self.lr_filenames = [os.path.join(self.lr_path, x) for x in os.listdir(self.lr_path) if self.is_image_file(x)]
        self.hr_filenames = [os.path.join(self.hr_path, x) for x in os.listdir(self.hr_path) if self.is_image_file(x)]

        # Cache transformation
        self.transform = ToTensor()

    def __getitem__(self, index):
        # Extract image names
        image_name = os.path.basename(self.lr_filenames[index])

        # Open images
        lr_image = Image.open(self.lr_filenames[index]).convert('RGB')
        hr_image = Image.open(self.hr_filenames[index]).convert('RGB')

        # Apply transformations
        lr_image = self.transform(lr_image)
        hr_image = self.transform(hr_image)

        # High-resolution restoration image from low-resolution
        hr_restore_img = F.interpolate(lr_image.unsqueeze(0), scale_factor=self.upscale_factor, mode='bicubic', align_corners=False).squeeze(0)

        return image_name, lr_image, hr_restore_img, hr_image

    def __len__(self):
        return len(self.lr_filenames)

    @staticmethod
    def is_image_file(filename):
        """Check if a file is an image."""
        return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'])

# Main testing script
if __name__ == '__main__':
    UPSCALE_FACTOR = 4
    MODEL_NAME = 'netGc_epoch_4_84.pth'

    print(f"Using model: {MODEL_NAME}")

    # Initialize model
    model = Generator(UPSCALE_FACTOR).eval()
    model.load_state_dict(torch.load(f'epochs/{MODEL_NAME}', map_location=torch.device('cpu')))

    # Load dataset
    test_set = TestDatasetFromFolder('G:/My Drive/SR/dataset', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

    # Ensure output directory exists
    out_path = f'benchmark_results/SRF_{UPSCALE_FACTOR}/'
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize results storage
    results = {'psnr': [], 'ssim': []}
    test_bar = tqdm(test_loader, desc='[Testing benchmark datasets]')

    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        with torch.no_grad():
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)

        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).mean().item()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        # Save results
        test_images = torch.stack([
            display_transform()(hr_restore_img.squeeze(0)),
            display_transform()(hr_image.cpu().squeeze(0)),
            display_transform()(sr_image.cpu().squeeze(0))
        ])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        new_image_name = f"{image_name.split('.')[0]}_psnr_{psnr:.4f}_ssim_{ssim:.4f}.{image_name.split('.')[-1]}"
        utils.save_image(image, os.path.join(out_path, new_image_name), padding=5)

        results['psnr'].append(psnr)
        results['ssim'].append(ssim)

    # Save statistics
    stats_path = 'statistics/'
    os.makedirs(stats_path, exist_ok=True)
    data_frame = pd.DataFrame(results, columns=['psnr', 'ssim'])
    data_frame.to_csv(f'{stats_path}srf_{UPSCALE_FACTOR}_test_results.csv', index_label='Dataset')

    # Plot results
    for metric in ['psnr', 'ssim']:
        plt.figure()
        plt.plot(range(len(results[metric])), results[metric], marker='o')
        plt.xlabel("Image Index")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs Image Index")
        plt.grid()
        plt.show()