import os
import torch
import cv2
import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
#from torchvision.transforms import ToTensor, ToPILImag
from model import Generator

scale_factor =4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator(scale_factor).to(device)
model.load_state_dict(torch.load('F:/SRGAN/epochs/netG_epoch_4_300.pth', map_location=torch.device('cpu'))) # Load trained model
model.eval()

def super_resolve_images(lr_dir, sr_dir, model):
    if not os.path.exists(sr_dir):
        os.makedirs(sr_dir)
    
    for img_name in os.listdir(lr_dir):
        lr_path = os.path.join(lr_dir, img_name)
        img = cv2.imread(lr_path)
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize
        img = img.to(device)
        with torch.no_grad():
            sr_img = model(img).cpu().squeeze().permute(1, 2, 0).numpy() * 255.0
            sr_img = sr_img.clip(0, 255).astype('uint8')
            cv2.imwrite(os.path.join(sr_dir, img_name), sr_img)

# Example usage
super_resolve_images("F:/SRGAN/static/upload/Set14/LR", "F:/SRGAN/static/upload/Set14/SR", model)
