import cv2
import os



def downsample_images(input_dir, output_dir, scale=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        lr_img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_dir, img_name), lr_img)

# Example usage
downsample_images('static/upload/Set14/HR','static/upload/Set14/LR', scale=4)
