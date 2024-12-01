import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import glob
import time

class GaussianPyramid:
    def __init__(self, image, levels=5):
        self.image = image
        self.levels = levels
        self.pyramid = [image]

    def build_pyramid(self):
        current_level = self.image
        for _ in range(self.levels):
            current_level = cv2.pyrDown(current_level)
            self.pyramid.append(current_level)
        return self.pyramid


class LaplacianPyramid:
    def __init__(self, gaussian_pyramid):
        self.gaussian_pyramid = gaussian_pyramid
        self.pyramid = []

    def build_pyramid(self):
        gaussian_levels = self.gaussian_pyramid.build_pyramid()
        for i in range(len(gaussian_levels) - 1):
            GE = cv2.pyrUp(gaussian_levels[i + 1], dstsize=(gaussian_levels[i].shape[1], gaussian_levels[i].shape[0]))
            L = cv2.subtract(gaussian_levels[i], GE)
            self.pyramid.append(L)
        self.pyramid.append(gaussian_levels[-1])
        return self.pyramid


def reconstruct_image(laplacian_pyramid):
    image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        image = cv2.pyrUp(image, dstsize=(laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
        image = cv2.add(image, laplacian_pyramid[i])
    return image

def add_gaussian_noise(image, mean=0, std=0.1):
    # Generate Gaussian noise
    gaussian = np.random.normal(mean, std, image.shape)

    #  Add the gaussian noise to the image
    noisy_image = np.clip(image + gaussian * 255, 0, 255).astype(np.uint8)

    return noisy_image

def calculate_psnr(original_image, denoised_image):
    mse = np.mean((original_image - denoised_image) ** 2)
    if mse == 0:
        return float('inf') # Infinite PSNR, meaning no noise at all
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calculate_ssim(original_image, denoised_image):
    # Convert images to grayscale (SSIM only works on grayscale)
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    ssim_value, _ = ssim(original_gray, denoised_gray, full=True)
    return ssim_value

for file_path in glob.glob('Data/*.png'):
    print(f'Loading image: {file_path}')
    original_img = cv2.imread(file_path)
    noisy_img = add_gaussian_noise(original_img)
    noisy_loc = 'Noisy Images/' + os.path.basename(file_path)
    cv2.imwrite(noisy_loc, noisy_img)

    start = time.time()
    gaussian_pyr = GaussianPyramid(noisy_img)
    laplacian_pyr = LaplacianPyramid(gaussian_pyr)

    laplacian_levels = laplacian_pyr.build_pyramid()

    reconstructed_image = reconstruct_image(laplacian_levels)
    end = time.time()

    elapsed = end - start
    print(f'Elapsed time: {elapsed} seconds')
    loc = 'Output/' + os.path.basename(file_path)
    cv2.imwrite(loc, reconstructed_image)
    psnr_val = calculate_psnr(original_img, reconstructed_image)
    ssim_val = calculate_ssim(original_img, reconstructed_image)
    print(f'Peak signal to noise ratio between original image and denoised image: {psnr_val}')
    print(f'Structural similarity index between original image and denoised image: {ssim_val}')
