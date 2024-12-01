import cv2
import time
import glob
import os
from PyramidalDenoiser import GaussianPyramid
from PyramidalDenoiser import LaplacianPyramid
from PyramidalDenoiser import add_gaussian_noise
from PyramidalDenoiser import reconstruct_image
from PyramidalDenoiser import calculate_ssim
from PyramidalDenoiser import calculate_psnr

def denoise_images():
    images = glob.glob(os.path.join('Data/Images/', '*.png'))
    for file_path in images:
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