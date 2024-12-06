import cv2
import time
import glob
import os

import numpy as np
from PyramidalDenoiser import GaussianPyramid
from PyramidalDenoiser import LaplacianPyramid
from PyramidalDenoiser import add_gaussian_noise
from PyramidalDenoiser import reconstruct_image
from PyramidalDenoiser import calculate_ssim
from PyramidalDenoiser import calculate_psnr
import matplotlib.pyplot as plt

def denoise_images():
    images = glob.glob(os.path.join('Data/images/', '*.png'))[:50]
    total_psnr = 0
    total_ssim = 0
    total_time = 0
    total_count = 0
    for file_path in images:
        total_count += 1
        print(f'Loading image: {file_path}')
        original_img = cv2.imread(file_path)
        noisy_img = add_gaussian_noise(original_img)
        noisy_loc = 'Noisy/' + os.path.basename(file_path)
        #cv2.imwrite(noisy_loc, noisy_img)
        start = time.time()
        gaussian_pyr = GaussianPyramid(noisy_img)
        laplacian_pyr = LaplacianPyramid(gaussian_pyr)

        laplacian_levels = laplacian_pyr.build_pyramid()

        reconstructed_image = reconstruct_image(laplacian_levels)
        end = time.time()

        elapsed = end - start
        print(f'Elapsed time: {elapsed} seconds')
        loc = 'Output/' + os.path.basename(file_path)
        #cv2.imwrite(loc, reconstructed_image)
        psnr_val = calculate_psnr((original_img / 255).astype(np.float32), (noisy_img / 255).astype(np.float32))
        ssim_val = calculate_ssim((original_img / 255).astype(np.float32), (noisy_img / 255).astype(np.float32))
        total_psnr += psnr_val
        total_ssim += ssim_val
        total_time += elapsed
        print(f'Peak signal to noise ratio between original image and denoised image: {psnr_val}')
        print(f'Structural similarity index between original image and denoised image: {ssim_val}')

    avg_psnr = total_psnr / total_count
    avg_ssim = total_ssim / total_count
    avg_time = total_time / total_count
    print(f'Average PSNR: {avg_psnr}')
    print(f'Average SSIM: {avg_ssim}')
    print(f'Average time taken: {avg_time}')

    print('Saving results...')
    # Create a table to save the results
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table_data = [
        ['Metric', 'Average Value'],
        ['PSNR', f'{avg_psnr:.2f}'],
        ['SSIM', f'{avg_ssim:.2f}'],
        ['Time (s)', f'{avg_time:.2f}']
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.2)

    # Save the table as a PNG file
    plt.savefig('average_metrics_table.png')
    plt.close()

    print('Table saved as average_metrics_table.png')