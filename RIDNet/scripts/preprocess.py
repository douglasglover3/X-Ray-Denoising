import cv2
import os
import numpy as np
from glob import glob

'''
function contains code to convert image to grayscale, resize to 256x256, and normalize pixel values
'''


def preprocess_images(input_folder, output_folder, image_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in glob(os.path.join(input_folder, '*.*')):
        # Read image
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to 256x256
        resized_img = cv2.resize(gray_img, image_size)

        # Normalize pixel values
        normalized_img = resized_img / 255.0

        # Save preprocessed image
        output_path = os.path.join(output_folder, os.path.basename(file_path))
        cv2.imwrite(output_path, (normalized_img * 255).astype(np.uint8))
        print(f"Preprocessed and saved: {output_path}")


# Preprocess dataset from raw folder
preprocess_images('../data/raw', '../data/preprocessed')
