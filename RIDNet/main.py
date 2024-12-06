import torch
from torchvision import transforms
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from scripts.ridnet import RIDNet
from skimage.metrics import structural_similarity as ssim

# Load the RIDNet model
model = RIDNet()
model.load_state_dict(torch.load('./ridnet.pth', map_location=torch.device('cpu')))
model.eval()


# Define image preprocessing and postprocessing functions
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure grayscale
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize (mean=0.5, std=0.5)
    ])
    image = Image.open(image_path).convert('L')  # Open and convert to grayscale
    return transform(image).unsqueeze(0), image  # Add batch dimension and return original


def postprocess_image(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)  # Denormalize
    return transforms.ToPILImage()(tensor)


def calculate_ssim(image1, image2):
    """
    Calculate SSIM between two images.

    :param image1: First image (numpy array).
    :param image2: Second image (numpy array).
    :return: SSIM score.
    """
    return ssim(image1, image2, data_range=image2.max() - image2.min())


# X-ray image file paths
xray_images = ['data/final/test/00000002_000.png', 'data/final/test/00000003_000.png',
               'data/final/test/00000003_001.png']

# Initialize variables to calculate average denoising time
total_time = 0
image_count = len(xray_images)

# Modify denoise and display loop to include SSIM calculation
for i, image_path in enumerate(xray_images):
    try:
        # Start timing
        start_time = time.time()

        # Preprocess image
        input_tensor, original_image = preprocess_image(image_path)

        # Denoise with the model
        with torch.no_grad():
            denoised_tensor = model(input_tensor)

        # Stop timing
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        print(f"Time taken for Image {i + 1}: {elapsed_time:.4f} seconds")

        # Postprocess the image
        denoised_image = postprocess_image(denoised_tensor)

        # Calculate SSIM
        original_np = np.array(original_image)
        denoised_np = np.array(denoised_image)
        ssim_score = calculate_ssim(original_np, denoised_np)
        print(f"SSIM for Image {i + 1}: {ssim_score}")

        # Display original, preprocessed, and denoised images
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f'Original Image {i + 1}')
        plt.axis('off')

        # Preprocessed image
        plt.subplot(1, 3, 2)
        plt.imshow(input_tensor.squeeze().numpy(), cmap='gray')
        plt.title(f'Preprocessed Image {i + 1}')
        plt.axis('off')

        # Postprocessed (Denoised) image
        plt.subplot(1, 3, 3)
        plt.imshow(denoised_image, cmap='gray')
        plt.title(f'Denoised Image {i + 1} (SSIM: {ssim_score:.4f})')
        plt.axis('off')

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Calculate and print average time
average_time = total_time / image_count
print(f"Average time to denoise images: {average_time:.4f} seconds")

plt.show()


