import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scripts.ridnet import RIDNet

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


# X-ray image file paths
xray_images = ['data/final/test/00000002_000.png', 'data/final/test/00000003_000.png',
               'data/final/test/00000003_001.png']

# Denoise and display images
for i, image_path in enumerate(xray_images):
    try:
        # Preprocess image
        input_tensor, original_image = preprocess_image(image_path)

        # Denoise with the model
        with torch.no_grad():
            denoised_tensor = model(input_tensor)

        # Postprocess the image
        denoised_image = postprocess_image(denoised_tensor)

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
        plt.title(f'Denoised Image {i + 1}')
        plt.axis('off')
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

plt.show()
