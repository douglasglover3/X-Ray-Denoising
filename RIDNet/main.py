import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scripts.ridnet import RIDNet  #

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
    return transform(image).unsqueeze(0)  # Add batch dimension


def postprocess_image(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)  # Denormalize
    return transforms.ToPILImage()(tensor)


# X-ray image file paths (replace with actual paths)
xray_images = ['data/final/test/00000002_000.png', 'data/final/test/00000003_000.png',
               'data/final/test/00000003_001.png']

# Denoise and display images
for i, image_path in enumerate(xray_images):
    try:
        # Preprocess image
        input_image = preprocess_image(image_path)

        # Denoise with the model
        with torch.no_grad():
            denoised_image = model(input_image)

        # Postprocess and display the image
        output_image = postprocess_image(denoised_image)
        plt.figure()
        plt.imshow(output_image, cmap='gray')
        plt.title(f'Denoised Image {i + 1}')
        plt.axis('off')
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

plt.show()
