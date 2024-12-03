import argparse
from PIL import Image
import torch
import numpy as np
from TrainTest import calculate_psnr, calculate_ssim
from AutoEncoder import AutoEncoder
import matplotlib.pyplot as plt
from torchvision import transforms
from AddGaussianNoise import AddGaussianNoise

def apply_autoencoder(autoencoder: AutoEncoder, original, noisy):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_dataset: dataset of test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    autoencoder.eval()
    
    images = []
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        # Predict for data by doing forward pass
        output: torch.Tensor = autoencoder(noisy)
        
        original_image = original.permute(1, 2, 0).view(1024, 1024).numpy()
        noisy_image = noisy.permute(1, 2, 0).view(1024, 1024).numpy()
        output_image = output.permute(1, 2, 0).detach().view(1024, 1024).numpy()
        psnr = calculate_psnr(original_image, output_image)
        ssim = calculate_ssim(original_image, output_image)

        #Grab one image from batch for examination
        images.append(('Original Image', original_image))
        images.append(('Noisy Image', noisy_image))
        images.append(('Decoded Image', output_image))
            
        print('PSNR: ' + str(psnr))
        print('SSIM: ' + str(ssim))
            
    print('Saving image as result.png...')
    #Save one image from each batch
    fig, axes = plt.subplots(1, 3)
    for index, (title, image) in enumerate(images):
        im = axes[index % 3]
        im.imshow(image, cmap='gray')
        im.set_title(title)
        im.axis('off')
        if index % 3 == 2:
            plt.tight_layout()
            plt.savefig(f'./Autoencoder/result.png', dpi=300)
    print('Done.')



#Transform into tensor and convert to grayscale
tensor_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#Transform into tensor, convert to grayscale, and adds noise
noisy_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    AddGaussianNoise(0., 0.1)
])

parser = argparse.ArgumentParser('Autoencoder apply to image')
parser.add_argument('--image', type=str, default='data/images/00000001_000.png', help='Path for image to apply.')
FLAGS = None
FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image

#Get image and noisy image
image = Image.open(image_path)
original_image = tensor_transform(image) #load original image
noisy_image = noisy_transform(image) #load noisy image

#Get model
model = AutoEncoder().to('cpu')
model.load_state_dict(torch.load('./Autoencoder/autoencoder.pth', weights_only=True))

apply_autoencoder(model, original_image, noisy_image)