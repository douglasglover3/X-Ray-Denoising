# Import necessary libraries
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from dataset import XRayDataset
from ridnet import RIDNet


# Hyperparameters
# batch_size = 32
# epochs = 50
# learning_rate = 1e-3

# Testing Hyperparameters
batch_size = 8
epochs = 1
learning_rate = 5e-4

# Prepare dataset and data loaders
train_transform = transforms.Compose([
    transforms.ToTensor()
])

# testing datasets
train_dataset = XRayDataset(data_dir="data", split="train")
test_dataset = XRayDataset(data_dir="data", split="test")

# train_dataset = XRayDataset(data_dir="data", split="train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RIDNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Starting training...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        for noisy_img, clean_img in tepoch:
            noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)

            optimizer.zero_grad()
            denoised_img = model(noisy_img)
            loss = criterion(denoised_img, clean_img)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "../ridnet.pth")
print("Model saved to 'ridnet.pth'")


def calculate_psnr(denoised_img, clean_img):
    mse = torch.mean((denoised_img - clean_img) ** 2)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))  # Assuming image range is [0, 1]


print("Starting testing...")
test_dataset = XRayDataset(data_dir="data", split="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model.eval()

to_pil = ToPILImage()
total_psnr = 0
num_samples = 0

for idx, (noisy_img, clean_img) in enumerate(test_loader):
    noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)

    with torch.no_grad():
        denoised_img = model(noisy_img).squeeze(0).cpu()

    # Calculate PSNR
    psnr = calculate_psnr(denoised_img, clean_img.squeeze(0).cpu())
    total_psnr += psnr
    num_samples += 1

    # Ensure denoised_img has 1 channel before passing to ToPILImage
    denoised_img = denoised_img.squeeze(0)

    # Convert images to PIL for visualization
    noisy_img = noisy_img.squeeze(0).squeeze(0).cpu()  # Squeeze to remove batch and channel dimensions
    clean_img = clean_img.squeeze(0).squeeze(0).cpu()  # Squeeze to remove batch and channel dimensions

    noisy_img, denoised_img, clean_img = map(to_pil, [noisy_img, denoised_img, clean_img])

    # Show images
    noisy_img.show(title="Noisy Image")
    denoised_img.show(title="Denoised Image")
    clean_img.show(title="Clean Image")

    # Break after showing 5 images for efficiency
    if idx == 4:
        break

average_psnr = total_psnr / num_samples
print(f"Average PSNR (accuracy): {average_psnr:.2f} dB")
print("Training and evaluation finished")
