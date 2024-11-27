# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from dataset import XRayDataset
from ridnet import RIDNet
import os

# Hyperparameters
batch_size = 32
epochs = 50
learning_rate = 1e-3

# Prepare dataset and data loaders
train_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = XRayDataset(data_dir="data", split="train")
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
torch.save(model.state_dict(), "ridnet.pth")
print("Model saved to 'ridnet.pth'")

# Testing and visualization
print("Starting testing...")
test_dataset = XRayDataset(data_dir="data", split="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model.eval()

to_pil = ToPILImage()
for idx, (noisy_img, clean_img) in enumerate(test_loader):
    noisy_img = noisy_img.to(device)

    with torch.no_grad():
        denoised_img = model(noisy_img).squeeze(0).cpu()

    # Convert images to PIL for visualization
    noisy_img, denoised_img, clean_img = map(to_pil, [
        noisy_img.squeeze().cpu(), denoised_img, clean_img.squeeze()
    ])

    # Show images
    noisy_img.show(title="Noisy Image")
    denoised_img.show(title="Denoised Image")
    clean_img.show(title="Clean Image")

    # Break after showing 5 images for efficiency
    if idx == 4:
        break
