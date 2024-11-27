from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

# Define the XRayDataset class to create the dataset
class XRayDataset(Dataset):
    def __init__(self, data_dir, split):
        """
                Initialize the dataset by loading file paths for training, validation, or testing.

                Args:
                    data_dir (str): Path to the root data directory containing the 'final' folder.
                    split (str): Dataset split - 'train', 'val', or 'test'.
                """
        self.data_dir = data_dir
        self.split = split
        self.clean_dir = os.path.join(data_dir, 'final', 'train')  # Clean images directory
        self.noisy_dir = os.path.join(data_dir, 'final', split)  # Noisy images directory for the specified split

        # Load clean and noisy image file paths
        self.clean_files = glob.glob(os.path.join(self.clean_dir, '*.*'))
        self.noisy_files = glob.glob(os.path.join(self.noisy_dir, '*.*'))

        # Ensure consistent pairing by file name
        self.clean_files = sorted(self.clean_files, key=lambda x: os.path.basename(x))
        self.noisy_files = sorted(self.noisy_files, key=lambda x: os.path.basename(x))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        """Return the number of paired clean and noisy images."""
        return min(len(self.clean_files), len(self.noisy_files))

    def __getitem__(self, idx):
        """
        Get a pair of noisy and clean images.

        Args:
            idx (int): Index of the image to fetch.

        Returns:
            Tuple[Tensor, Tensor]: Transformed noisy and clean images.
        """
        # Retrieve file paths for clean and noisy images
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_files[idx]

        # Load images
        clean_img = Image.open(clean_path)
        noisy_img = Image.open(noisy_path)

        # Transform images and return the pair
        return self.transform(noisy_img), self.transform(clean_img)


