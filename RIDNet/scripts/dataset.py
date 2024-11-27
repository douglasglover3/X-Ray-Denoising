from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

# Define the XRayDataset class to create the dataset
class XRayDataset(Dataset):
    def __init__(self, data_dir, split):