from sklearn.model_selection import train_test_split
import shutil
import os
from glob import glob

"""
    Splits the preprocessed folder into training data and noisy folder into validation and testing data.
    
"""


def split_dataset(preprocessed_folder, noisy_folder, output_base_folder, train_size=0.7, val_size=0.5, test_size=0.5):
    # Ensure the sum of val_size and test_size equals 1.0
    if val_size + test_size != 1.0:
        raise ValueError("Validation and test sizes must sum to 1.0 for noisy data splitting.")

    # Step 1: Split preprocessed data for training
    preprocessed_files = glob(os.path.join(preprocessed_folder, '*.*'))  # Get all preprocessed file paths
    train_files = train_test_split(preprocessed_files, train_size=train_size, random_state=42)[0]  # Only need training

    # Step 2: Split noisy data for validation and testing
    noisy_files = glob(os.path.join(noisy_folder, '*.*'))  # Get all noisy file paths
    val_files, test_files = train_test_split(noisy_files, test_size=test_size, random_state=42)

    # Helper function to copy files to a destination folder
    def copy_files(file_list, dest_folder):
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        for file_path in file_list:
            shutil.copy(file_path, dest_folder)

    # Step 3: Copy files to their respective folders
    copy_files(train_files, os.path.join(output_base_folder, 'train'))  # Training data
    copy_files(val_files, os.path.join(output_base_folder, 'val'))  # Validation data
    copy_files(test_files, os.path.join(output_base_folder, 'test'))  # Testing data

    # Print a summary of the splits
    print(f"Dataset split completed.")
    print(f"Train: {len(train_files)} files")
    print(f"Validation: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")


# split images from preprocessed and noisy folders to train, validation, and test folders
split_dataset(
    preprocessed_folder='../data/preprocessed',  # Preprocessed clean data for training
    noisy_folder='../data/noisy',  # Noisy data for validation and testing
    output_base_folder='../data/final'  # Final output folder for splits
)
