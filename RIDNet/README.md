# RIDNet Project

This project implements the Residual Image Denoising Network (RIDNet) for image denoising tasks. The dataset used includes noisy and clean images for training, validation, and testing. The code is modular, with scripts for preprocessing, training, and evaluation.

## Requirements
### Libraries
The project depends on the following Python libraries:

- google-cloud-storage
- numpy
- opencv-python
- scikit-learn
- scikit-image
- torch
- torchvision
- matplotlib
- Pillow
- tqdm

Install all dependencies using:

```commandline
pip install -r requirements.txt
```

---

## Dataset
The dataset is available at the following link:

[NIH Chest X-Ray Dataset](https://console.cloud.google.com/storage/browser/nih-chest-xray-project;tab=objects?forceOnBucketsSortingFiltering=true&project=cap5415-442520&prefix=&forceOnObjectsSortingFiltering=false)

Ensure the dataset is organized as follows:

```commandline
data/
├── raw/               # Original noisy and clean images
├── preprocessed/      # Preprocessed data (after normalization, cropping, etc.)
├── noisy/             # Noisy images
└── final/
    ├── train/         # Training images
    ├── val/           # Validation images
    └── test/          # Testing images
```

## Project Structure
The repository structure is as follows:

```commandline
RIDNet/
├── data/                # Dataset directory
├── scripts/             # Contains all Python scripts
│   ├── __init__.py      # Initializes the scripts module
│   ├── access_data.py   # Fetches dataset from cloud storage
│   ├── add_noise.py     # Adds noise to clean images
│   ├── dataset.py       # Creates PyTorch Dataset objects
│   ├── preprocess.py    # Preprocesses raw data
│   ├── ridnet.py        # Defines the RIDNet architecture
│   ├── split_data.py    # Splits data into train/val/test sets
│   └── train_evaluate_RIDNet.py # Script for training and evaluation
├── venv/                # Virtual environment (optional)
├── main.py              # Entry point to the project
├── requirements.txt     # Lists project dependencies
├── ridnet.pth           # Pre-trained model weights (if available)
└── .gitignore           # Ignore unnecessary files
```

## How to Run the Project
Follow these steps to run the project:

Step 1: Set Up the Environment

1. Clone this repository:
   ```commandline
    git clone https://github.com/douglasglover3/X-Ray-Denoising/tree/af/feature/dataPrep/RIDNet
    cd RIDNet
    ```
2. Create and activate a virtual environment (optional but recommended):
    ```
   python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   
3. Install dependencies:
    ```
   pip install -r requirements.txt
   ```
   
Step 2: Prepare the Dataset

1. Download the dataset from the link provided above.
2. Place the downloaded files in the data/raw folder.
3. Run the preprocess.py script to preprocess the dataset:
    ```commandline
    python scripts/preprocess.py
    ```
  
4. Split the data into training, validation, and testing sets using split_data.py: 
     ```
     python scripts/split_data.py
     ```
   
Step 3: Train and Evaluate the Model

1. Train the RIDNet model using the training script:
    ```commandline
    python scripts/train_evaluate_RIDNet.py
    ```

2. During training, the script logs performance metrics such as training and validation loss.

Step 4: Test the Model
1. Evaluate the model on the test set using:

    ```commandline
    python scripts/train_evaluate_RIDNet.py --test
    ```
   
Step 5: Use Pre-Trained Weights (Optional)

If you have pre-trained weights (ridnet.pth), you can load them for evaluation or further fine-tuning. Ensure the weights file is in the project root directory.

Step 6: Access Results

The denoised outputs and performance metrics (e.g., PSNR, SSIM) will be saved in the appropriate subdirectories within the data/ folder.

## Notes

The script add_noise.py is optional for augmenting the dataset by adding synthetic noise to clean images.
The project uses PyTorch for defining and training the RIDNet architecture (ridnet.py).

## Contact

For questions or contributions, feel free to contact Alejandro Fuste.


