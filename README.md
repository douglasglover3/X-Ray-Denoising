
## Setup

Use pip install -r requirements.txt to install all necessary libraries.

Place dataset images in /data/images

Model will be saved after every epoch in ./outputs. 

`python TrainTest.py --load_epoch n`

This will load the saved 'epoch_n.pth' and resume training from there.

The final model will be saved as model.pth. To load model.pth, use this command.

`python TrainTest.py --load_model`

This will skip training and begin testing immediately.

