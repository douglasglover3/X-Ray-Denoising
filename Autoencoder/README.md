
## Autoencoder

Use pip install -r requirements.txt to install all necessary libraries.

Place dataset images in /data/images

Model will be saved after every epoch in ./Autoencoder/outputs/train. 

`python Autoencoder/TrainTest.py --load_epoch n`

This will load the saved 'epoch_n.pth' and resume training from there.

The final model will be saved as autoencoder.pth. To load autoencoder.pth, use this command.

`python Autoencoder/TrainTest.py --load_model`

This will skip training and begin testing immediately.

