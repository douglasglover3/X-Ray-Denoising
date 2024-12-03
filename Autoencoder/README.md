
## Autoencoder


### Simple setup

Use pip install -r requirements.txt to install all necessary libraries.

Use the following command to run test images in autoencoder model.
This will add noise to the xray images and the autoencoder will attempt to denoise them.
This will output PSNR and SSIM analytics, as well as a test image from each batch for examination.

`python Autoencoder/TrainTest.py --load_model`

### Additional details

Use pip install -r requirements.txt to install all necessary libraries.

Model will be saved after every epoch in ./Autoencoder/outputs/train. 

`python Autoencoder/TrainTest.py --load_epoch n`

This will load the saved model 'epoch_n.pth' and resume training from there.

The final model will be saved as autoencoder.pth. To load autoencoder.pth, use this command.

`python Autoencoder/TrainTest.py --load_model`

This will skip training and begin testing immediately. Testing will output a PSNR and SSIM value, as well as an image from each batch.

