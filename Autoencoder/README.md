
## Autoencoder


### Simple demo

Use pip install -r requirements.txt to install all necessary libraries.

Use the following command to apply the autoencoder onto an image at the defined image path. It will automatically add noise to the image and produce an output image.

`python Autoencoder/ApplyToImage.py --image <image-path>`

Results are saved as Autoencoder/result.png.

### Training and other details

Use pip install -r requirements.txt to install all necessary libraries.

Model will be saved after every epoch in ./Autoencoder/outputs/train. 

`python Autoencoder/TrainTest.py --load_epoch n`

This will load the saved model 'epoch_n.pth' and resume training from there.

The final model will be saved as autoencoder.pth. To load autoencoder.pth, use this command.

`python Autoencoder/TrainTest.py --load_model`

This will skip training and begin testing immediately. Testing will output a PSNR and SSIM value, as well as an image from each batch.

