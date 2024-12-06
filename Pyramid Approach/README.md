## Pyramid Denoising Approach

### Simple demo

Clone the git repository into X-Ray-Denoising if you have not already.

Use pip install -r requirements.txt to install all necessary libraries.

Run the following command to execute the code (assumes that you are in the Pyramid Approach folder).
`python main.py`

It is not necessary to have the dataset downloaded and extracted, as the code will do this for you.

The resulting images from adding Gaussian noise will be created in X-Ray-Denoising\Pyramid Approach\Noisy Images.

The resulting images from denoising the images with added noise will be created in X-Ray-Denoising\Pyramid Approach\outputs.

The code will output a PSNR and SSIM, as well as the time taken to run the Gaussian-Laplacian pyramid denoising algorithm.