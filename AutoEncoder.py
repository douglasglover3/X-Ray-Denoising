import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Defined layers for convolutional encoder
        self.encoder_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # Defined layers for convolutional decoder
        self.decoder_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.decoder_conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.decoder_conv3 = nn.Conv2d(32, 3, 3, padding=1)
        
        self.forward = self.autoencoder
    
    #Convolutional Autoencoder
    def autoencoder(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X
    
    def encoder(self, X):
        X = F.max_pool2d(F.relu(self.encoder_conv1(X)), (2,2))
        X = F.max_pool2d(F.relu(self.encoder_conv2(X)), 2)
        return X
    
    def decoder(self, X):
        X = F.interpolate(F.relu(self.decoder_conv1(X)), scale_factor=2)
        X = F.interpolate(F.relu(self.decoder_conv2(X)), scale_factor=2)
        X = self.decoder_conv3(X)
        return X
    
    
