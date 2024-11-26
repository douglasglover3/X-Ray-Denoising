import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Defined layers for convolutional encoder
        self.encoder_conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.encoder_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.encoder_conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.encoder_conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.encoder_conv5 = nn.Conv2d(32, 64, 3, padding=1)

        # Defined layers for convolutional decoder
        self.decoder_conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.decoder_conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.decoder_conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.decoder_conv4 = nn.Conv2d(16, 8, 3, padding=1)
        self.decoder_conv5 = nn.Conv2d(8, 3, 3, padding=1)
        
        self.forward = self.autoencoder2
    
    #Convolutional Autoencoder
    def autoencoder1(self, X):
        X = self.encoder1(X)
        X = self.decoder1(X)
        return X
    
    def autoencoder2(self, X): #One less convolutional layer than autoencoder1
        X = self.encoder2(X)
        X = self.decoder2(X)
        return X
    
    def autoencoder3(self, X): #Two less convolutional layers than autoencoder1
        X = self.encoder3(X)
        X = self.decoder3(X)
        return X
    
    def encoder1(self, X):
        X = F.relu(self.encoder_conv1(X))
        X = F.relu(self.encoder_conv2(X))
        X = F.relu(self.encoder_conv3(X))
        X = F.relu(self.encoder_conv4(X))
        return X
    
    def decoder1(self, X):
        X = F.relu(self.decoder_conv1(X))
        X = F.relu(self.decoder_conv2(X))
        X = F.relu(self.decoder_conv3(X))
        X = F.relu(self.decoder_conv4(X))
        X = self.decoder_conv5(X)
        return X
    
    def encoder2(self, X):
        X = F.relu(self.encoder_conv1(X))
        X = F.relu(self.encoder_conv2(X))
        X = F.relu(self.encoder_conv3(X))
        return X
    
    def decoder2(self, X):
        X = F.relu(self.decoder_conv2(X))
        X = F.relu(self.decoder_conv3(X))
        X = F.relu(self.decoder_conv4(X))
        X = self.decoder_conv5(X)
        return X
    
    def encoder3(self, X):
        X = F.relu(self.encoder_conv1(X))
        X = F.relu(self.encoder_conv2(X))
        return X
    
    def decoder3(self, X):
        X = F.relu(self.decoder_conv3(X))
        X = F.relu(self.decoder_conv4(X))
        X = self.decoder_conv5(X)
        return X
    
