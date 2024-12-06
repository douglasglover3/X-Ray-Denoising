import torch
import numpy as np

#Transform for applying gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        gaussian = np.random.normal(self.mean, self.std, tensor.shape)
        return np.clip(tensor + gaussian, -1, 1, dtype=np.float32)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)