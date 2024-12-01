import torch
import torch.nn as nn


# Define the RIDNet class, which is a type of neural network architecture
class RIDNet(nn.Module):
    def __init__(self):
        # Initialize the parent class (nn.Module)
        super(RIDNet, self).__init__()

        # The "head" layer: A convolutional layer that maps input to 64 channels
        # Input: Single-channel image (grayscale)
        # Output: 64 feature maps of the same spatial size (3x3 kernel, padding=1 maintains spatial dimensions)
        self.head = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True)

        # The "body" consists of a series of residual dense blocks (RDBs)
        # Each RDB has multiple convolutional and activation layers
        # Input: Feature maps from the head
        # Output: Processed feature maps after three residual dense blocks
        self.body = nn.Sequential(
            *[self.residual_dense_block(64) for _ in range(3)]
        )

        # The "tail" layer: A convolutional layer that maps 64 channels back to 1
        # Input: Processed feature maps from the body
        # Output: Single-channel output image of the same spatial size
        self.tail = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)

    # Define a residual dense block (RDB)
    def residual_dense_block(self, channels):
        # Create a sequence of convolutional and activation layers
        layers = []
        for i in range(4):  # Each RDB consists of 4 convolutional layers
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True))  # 3x3 convolution
            layers.append(nn.ReLU(inplace=True))  # ReLU activation with in-place computation
        return nn.Sequential(*layers)  # Combine layers into a single module

    # Define the forward pass
    def forward(self, x):
        # Pass the input through the head layer to extract initial features
        out = self.head(x)

        # Save the input features as a residual connection
        residual = out

        # Pass the features through the body (stack of RDBs)
        out = self.body(out)

        # Pass the processed features through the tail layer to reconstruct the output
        out = self.tail(out)

        # Add the residual connection to the output (skip connection)
        # This helps in learning the residual (difference) rather than the full transformation
        return self.tail(residual) + out
