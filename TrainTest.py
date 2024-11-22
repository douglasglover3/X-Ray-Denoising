from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from AutoEncoder import AutoEncoder 
import argparse
import numpy as np 

def train(autoencoder: AutoEncoder, device, train_dataset, optimizer, criterion, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    device: 'cuda' or 'cpu'.
    train_dataset: dataset of training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    batch_size: Batch size to be used.
    '''
    
    # Set autoencoder to train mode before each epoch
    autoencoder.train()
    
    # Empty list to store losses 
    losses = []
    
    num_batches = int(np.ceil(len(train_dataset) / batch_size))
    images = []
    for batch_index in range(num_batches):
        data_array = [train_dataset[i][0].numpy() for i in range(batch_index * batch_size, (batch_index+1) * batch_size)]

        data = torch.tensor(np.array(data_array))
        data = data.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output: torch.Tensor = autoencoder(data)
        loss = criterion(output, data)
        # Computes gradient based on final loss
        loss.backward()
        # Store loss
        losses.append(loss.item())
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()

        percent = 100 * (batch_index / num_batches)
        bar = '█' * int(percent) + '-' * (100 - int(percent))
        print(f' Batch {batch_index} / {num_batches} |{bar}| {percent:.2f}%', end = "\r")
        
    #Save one image from this epoch
    images.append(('Image', data[0].permute(1, 2, 0)))
    images.append(('Decoded Image', output[0].permute(1, 2, 0).detach().numpy()))       
        
    train_loss = float(np.mean(losses))
    print('\nTrain set: Average loss: {:.4f}\n'.format(float(np.mean(losses))))
    
    return train_loss, images
    

def test(autoencoder: AutoEncoder, device, test_dataset, batch_size):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_dataset: dataset of test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    autoencoder.eval()
    
    images = []
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        num_batches = int(np.ceil(len(test_dataset) / batch_size))
        
        for batch_index in range(0, num_batches):
            data_array = [test_dataset[i][0].numpy() for i in range(batch_index * batch_size, (batch_index+1) * batch_size)]

            data = torch.tensor(np.array(data_array))
            data = data.to(device)
            
            # Predict for data by doing forward pass
            output: torch.Tensor = autoencoder(data)
            
            images.append(('Image ' + str(batch_index), data[0].permute(1, 2, 0)))
            images.append(('Decoded Image ' + str(batch_index), output[0].permute(1, 2, 0).detach().numpy()))

            percent = 100 * (batch_index / num_batches)
            bar = '█' * int(percent) + '-' * (100 - int(percent))
            print(f' Batch {batch_index} / {num_batches} |{bar}| {percent:.2f}%', end = "\r")

    #Save one image from each batch
    fig, axes = plt.subplots(1, 2)
    for index, (title, image) in enumerate(images):
        im = axes[index % 2]
        im.imshow(image)
        im.set_title(title)
        im.axis('off')
        if index % 2 == 1:
            plt.tight_layout()
            plt.savefig(f'./outputs/test/result_{int(index / 2)}.png')

    
    return
    
def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = AutoEncoder().to(device)
    
    #Use MSE loss function
    criterion = nn.MSELoss()

    #Use SGD optimizer, with the learning rate defined in arguments
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
        
    
    # Load datasets for training and testing
    path ='./data'
    tensor_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(path, transform=tensor_transform)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    #create folders for output images if they dont exist
    if not os.path.exists("./outputs/train"): 
        os.makedirs("./outputs/train") 
    if not os.path.exists("./outputs/test"): 
        os.makedirs("./outputs/test") 

    # Run training for n_epochs specified in config 
    train_loss_array = []

    for epoch in range(1, FLAGS.num_epochs + 1):
        print("\nEpoch: " + str(epoch) + "\n")
        train_loss, images = train(model, device, train_dataset, optimizer, criterion, FLAGS.batch_size)
        train_loss_array.append(train_loss)

        
        #Save image of epoch
        fig, axes = plt.subplots(1, 2)

        #Input
        im = axes[0]
        im.set_title(images[0][0])
        im.imshow(images[0][1])
        im.axis('off')

        #Output
        im = axes[1]
        im.set_title(images[1][0])
        im.imshow(images[1][1])
        im.axis('off')

        plt.tight_layout()
        plt.savefig(f'./outputs/train/epoch_{epoch}.png')
        
    
    
    print("Training finished")
    print("Testing...")
    test(model, device, test_dataset, FLAGS.batch_size)
    print("\nTesting finished")
    
    
    
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    