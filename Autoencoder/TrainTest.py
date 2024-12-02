from __future__ import print_function
import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from AutoEncoder import AutoEncoder 
from AddGaussianNoise import AddGaussianNoise
import argparse
import numpy as np 
import datetime

def train(autoencoder: AutoEncoder, device, train_dataset, train_noisy_dataset, optimizer, criterion, batch_size):
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
    
    num_batches = int(np.floor(len(train_dataset) / batch_size))
    images = []
    batch_times = []
    eta = datetime.timedelta(seconds=0)
    for batch_index in range(num_batches):
        start = datetime.datetime.now()
        data_array = [train_dataset[i][0].numpy() for i in range(batch_index * batch_size, (batch_index+1) * batch_size)]
        noisy_array = [train_noisy_dataset[i][0].numpy() for i in range(batch_index * batch_size, (batch_index+1) * batch_size)]

        data = torch.tensor(np.array(data_array))
        data = data.to(device)

        noisy = torch.tensor(np.array(noisy_array))
        noisy = noisy.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output: torch.Tensor = autoencoder(noisy)
        loss = criterion(output, data)
        # Computes gradient based on final loss
        loss.backward()
        # Store loss
        losses.append(loss.item())
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()

        # Update progress bar
        percent = 100 * ((batch_index + 1) / num_batches)
        bar = '█' * int(percent) + '-' * (100 - int(percent))
        print(f'\tBatch {batch_index + 1} / {num_batches} |{bar}| {percent:.2f}% \tETA: {math.floor(eta.seconds / 360)} hours, {math.floor(eta.seconds / 60) % 60} minutes, {eta.seconds % 60} seconds ', end = "\r")
        finish = datetime.datetime.now()
        batch_times.append(finish - start)
        if len(batch_times) > 10:
                batch_times.pop(0)
        eta = (sum(batch_times, datetime.timedelta(0)) / len(batch_times)) * (num_batches - batch_index)
    print('\n')

    data = data.to('cpu')
    noisy = noisy.to('cpu')
    output = output.to('cpu')
        
    #Save one image from this epoch
    images.append(('Original Image', data[0].permute(1, 2, 0)))
    images.append(('Noisy Image', noisy[0].permute(1, 2, 0)))
    images.append(('Decoded Image', output[0].permute(1, 2, 0).detach().numpy()))       
        
    train_loss = float(np.mean(losses))
    print('Train set: Average loss: {:.4f}\n'.format(float(np.mean(losses))))
    
    return train_loss, images
    
def test(autoencoder: AutoEncoder, device, test_dataset, test_noisy_dataset, batch_size):
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
        num_batches = int(np.floor(len(test_dataset) / batch_size))

        batch_times = []
        eta = datetime.timedelta(seconds=0)
        for batch_index in range(0, num_batches):
            start = datetime.datetime.now()
            data_array = [test_dataset[i][0].numpy() for i in range(batch_index * batch_size, (batch_index+1) * batch_size)]
            noisy_array = [test_noisy_dataset[i][0].numpy() for i in range(batch_index * batch_size, (batch_index+1) * batch_size)]

            data = torch.tensor(np.array(data_array))

            noisy = torch.tensor(np.array(noisy_array))
            noisy = noisy.to(device)

            # Predict for data by doing forward pass
            output: torch.Tensor = autoencoder(noisy)

            data = data.to('cpu')
            noisy = noisy.to('cpu')
            output = output.to('cpu')
            
            images.append(('Original Image ' + str(batch_index), data[0].permute(1, 2, 0)))
            images.append(('Noisy Image ' + str(batch_index), noisy[0].permute(1, 2, 0)))
            images.append(('Decoded Image ' + str(batch_index), output[0].permute(1, 2, 0).detach().numpy()))

            # Update progress bar
            percent = 100 * ((batch_index + 1) / num_batches)
            bar = '█' * int(percent) + '-' * (100 - int(percent))
            print(f'\tBatch {batch_index + 1} / {num_batches} |{bar}| {percent:.2f}% \tETA: {math.floor(eta.seconds / 360)} hours, {math.floor(eta.seconds / 60) % 60} minutes, {eta.seconds % 60} seconds', end = "\r")
            finish = datetime.datetime.now()
            batch_times.append(finish - start)
            if len(batch_times) > 10:
                batch_times.pop(0)
            eta = (sum(batch_times, datetime.timedelta(0)) / len(batch_times)) * (num_batches - batch_index)
        print('\n')
            
    print('Saving testing images...')
    #Save one image from each batch
    fig, axes = plt.subplots(1, 3)
    for index, (title, image) in enumerate(images):
        im = axes[index % 3]
        im.imshow(image, cmap='gray')
        im.set_title(title)
        im.axis('off')
        if index % 3 == 2:
            plt.tight_layout()
            plt.savefig(f'./Autoencoder/outputs/test/result_{int(index / 3) + 1}.png', dpi=300)

    
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
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    noisy_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        AddGaussianNoise(0., 0.1)
    ])


    dataset = datasets.ImageFolder(path, transform=tensor_transform)
    noisy_dataset = datasets.ImageFolder(path, transform=noisy_transform)

    generator1 = torch.Generator().manual_seed(123)
    generator2 = torch.Generator().manual_seed(123)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator1)
    train_noisy_dataset, test_noisy_dataset = torch.utils.data.random_split(noisy_dataset, [0.9, 0.1], generator2)

    #create folders for output images if they dont exist
    if not os.path.exists("./Autoencoder/outputs/train"): 
        os.makedirs("./Autoencoder/outputs/train") 
    if not os.path.exists("./Autoencoder/outputs/test"): 
        os.makedirs("./Autoencoder/outputs/test") 

    starting_epoch = 1
    if FLAGS.load_epoch != None:
        starting_epoch = FLAGS.load_epoch + 1
        model.load_state_dict(torch.load(f'./Autoencoder/outputs/train/epoch_{FLAGS.load_epoch}.pth', weights_only=True))

    # Run training for n_epochs specified in config 
    train_loss_array = []
    if FLAGS.load_model == False:
        for epoch in range(starting_epoch, FLAGS.num_epochs + 1):
            print("\nEpoch: " + str(epoch) + "\n")
            train_loss, images = train(model, device, train_dataset, train_noisy_dataset, optimizer, criterion, FLAGS.batch_size)
            train_loss_array.append(train_loss)
            torch.save(model.state_dict(), f'./Autoencoder/outputs/train/epoch_{epoch}.pth')

            
            #Save image of epoch
            fig, axes = plt.subplots(1, 3)
            
            #Input
            im = axes[0]
            im.set_title(images[0][0])
            im.imshow(images[0][1], cmap='gray')
            im.axis('off')

            #Noisy
            im = axes[1]
            im.set_title(images[1][0])
            im.imshow(images[1][1], cmap='gray')
            im.axis('off')

            #Output
            im = axes[2]
            im.set_title(images[2][0])
            im.imshow(images[2][1], cmap='gray')
            im.axis('off')

            plt.tight_layout()
            plt.savefig(f'./Autoencoder/outputs/train/epoch_{epoch}.png', dpi=1200)

        torch.save(model.state_dict(), './Autoencoder/autoencoder.pth') 
    else:
        if os.path.exists("./Autoencoder/autoencoder.pth"): 
            model.load_state_dict(torch.load('./Autoencoder/autoencoder.pth', weights_only=True))
        else:     
            print("Could not load model.")
            exit(-1)
    
    print("Training finished")
    print("Testing...")
    test(model, device, test_dataset, test_noisy_dataset, FLAGS.batch_size)
    print("Testing finished")
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--load_model',
                        action=argparse.BooleanOptionalAction, default=False, 
                        help='Adding this flag will load the model from file.')
    parser.add_argument('--load_epoch',
                        type=int, default=None,
                        help='Epoch to load and to start training from.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
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
    
    