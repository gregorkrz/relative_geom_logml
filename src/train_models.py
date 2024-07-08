import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
import numpy as np
import random

if __name__ == '__main__':
    # Set the seed for reproducibility
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # Define the transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Initialize ResNet18 model
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)  # CIFAR-10 has 10 classes

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Function to save model weights and latent vectors
    def save_checkpoint(epoch, model, latent_vectors, checkpoint_dir='checkpoints'):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Save model weights
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch}_{seed}.pth'))
        
        # Save latent vectors
        np.save(os.path.join(checkpoint_dir, f'latent_vectors_epoch_{epoch}_{seed}.npy'), latent_vectors)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}')

        # Validation step to get latent vectors
        model.eval()
        latent_vectors = []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                latent_features = outputs[:, :model.fc.in_features]  # Get the size of the last layer before classification
                latent_vectors.append(latent_features.cpu().numpy())
        
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        save_checkpoint(epoch, model, latent_vectors)

    print('Training complete')
