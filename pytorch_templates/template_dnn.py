# ==================================================================
# Deep Neural Network Model
# ==================================================================
# A simple neural network model that uses the MNIST dataset

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Dataset loading
def load_data(batch_size=64):
    # Transform to convert images to tensor
    transform = transforms.ToTensor()

    # Load MNIST dataset
    # The train_data is of the form (Tensor, int) where the first element is the image
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform)

    # Print some information about the dataset
    # print(f"Train data example shape: {train_data[0][0].shape}")
    # print(f"Train data label: {train_data[0][1]}")
    # print(f"Train data example type: {type(train_data[0][0])}")
    # print(f"Train data label type: {type(train_data[0][1])}")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Model definition


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            # Input image size is 28x28, so the first layer will flatten it to 784
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # Output layer with 10 classes (digits 0-9)
        )

    def forward(self, x):
        return self.model(x)

# Training loop


def train(model, dataloader, loss_fn, optimizer):
    # Set the model to training mode
    model.train()

    # Iterate through the training data
    for X, y in dataloader:
        # Move data to the device (GPU or CPU)
        # The shape of X is (batch_size, 1, 28, 28) and y is (batch_size)
        X, y = X.to(device), y.to(device)

        # Make a prediction based on the current state of the model
        # The shape of pred is (batch_size, N_output_classes)
        pred = model(X)

        # Calculate the loss based on that prediction and the labels
        loss = loss_fn(pred, y)

        # Calculates the gradients of the loss with respect to the model parameters
        # Stores the gradient in the .grad attribute of the model objects
        loss.backward()

        # Zero the gradients before the backward pass
        optimizer.zero_grad()

        # Update the model parameters
        optimizer.step()

# Evaluation loop


def test(model, dataloader, loss_fn):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track total and correct predictions
    total, correct, loss_total = 0, 0, 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate through the test data
        for X, y in dataloader:
            # Move data to the device (GPU or CPU)
            X, y = X.to(device), y.to(device)

            # Forward pass
            pred = model(X)

            # Calculate the loss
            loss_total += loss_fn(pred, y).item()

            # Calculate the number of correct predictions
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Update the total number of samples
            total += y.size(0)

    # Calculate accuracy and average loss
    accuracy = correct / total
    print(
        f"Test Accuracy: {accuracy:.2%}, Avg Loss: {loss_total/len(dataloader):.4f}")

# Main execution


def main():
    # Load the data
    train_loader, test_loader = load_data()

    # Initialize the model & move it to the device
    model = NeuralNetwork().to(device)

    # Using CrossEntropyLoss for multi-class classification
    loss_fn = nn.CrossEntropyLoss()

    # Using Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training and testing the model
    for epoch in range(2):
        print(f"Epoch {epoch+1}")
        # Train the model
        train(model, train_loader, loss_fn, optimizer)
        # Test the model
        test(model, test_loader, loss_fn)


if __name__ == "__main__":
    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    main()
