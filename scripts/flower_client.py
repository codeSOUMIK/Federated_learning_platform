"""
Flower Federated Learning Client Implementation
This script implements a Flower client for federated learning.
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from typing import Dict, List, Tuple
import argparse
import logging

print("Flower Client script started!") # Added for debugging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """Simple CNN model for demonstration"""
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_model_parameters(model):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model, parameters):
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def train_model(model, train_loader, epochs: int = 1, learning_rate: float = 0.01):
    """Train the model on local data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(data)
            total_samples += len(data)
    
    avg_loss = total_loss / total_samples
    logger.info(f"Training completed. Average loss: {avg_loss:.4f}")
    
    return avg_loss

def evaluate_model(model, test_loader):
    """Evaluate the model on test data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            total_loss += criterion(output, target).item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)
    
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    
    logger.info(f"Evaluation completed. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy

class FlowerClient(fl.client.NumPyClient):
    """Flower client implementation."""
    
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model = SimpleModel()
        
        # Define transformations for your custom dataset
        # These transformations should match what your model expects.
        # Example for image data (adjust as needed for your specific dataset):
        transform = transforms.Compose([
            transforms.Resize((32, 32)), # Resize images to 32x32, matching SimpleModel input
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
        ])
        
        # Load your custom dataset
        # Assuming your dataset is an image folder where subfolders are classes
        # You will need to pass the path to your dataset root directory via --data-path
        try:
            trainset = datasets.ImageFolder(root=self.data_path + "/train", transform=transform)
            testset = datasets.ImageFolder(root=self.data_path + "/test", transform=transform)
        except Exception as e:
            logger.error(f"Error loading dataset from {self.data_path}: {e}")
            raise RuntimeError(f"Could not load dataset. Make sure your data is in {self.data_path}/train and {self.data_path}/test and is compatible with ImageFolder. Error: {e}")

        # Create data loaders
        self.train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(testset, batch_size=32, shuffle=False)
        
        logger.info(f"Client {client_id} initialized with {len(self.train_loader.dataset)} training samples and {len(self.test_loader.dataset)} test samples from {self.data_path}")
    
    def get_parameters(self, config):
        """Return current model parameters."""
        return get_model_parameters(self.model)
    
    def fit(self, parameters, config):
        """Train the model with given parameters."""
        logger.info(f"Client {self.client_id}: Starting training round")
        
        # Update model with received parameters
        set_model_parameters(self.model, parameters)
        
        # Train the model
        epochs = config.get("epochs", 1)
        learning_rate = config.get("learning_rate", 0.01)
        
        loss = train_model(self.model, self.train_loader, epochs, learning_rate)
        
        # Return updated parameters and metrics
        return get_model_parameters(self.model), len(self.train_loader.dataset), {"loss": loss}
    
    def evaluate(self, parameters, config):
        """Evaluate the model with given parameters."""
        logger.info(f"Client {self.client_id}: Starting evaluation")
        
        # Update model with received parameters
        set_model_parameters(self.model, parameters)
        
        # Evaluate the model
        loss, accuracy = evaluate_model(self.model, self.test_loader)
        
        # Return evaluation results
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

def start_flower_client(
    server_address: str = "localhost:8080",
    client_id: int = 1,
    data_path: str = "./data" # Default data path
):
    """Start a Flower client."""
    
    logger.info(f"Starting Flower client {client_id}")
    logger.info(f"Connecting to server at {server_address}")
    logger.info(f"Using data from: {data_path}")
    
    # Create client
    client = FlowerClient(client_id=client_id, data_path=data_path)
    
    # Start client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--server", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--client-id", type=int, default=1, help="Client ID")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to the client's local dataset")
    
    args = parser.parse_args()
    
    print("🌸 Flower Federated Learning Client")
    print("=" * 50)
    print(f"Client ID: {args.client_id}")
    print(f"Server: {args.server}")
    print(f"Data Path: {args.data_path}")
    print("=" * 50)
    
    start_flower_client(
        server_address=args.server,
        client_id=args.client_id,
        data_path=args.data_path
    )
