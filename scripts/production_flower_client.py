"""
Production-ready Flower Client with Real Dataset Support
Comprehensive client implementation for federated learning
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import time
import psutil
import platform
from typing import Dict, List, Tuple, Optional, Any
import logging
import argparse
import os
from pathlib import Path

# Import our model classes
from models.base_model import create_model, BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    """Custom dataset class for federated learning"""
    
    def __init__(self, data_path: str, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = []
        self.labels = []
        self.load_data()
    
    def load_data(self):
        """Load data from the specified path"""
        # This is a placeholder - implement based on your data format
        # For now, we'll create synthetic data
        logger.info(f"Loading data from {self.data_path}")
        
        # Generate synthetic data (replace with real data loading)
        num_samples = 1000
        self.samples = torch.randn(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, 10, (num_samples,))
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

class DataManager:
    """Manages data loading and preprocessing for clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_type = config.get('dataset_type', 'cifar10')
        self.data_path = config.get('data_path', './data')
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 2)
        
        # Create data directory
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def load_dataset(self, client_id: int, total_clients: int) -> Tuple[DataLoader, DataLoader]:
        """Load and partition dataset for a specific client"""
        
        if self.dataset_type == 'cifar10':
            return self._load_cifar10(client_id, total_clients)
        elif self.dataset_type == 'custom':
            return self._load_custom_dataset(client_id, total_clients)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
    
    def _load_cifar10(self, client_id: int, total_clients: int) -> Tuple[DataLoader, DataLoader]:
        """Load CIFAR-10 dataset with client-specific partitioning"""
        
        # Download CIFAR-10
        trainset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=True, download=True, transform=self.train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=False, download=True, transform=self.test_transform
        )
        
        # Partition data for federated learning
        train_data, test_data = self._partition_data(trainset, testset, client_id, total_clients)
        
        # Create data loaders
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        logger.info(f"Client {client_id}: {len(train_data)} train, {len(test_data)} test samples")
        
        return train_loader, test_loader
    
    def _load_custom_dataset(self, client_id: int, total_clients: int) -> Tuple[DataLoader, DataLoader]:
        """Load custom dataset"""
        
        dataset = CustomDataset(self.data_path, transform=self.train_transform)
        
        # Split into train and test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size])
        
        # Partition for federated learning
        train_data, test_data = self._partition_data(train_data, test_data, client_id, total_clients)
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def _partition_data(self, train_data, test_data, client_id: int, total_clients: int):
        """Partition data among clients (can implement IID or non-IID)"""
        
        # Simple IID partitioning
        train_size = len(train_data) // total_clients
        test_size = len(test_data) // total_clients
        
        train_start = client_id * train_size
        train_end = train_start + train_size
        test_start = client_id * test_size
        test_end = test_start + test_size
        
        # Handle last client getting remaining data
        if client_id == total_clients - 1:
            train_end = len(train_data)
            test_end = len(test_data)
        
        train_subset = torch.utils.data.Subset(train_data, range(train_start, train_end))
        test_subset = torch.utils.data.Subset(test_data, range(test_start, test_end))
        
        return train_subset, test_subset

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model: BaseModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, train_loader: DataLoader, epochs: int, learning_rate: float) -> Dict[str, float]:
        """Train the model"""
        
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        
        total_loss = 0.0
        total_samples = 0
        correct = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item() * len(data)
                epoch_samples += len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                if batch_idx % 100 == 0:
                    logger.debug(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            logger.info(f'Epoch {epoch+1}/{epochs} completed, Loss: {epoch_loss/epoch_samples:.6f}')
        
        training_time = time.time() - start_time
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'training_time': training_time,
            'data_samples': total_samples
        }
        
        logger.info(f'Training completed: Loss={avg_loss:.6f}, Accuracy={accuracy:.6f}, Time={training_time:.2f}s')
        
        return metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                total_loss += self.criterion(output, target).item() * len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(data)
        
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'data_samples': total_samples
        }
        
        logger.info(f'Evaluation completed: Loss={avg_loss:.6f}, Accuracy={accuracy:.6f}')
        
        return metrics

class ProductionFlowerClient(fl.client.NumPyClient):
    """Production-ready Flower client"""
    
    def __init__(self, client_id: int, config: Dict[str, Any]):
        self.client_id = client_id
        self.config = config
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Client {client_id} using device: {self.device}")
        
        # Create model
        model_type = config.get('model_type', 'cnn')
        model_params = config.get('model_params', {})
        self.model = create_model(model_type, **model_params)
        
        # Initialize trainer
        self.trainer = ModelTrainer(self.model, self.device)
        
        # Initialize data manager
        self.data_manager = DataManager(config)
        
        # Load data
        total_clients = config.get('total_clients', 1)
        self.train_loader, self.test_loader = self.data_manager.load_dataset(client_id, total_clients)
        
        # Client metrics
        self.round_metrics = []
        
        logger.info(f"Client {client_id} initialized successfully")
        logger.info(f"Model: {self.model.model_name} ({self.model.count_parameters()} parameters)")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return current model parameters"""
        return self.model.get_parameters()
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train the model with given parameters"""
        
        server_round = config.get("server_round", 0)
        epochs = config.get("epochs", 1)
        learning_rate = config.get("learning_rate", 0.01)
        
        logger.info(f"Client {self.client_id} starting training round {server_round}")
        
        # Set model parameters
        self.model.set_parameters(parameters)
        
        # Train model
        metrics = self.trainer.train(self.train_loader, epochs, learning_rate)
        
        # Store metrics
        round_metric = {
            'round': server_round,
            'type': 'training',
            'metrics': metrics,
            'timestamp': time.time()
        }
        self.round_metrics.append(round_metric)
        
        # Return updated parameters and metrics
        return (
            self.model.get_parameters(),
            len(self.train_loader.dataset),
            metrics
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model with given parameters"""
        
        server_round = config.get("server_round", 0)
        logger.info(f"Client {self.client_id} starting evaluation round {server_round}")
        
        # Set model parameters
        self.model.set_parameters(parameters)
        
        # Evaluate model
        metrics = self.trainer.evaluate(self.test_loader)
        
        # Store metrics
        round_metric = {
            'round': server_round,
            'type': 'evaluation',
            'metrics': metrics,
            'timestamp': time.time()
        }
        self.round_metrics.append(round_metric)
        
        # Return loss, number of examples, and metrics
        return (
            metrics['loss'],
            len(self.test_loader.dataset),
            {'accuracy': metrics['accuracy']}
        )
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get comprehensive client information"""
        
        # Get hardware information
        hardware_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            hardware_info['gpu_name'] = torch.cuda.get_device_name(0)
            hardware_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
        
        # Get data information
        data_info = {
            'train_samples': len(self.train_loader.dataset),
            'test_samples': len(self.test_loader.dataset),
            'batch_size': self.train_loader.batch_size,
            'dataset_type': self.config.get('dataset_type', 'unknown')
        }
        
        # Get model information
        model_info = self.model.get_model_config()
        
        return {
            'client_id': self.client_id,
            'hardware_info': hardware_info,
            'data_info': data_info,
            'model_info': model_info,
            'capabilities': {
                'gpu': torch.cuda.is_available(),
                'memory': f"{psutil.virtual_memory().total // (1024**3)}GB",
                'cpu_cores': psutil.cpu_count()
            },
            'performance_metrics': self.get_performance_summary()
        }
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary across all rounds"""
        
        if not self.round_metrics:
            return {}
        
        training_metrics = [m for m in self.round_metrics if m['type'] == 'training']
        evaluation_metrics = [m for m in self.round_metrics if m['type'] == 'evaluation']
        
        summary = {
            'total_rounds': len(self.round_metrics),
            'training_rounds': len(training_metrics),
            'evaluation_rounds': len(evaluation_metrics)
        }
        
        if training_metrics:
            train_accuracies = [m['metrics']['accuracy'] for m in training_metrics]
            train_losses = [m['metrics']['loss'] for m in training_metrics]
            train_times = [m['metrics']['training_time'] for m in training_metrics]
            
            summary.update({
                'avg_train_accuracy': np.mean(train_accuracies),
                'best_train_accuracy': np.max(train_accuracies),
                'avg_train_loss': np.mean(train_losses),
                'best_train_loss': np.min(train_losses),
                'avg_training_time': np.mean(train_times),
                'total_training_time': np.sum(train_times)
            })
        
        if evaluation_metrics:
            eval_accuracies = [m['metrics']['accuracy'] for m in evaluation_metrics]
            eval_losses = [m['metrics']['loss'] for m in evaluation_metrics]
            
            summary.update({
                'avg_eval_accuracy': np.mean(eval_accuracies),
                'best_eval_accuracy': np.max(eval_accuracies),
                'avg_eval_loss': np.mean(eval_losses),
                'best_eval_loss': np.min(eval_losses)
            })
        
        return summary

def create_client_config(args) -> Dict[str, Any]:
    """Create client configuration from command line arguments"""
    
    config = {
        'model_type': args.model_type,
        'model_params': {
            'num_classes': args.num_classes,
            'input_channels': 3 if args.dataset_type in ['cifar10', 'custom'] else 1
        },
        'dataset_type': args.dataset_type,
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'total_clients': args.total_clients
    }
    
    return config

def main():
    """Main function to run the federated learning client"""
    
    parser = argparse.ArgumentParser(description="Production Flower Client")
    
    # Server connection
    parser.add_argument("--server", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--client-id", type=int, default=0, help="Client ID")
    
    # Model configuration
    parser.add_argument("--model-type", type=str, default="cnn", 
                       choices=["cnn", "resnet18", "mlp"], help="Model architecture")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    
    # Data configuration
    parser.add_argument("--dataset-type", type=str, default="cifar10",
                       choices=["cifar10", "custom"], help="Dataset type")
    parser.add_argument("--data-path", type=str, default="./data", help="Data directory path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--total-clients", type=int, default=10, help="Total number of clients")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("🌸 Production Flower Federated Learning Client")
    print("=" * 60)
    print(f"Client ID: {args.client_id}")
    print(f"Server: {args.server}")
    print(f"Model: {args.model_type}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    try:
        # Create client configuration
        config = create_client_config(args)
        
        # Initialize client
        client = ProductionFlowerClient(args.client_id, config)
        
        # Print client information
        client_info = client.get_client_info()
        logger.info(f"Client capabilities: {client_info['capabilities']}")
        logger.info(f"Data info: {client_info['data_info']}")
        
        # Start client
        logger.info(f"Connecting to server at {args.server}")
        fl.client.start_numpy_client(
            server_address=args.server,
            client=client
        )
        
        # Print final performance summary
        performance = client.get_performance_summary()
        if performance:
            print("\n" + "=" * 60)
            print("TRAINING SUMMARY")
            print("=" * 60)
            print(f"Total rounds: {performance.get('total_rounds', 0)}")
            print(f"Best training accuracy: {performance.get('best_train_accuracy', 0):.4f}")
            print(f"Best evaluation accuracy: {performance.get('best_eval_accuracy', 0):.4f}")
            print(f"Total training time: {performance.get('total_training_time', 0):.2f}s")
            print("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Client failed: {e}")
        raise

if __name__ == "__main__":
    main()
