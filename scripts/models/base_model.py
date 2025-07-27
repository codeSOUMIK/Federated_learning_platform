"""
Base model classes for federated learning
Provides common interfaces for different model types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np

class BaseModel(ABC, nn.Module):
    """Abstract base class for all federated learning models"""
    
    def __init__(self):
        super().__init__()
        self.model_name = "BaseModel"
        self.num_classes = 10
        
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model"""
        pass
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization"""
        pass
    
    def get_parameters(self) -> List[np.ndarray]:
        """Extract model parameters as numpy arrays"""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CNNModel(BaseModel):
    """Convolutional Neural Network for image classification"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        self.model_name = "CNN"
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Flatten and fully connected
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_model_config(self) -> Dict[str, Any]:
        return {
            "model_type": "CNN",
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "parameters": self.count_parameters()
        }

class ResNetBlock(nn.Module):
    """Basic ResNet block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(BaseModel):
    """ResNet-18 implementation for federated learning"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.model_name = "ResNet-18"
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_model_config(self) -> Dict[str, Any]:
        return {
            "model_type": "ResNet-18",
            "num_classes": self.num_classes,
            "parameters": self.count_parameters()
        }

class MLPModel(BaseModel):
    """Multi-Layer Perceptron for tabular data"""
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10):
        super().__init__()
        self.model_name = "MLP"
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.network(x)
        return F.log_softmax(x, dim=1)
    
    def get_model_config(self) -> Dict[str, Any]:
        return {
            "model_type": "MLP",
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "num_classes": self.num_classes,
            "parameters": self.count_parameters()
        }

def create_model(model_type: str, **kwargs) -> BaseModel:
    """Factory function to create models"""
    models = {
        "cnn": CNNModel,
        "resnet18": ResNet18,
        "mlp": MLPModel
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type.lower()](**kwargs)
