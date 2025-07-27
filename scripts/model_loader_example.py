"""
Example script for loading and using downloaded federated learning models
This script demonstrates how to load and use models downloaded from the platform
"""

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
import zipfile
import tempfile
import os

class CNNModel(nn.Module):
    """CNN model architecture matching the federated learning setup"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 28 * 28, num_classes)  # Adjust based on input size
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class MLPModel(nn.Module):
    """MLP model architecture matching the federated learning setup"""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLPModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

class FederatedModelLoader:
    """Utility class for loading federated learning models"""
    
    def __init__(self):
        self.model = None
        self.metadata = None
        self.config = None
        self.training_history = None
    
    def load_from_zip(self, zip_path: str):
        """Load model from ZIP package downloaded from platform"""
        print(f"Loading model from ZIP: {zip_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Load metadata
            metadata_path = os.path.join(temp_dir, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"Loaded metadata for project: {self.metadata['projectName']}")
            
            # Load configuration
            config_path = os.path.join(temp_dir, 'model_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"Loaded config for model type: {self.config['model_type']}")
            
            # Load training history
            history_path = os.path.join(temp_dir, 'training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
                print(f"Loaded training history with {len(self.training_history['rounds'])} rounds")
            
            # Load model weights
            weights_path = os.path.join(temp_dir, 'model_weights.bin')
            if os.path.exists(weights_path):
                with open(weights_path, 'rb') as f:
                    weights_data = f.read()
                    weights_json = json.loads(weights_data.decode('utf-8'))
                
                # Create model based on type
                if self.config['model_type'] == 'cnn':
                    self.model = CNNModel(
                        num_classes=self.config['parameters'].get('num_classes', 10),
                        input_channels=self.config['parameters'].get('input_channels', 3)
                    )
                else:
                    self.model = MLPModel(
                        input_size=self.config['parameters'].get('input_size', 784),
                        hidden_size=self.config['parameters'].get('hidden_size', 128),
                        num_classes=self.config['parameters'].get('num_classes', 10)
                    )
                
                # Load weights into model
                self._load_weights_from_dict(weights_json)
                print(f"Successfully loaded {self.config['model_type'].upper()} model")
            
            return self
    
    def load_from_pytorch(self, pth_path: str, metadata_path: str = None):
        """Load model from PyTorch .pth file"""
        print(f"Loading PyTorch model from: {pth_path}")
        
        # Load metadata if provided
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load PyTorch model
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            # Standard PyTorch checkpoint format
            model_type = checkpoint.get('model_type', 'cnn')
            model_params = checkpoint.get('model_params', {})
            
            if model_type == 'cnn':
                self.model = CNNModel(**model_params)
            else:
                self.model = MLPModel(**model_params)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.metadata = checkpoint.get('metadata', {})
            self.config = checkpoint.get('config', {})
        else:
            # Direct state dict
            print("Warning: Loading direct state dict, model architecture must be known")
            # You'll need to specify the model architecture manually
        
        print("PyTorch model loaded successfully")
        return self
    
    def _load_weights_from_dict(self, weights_dict):
        """Load weights from dictionary format into PyTorch model"""
        state_dict = {}
        
        for layer_name, layer_weights in weights_dict.items():
            if isinstance(layer_weights, dict):
                # Handle layers with weight and bias
                if 'weight' in layer_weights:
                    weight_tensor = torch.tensor(layer_weights['weight'], dtype=torch.float32)
                    # Reshape based on layer type
                    if 'conv' in layer_name:
                        # Convolutional layer weights
                        if layer_name == 'conv1':
                            weight_tensor = weight_tensor.view(32, 3, 3, 3)
                        elif layer_name == 'conv2':
                            weight_tensor = weight_tensor.view(64, 32, 3, 3)
                        elif layer_name == 'conv3':
                            weight_tensor = weight_tensor.view(128, 64, 3, 3)
                    elif 'fc' in layer_name or 'output' in layer_name:
                        # Fully connected layer weights
                        out_features = len(layer_weights['bias']) if 'bias' in layer_weights else weight_tensor.size(0)
                        in_features = len(layer_weights['weight']) // out_features
                        weight_tensor = weight_tensor.view(out_features, in_features)
                    elif 'hidden' in layer_name:
                        # Hidden layer weights
                        out_features = len(layer_weights['bias']) if 'bias' in layer_weights else 128
                        in_features = len(layer_weights['weight']) // out_features
                        weight_tensor = weight_tensor.view(out_features, in_features)
                    
                    state_dict[f"{layer_name}.weight"] = weight_tensor
                
                if 'bias' in layer_weights:
                    bias_tensor = torch.tensor(layer_weights['bias'], dtype=torch.float32)
                    state_dict[f"{layer_name}.bias"] = bias_tensor
        
        self.model.load_state_dict(state_dict, strict=False)
    
    def predict(self, input_data):
        """Make predictions using the loaded model"""
        if self.model is None:
            raise ValueError("No model loaded. Call load_from_zip() or load_from_pytorch() first.")
        
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                input_tensor = input_data
            
            # Add batch dimension if needed
            if len(input_tensor.shape) == 3:  # Single image
                input_tensor = input_tensor.unsqueeze(0)
            elif len(input_tensor.shape) == 1:  # Single vector
                input_tensor = input_tensor.unsqueeze(0)
            
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            return predictions.numpy(), probabilities.numpy()
    
    def get_model_info(self):
        """Get comprehensive model information"""
        info = {
            'model_loaded': self.model is not None,
            'metadata': self.metadata,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            })
        
        return info
    
    def export_to_onnx(self, output_path: str, input_shape: tuple = None):
        """Export loaded model to ONNX format"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        if input_shape is None:
            # Default input shapes based on model type
            if self.config and self.config.get('model_type') == 'cnn':
                input_shape = (1, 3, 224, 224)  # Batch, Channels, Height, Width
            else:
                input_shape = (1, 784)  # Batch, Features
        
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to ONNX: {output_path}")

# Example usage
def main():
    """Example of how to use the FederatedModelLoader"""
    
    # Initialize loader
    loader = FederatedModelLoader()
    
    # Example 1: Load from ZIP package
    zip_path = "path/to/your/downloaded_model.zip"
    if os.path.exists(zip_path):
        print("Loading model from ZIP package...")
        loader.load_from_zip(zip_path)
        
        # Get model information
        info = loader.get_model_info()
        print(f"Model Info: {json.dumps(info, indent=2)}")
        
        # Make predictions (example with random data)
        if info['model_loaded']:
            if info['config']['model_type'] == 'cnn':
                # Example image data (3 channels, 224x224)
                sample_image = np.random.randn(3, 224, 224).astype(np.float32)
                predictions, probabilities = loader.predict(sample_image)
                print(f"Prediction: Class {predictions[0]}")
                print(f"Confidence: {probabilities[0].max():.3f}")
            else:
                # Example vector data
                sample_vector = np.random.randn(784).astype(np.float32)
                predictions, probabilities = loader.predict(sample_vector)
                print(f"Prediction: Class {predictions[0]}")
                print(f"Confidence: {probabilities[0].max():.3f}")
            
            # Export to ONNX
            loader.export_to_onnx("exported_model.onnx")
    
    # Example 2: Load from PyTorch file
    pth_path = "path/to/your/model.pth"
    if os.path.exists(pth_path):
        print("Loading model from PyTorch file...")
        loader.load_from_pytorch(pth_path)
        
        # Use the model...
        info = loader.get_model_info()
        print(f"Loaded PyTorch model with {info['total_parameters']} parameters")

if __name__ == "__main__":
    main()
