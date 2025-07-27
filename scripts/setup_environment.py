"""
Setup script for Flower Federated Learning environment
This script installs required dependencies and sets up the environment.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False
    return True

def setup_flower_environment():
    """Set up the Flower federated learning environment."""
    
    print("🌸 Setting up Flower Federated Learning Environment")
    print("=" * 60)
    
    # Required packages
    packages = [
        "flwr[simulation]>=1.6.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.8.0",
    ]
    
    print("Installing required packages...")
    print("-" * 40)
    
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        print("Please install these packages manually.")
        return False
    
    print("\n✅ All packages installed successfully!")
    
    # Create necessary directories
    directories = [
        "data",
        "models",
        "logs",
        "results",
        "configs"
    ]
    
    print("\nCreating project directories...")
    print("-" * 40)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Create configuration files
    create_config_files()
    
    print("\n🎉 Flower environment setup completed!")
    print("\nNext steps:")
    print("1. Run the server: python scripts/flower_server.py")
    print("2. Run clients: python scripts/flower_client.py --client-id 1")
    print("3. Monitor training through the web interface")
    
    return True

def create_config_files():
    """Create default configuration files."""
    
    # Server configuration
    server_config = """# Flower Server Configuration
server:
  address: "0.0.0.0:8080"
  num_rounds: 10
  min_clients: 2
  min_available_clients: 2

strategy:
  name: "FedAvg"
  fraction_fit: 1.0
  fraction_evaluate: 1.0
  
model:
  name: "SimpleModel"
  num_classes: 10
  
training:
  epochs: 1
  learning_rate: 0.01
  batch_size: 32
"""
    
    with open("configs/server_config.yaml", "w") as f:
        f.write(server_config)
    
    # Client configuration
    client_config = """# Flower Client Configuration
client:
  server_address: "localhost:8080"
  num_samples: 1000
  
training:
  epochs: 1
  learning_rate: 0.01
  batch_size: 32
  
data:
  synthetic: true
  non_iid: true
"""
    
    with open("configs/client_config.yaml", "w") as f:
        f.write(client_config)
    
    print("📄 Created configuration files in configs/")

if __name__ == "__main__":
    setup_flower_environment()
