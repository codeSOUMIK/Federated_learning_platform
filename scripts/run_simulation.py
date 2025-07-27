"""
Flower Federated Learning Simulation
This script runs a complete federated learning simulation with multiple clients.
"""

import flwr as fl
from flwr.simulation import start_simulation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

class SimpleModel(nn.Module):
    """Simple CNN model for simulation"""
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

def generate_client_data(client_id: int, num_samples: int = 1000, non_iid: bool = True):
    """Generate data for a specific client."""
    np.random.seed(client_id + 42)
    
    # Generate synthetic image data
    X = np.random.randn(num_samples, 3, 32, 32).astype(np.float32)
    
    if non_iid:
        # Create non-IID data distribution
        if client_id < 3:
            # Clients 0-2: More samples from classes 0-3
            y = np.random.choice(range(4), num_samples, p=[0.4, 0.3, 0.2, 0.1])
        elif client_id < 6:
            # Clients 3-5: More samples from classes 4-7
            y = np.random.choice(range(4, 8), num_samples, p=[0.4, 0.3, 0.2, 0.1])
        else:
            # Clients 6+: More samples from classes 8-9
            y = np.random.choice(range(8, 10), num_samples, p=[0.6, 0.4])
    else:
        # IID data distribution
        y = np.random.randint(0, 10, num_samples)
    
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)

class SimulationClient(fl.client.NumPyClient):
    """Client for simulation."""
    
    def __init__(self, client_id: int, num_samples: int = 1000, non_iid: bool = True):
        self.client_id = client_id
        self.model = SimpleModel()
        
        # Generate client data
        X, y = generate_client_data(client_id, num_samples, non_iid)
        
        # Split into train and test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"Client {client_id}: {len(X_train)} train, {len(X_test)} test samples")
    
    def get_parameters(self, config):
        return get_model_parameters(self.model)
    
    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        
        # Train
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        epochs = config.get("epochs", 1)
        total_loss = 0.0
        
        for epoch in range(epochs):
            for data, target in self.train_loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / (len(self.train_loader) * epochs)
        
        return get_model_parameters(self.model), len(self.train_loader.dataset), {"loss": avg_loss}
    
    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        
        # Evaluate
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        return avg_loss, len(self.test_loader.dataset), {"accuracy": accuracy}

def client_fn(cid: str) -> SimulationClient:
    """Create a client for simulation."""
    return SimulationClient(client_id=int(cid), num_samples=1000, non_iid=True)

class LoggingStrategy(fl.server.strategy.FedAvg):
    """Strategy with enhanced logging for simulation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = {
            "rounds": [],
            "losses": [],
            "accuracies": [],
            "num_clients": []
        }
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results and log metrics."""
        
        if not results:
            return None, {}
        
        # Calculate weighted average of metrics
        total_examples = sum([num_examples for _, num_examples, _ in results])
        weighted_loss = sum([loss * num_examples for loss, num_examples, _ in results]) / total_examples
        
        # Calculate accuracy if available
        accuracies = [metrics.get("accuracy", 0) for _, _, metrics in results if "accuracy" in metrics]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        # Log results
        print(f"Round {server_round}: Loss={weighted_loss:.4f}, Accuracy={avg_accuracy:.4f}, Clients={len(results)}")
        
        # Store results
        self.results["rounds"].append(server_round)
        self.results["losses"].append(weighted_loss)
        self.results["accuracies"].append(avg_accuracy)
        self.results["num_clients"].append(len(results))
        
        return weighted_loss, {"accuracy": avg_accuracy}
    
    def save_results(self, filename: str):
        """Save simulation results to file."""
        os.makedirs("results", exist_ok=True)
        with open(f"results/{filename}", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to results/{filename}")

def plot_results(results: Dict, save_path: str = None):
    """Plot simulation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(results["rounds"], results["losses"], 'b-', marker='o')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(results["rounds"], results["accuracies"], 'g-', marker='s')
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{save_path}")
        print(f"Plot saved to results/{save_path}")
    
    plt.show()

def run_simulation(
    num_clients: int = 10,
    num_rounds: int = 10,
    fraction_fit: float = 0.5,
    fraction_evaluate: float = 0.5,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
):
    """Run federated learning simulation."""
    
    print("🌸 Starting Flower Federated Learning Simulation")
    print("=" * 60)
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Fraction fit: {fraction_fit}")
    print(f"Min fit clients: {min_fit_clients}")
    print("=" * 60)
    
    # Create strategy
    strategy = LoggingStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=num_clients,
    )
    
    # Run simulation
    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
    
    # Save and plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save strategy results
    strategy.save_results(f"simulation_results_{timestamp}.json")
    
    # Plot results
    plot_results(strategy.results, f"simulation_plot_{timestamp}.png")
    
    print("\n🎉 Simulation completed successfully!")
    print(f"Final accuracy: {strategy.results['accuracies'][-1]:.4f}")
    print(f"Final loss: {strategy.results['losses'][-1]:.4f}")
    
    return history, strategy.results

if __name__ == "__main__":
    # Simulation parameters
    NUM_CLIENTS = 10
    NUM_ROUNDS = 10
    FRACTION_FIT = 0.5  # 50% of clients participate in each round
    FRACTION_EVALUATE = 0.5
    MIN_FIT_CLIENTS = 3
    MIN_EVALUATE_CLIENTS = 3
    
    # Run simulation
    history, results = run_simulation(
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
    )
