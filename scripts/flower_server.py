"""
Flower Federated Learning Server Implementation
This script sets up a Flower server for federated learning experiments.
"""

import flwr as fl
import numpy as np
from typing import Dict, Optional, Tuple, List
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.server.strategy import FedAvg
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os

# Define the path to the projects.json file relative to this script
PROJECTS_FILE_PATH = os.path.join(os.path.dirname(__file__), '..\', 'data\', 'projects.json')

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

class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy with enhanced logging and evaluation."""
    
    def __init__(self, project_id: str, **kwargs):
        super().__init__(**kwargs)
        self.model = SimpleModel()
        self.project_id = project_id
        
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        parameters = get_model_parameters(self.model)
        return fl.common.ndarrays_to_parameters(parameters)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        
        # Log round information
        print(f"Round {server_round}: Aggregating {len(results)} client updates")
        
        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # Update server model
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            set_model_parameters(self.model, aggregated_ndarrays)
            
            # Log aggregated metrics
            if aggregated_metrics:
                print(f"Round {server_round} metrics: {aggregated_metrics}")
        
        return aggregated_parameters, aggregated_metrics
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the global model on a test dataset."""
        
        # Convert parameters to model weights
        parameters_ndarrays = fl.common.parameters_to_ndarrays(parameters)
        set_model_parameters(self.model, parameters_ndarrays)
        
        # Simulate evaluation (in practice, use actual test data)
        # This would typically evaluate on a centralized test set
        loss = np.random.uniform(0.1, 1.0)  # Simulated loss
        accuracy = np.random.uniform(0.7, 0.95)  # Simulated accuracy
        
        print(f"Round {server_round} - Server evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        # Update project status via Next.js API
        try:
            import requests
            api_url = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:3000")
            update_url = f"{api_url}/api/projects/{self.project_id}"
            
            payload = {
                "accuracy": accuracy,
                "currentRound": server_round,
                "status": "running" if server_round < self.num_rounds else "completed",
            }
            
            response = requests.put(update_url, json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors
            print(f"Successfully updated project {self.project_id} via API. Status: {response.status_code}")
        except Exception as e:
            print(f"Error updating project via API: {e}")

        return loss, {"accuracy": accuracy}

def start_flower_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    min_clients: int = 2,
    min_available_clients: int = 2,
    project_id: str = "default_project"
):
    """Start the Flower federated learning server."""
    
    print(f"Starting Flower server on {server_address}")
    print(f"Configuration: {num_rounds} rounds, min {min_clients} clients, Project ID: {project_id}")
    
    # Define strategy
    strategy = CustomFedAvg(
        project_id=project_id,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=min_clients,  # Never sample less than min_clients for training
        min_evaluate_clients=min_clients,  # Never sample less than min_clients for evaluation
        min_available_clients=min_available_clients,  # Wait until all clients are available
        evaluate_fn=None,  # Use custom evaluate method in strategy
        num_rounds=num_rounds, # Pass num_rounds to strategy for status update
    )
    
    # Configure server
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum number of clients required for training")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum number of clients that need to be connected to start a round")
    parser.add_argument("--project-id", type=str, default="default_project", help="ID of the project to update in projects.json")
    
    args = parser.parse_args()
    
    print("🌸 Flower Federated Learning Server")
    print("=" * 50)
    print(f"Server Address: {args.server_address}")
    print(f"Rounds: {args.rounds}")
    print(f"Min Clients: {args.min_clients}")
    print(f"Min Available Clients: {args.min_available_clients}")
    print(f"Project ID: {args.project_id}")
    print("=" * 50)
    
    start_flower_server(
        server_address=args.server_address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        min_available_clients=args.min_available_clients,
        project_id=args.project_id
    )