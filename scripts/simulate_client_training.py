import argparse
import time
import random

def simulate_client_training(client_id: str, dataset_type: str):
    print(f"--- Simulating Client: {client_id} ---")
    print(f"[Client {client_id}] Initializing federated learning environment...")
    time.sleep(random.uniform(0.5, 1.5))

    print(f"[Client {client_id}] Checking for dependencies and model updates...")
    print(f"[Client {client_id}] Dependencies and model transferred successfully (simulated).")
    time.sleep(random.uniform(0.5, 1.0))

    print(f"[Client {client_id}] Loading private dataset: {dataset_type}...")
    # Simulate different sample sizes based on dataset type
    if dataset_type == "mnist":
        num_samples = random.randint(5000, 10000)
    elif dataset_type == "cifar10":
        num_samples = random.randint(1000, 5000)
    elif dataset_type == "fashion_mnist":
        num_samples = random.randint(4000, 8000)
    else:
        num_samples = random.randint(1000, 10000)
    print(f"[Client {client_id}] Loaded {num_samples} samples of {dataset_type} data.")
    time.sleep(random.uniform(1.0, 2.0))

    print(f"[Client {client_id}] Starting local training...")
    for i in range(random.randint(2, 5)): # Simulate a few local epochs
        loss = round(random.uniform(0.1, 0.5), 4)
        accuracy = round(random.uniform(0.7, 0.95), 4)
        print(f"[Client {client_id}]   Epoch {i+1}: Loss = {loss}, Accuracy = {accuracy}")
        time.sleep(random.uniform(0.3, 0.8))

    print(f"[Client {client_id}] Local training complete. Calculating model updates...")
    time.sleep(random.uniform(0.5, 1.0))

    print(f"[Client {client_id}] Sending updated weights to Flower server for aggregation...")
    time.sleep(random.uniform(0.5, 1.5))
    print(f"[Client {client_id}] Weights sent successfully.")
    print(f"--- Client {client_id} Simulation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a federated learning client.")
    parser.add_argument("--client_id", type=str, required=True, help="The ID of the client.")
    parser.add_argument("--dataset_type", type=str, required=True, help="The type of dataset to simulate (e.g., mnist, cifar10).")
    args = parser.parse_args()

    simulate_client_training(args.client_id, args.dataset_type)
