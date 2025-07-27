"""
Example script showing how a real client can register with the federated learning platform
This script demonstrates the client-side registration process
"""

import requests
import json
import platform
import psutil
import socket
import time
from typing import Dict, Any

class FederatedLearningClient:
    """
    Example federated learning client that can register with the platform
    """
    
    def __init__(self, server_url: str = "http://localhost:3000", client_name: str = None):
        self.server_url = server_url
        self.client_name = client_name or f"Client-{socket.gethostname()}"
        self.client_id = None
        self.connection_key = None
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system hardware and capability information"""
        try:
            # Get CPU information
            cpu_info = platform.processor() or "Unknown CPU"
            cpu_cores = psutil.cpu_count(logical=False) or 1
            
            # Get memory information
            memory = psutil.virtual_memory()
            memory_gb = round(memory.total / (1024**3), 1)
            
            # Check for GPU (simplified check)
            has_gpu = False
            gpu_info = "No GPU detected"
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    has_gpu = True
                    gpu_info = gpus[0].name
            except ImportError:
                # Try nvidia-ml-py as alternative
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    gpu_count = pynvml.nvmlDeviceGetCount()
                    if gpu_count > 0:
                        has_gpu = True
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_info = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                except:
                    pass
            
            return {
                "capabilities": {
                    "gpu": has_gpu,
                    "memory": f"{memory_gb}GB",
                    "cpuCores": cpu_cores
                },
                "hardwareInfo": {
                    "cpu": cpu_info,
                    "gpu": gpu_info,
                    "platform": f"{platform.system()} {platform.release()}"
                }
            }
        except Exception as e:
            print(f"Warning: Could not gather complete system info: {e}")
            return {
                "capabilities": {
                    "gpu": False,
                    "memory": "Unknown",
                    "cpuCores": 1
                },
                "hardwareInfo": {
                    "cpu": "Unknown",
                    "gpu": "Unknown",
                    "platform": platform.system()
                }
            }
    
    def get_dataset_info(self, samples: int = 1000, classes: int = 10, dataset_type: str = "Medical Images") -> Dict[str, Any]:
        """Get dataset information (customize based on your actual dataset)"""
        return {
            "dataInfo": {
                "samples": samples,
                "classes": classes,
                "datasetType": dataset_type
            }
        }
    
    def register_with_server(self, samples: int = 1000, classes: int = 10, dataset_type: str = "Medical Images") -> bool:
        """Register this client with the federated learning server"""
        try:
            # Gather system information
            system_info = self.get_system_info()
            dataset_info = self.get_dataset_info(samples, classes, dataset_type)
            
            # Prepare registration data
            registration_data = {
                "name": self.client_name,
                **system_info,
                **dataset_info
            }
            
            print(f"Registering client '{self.client_name}' with server...")
            print(f"System info: {json.dumps(system_info, indent=2)}")
            
            # Send registration request
            response = requests.post(
                f"{self.server_url}/api/clients/register",
                json=registration_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 201:
                result = response.json()
                self.client_id = result.get("clientId")
                self.connection_key = result.get("connectionKey")
                
                print(f"✅ Registration successful!")
                print(f"Client ID: {self.client_id}")
                print(f"Connection Key: {self.connection_key}")
                return True
            else:
                error_data = response.json()
                print(f"❌ Registration failed: {error_data.get('error', 'Unknown error')}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error during registration: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during registration: {e}")
            return False
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat to server to maintain connection"""
        if not self.client_id:
            print("❌ Cannot send heartbeat: Client not registered")
            return False
            
        try:
            response = requests.put(
                f"{self.server_url}/api/clients/{self.client_id}/status",
                json={"status": "online", "lastSeen": time.time()},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                print("💓 Heartbeat sent successfully")
                return True
            else:
                print(f"❌ Heartbeat failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Heartbeat network error: {e}")
            return False
    
    def start_heartbeat_loop(self, interval: int = 30):
        """Start sending periodic heartbeats"""
        print(f"Starting heartbeat loop (every {interval} seconds)")
        try:
            while True:
                self.send_heartbeat()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Heartbeat loop stopped by user")

def main():
    """Example usage of the federated learning client"""
    print("🚀 Federated Learning Client Registration Example")
    print("=" * 50)
    
    # Initialize client
    client = FederatedLearningClient(
        server_url="http://localhost:3000",
        client_name="Example Hospital Client"
    )
    
    # Register with server
    success = client.register_with_server(
        samples=1500,  # Number of training samples
        classes=10,    # Number of classes in dataset
        dataset_type="Medical Images"  # Type of dataset
    )
    
    if success:
        print("\n🎉 Client registered successfully!")
        print("You can now see this client in the web dashboard at http://localhost:3000/clients")
        
        # Ask user if they want to start heartbeat
        try:
            start_heartbeat = input("\nStart heartbeat loop? (y/n): ").lower().strip()
            if start_heartbeat == 'y':
                client.start_heartbeat_loop()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print("\n❌ Registration failed. Please check your server connection and try again.")

if __name__ == "__main__":
    main()
