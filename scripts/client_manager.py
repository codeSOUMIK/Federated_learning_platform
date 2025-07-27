"""
Advanced Client Management System for Federated Learning
Handles client registration, monitoring, and communication
"""

import json
import time
import threading
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import socket
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClientStatus(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    TRAINING = "training"
    EVALUATING = "evaluating"
    ERROR = "error"
    DISCONNECTED = "disconnected"

@dataclass
class ClientInfo:
    """Client information structure"""
    client_id: str
    name: str
    ip_address: str
    port: int
    status: ClientStatus
    last_seen: datetime
    capabilities: Dict[str, Any]
    performance_metrics: Dict[str, float]
    data_info: Dict[str, Any]
    hardware_info: Dict[str, Any]
    created_at: datetime
    total_rounds: int = 0
    successful_rounds: int = 0
    failed_rounds: int = 0

class ClientDatabase:
    """Database manager for client information"""
    
    def __init__(self, db_path: str = "clients.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the client database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clients (
                client_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                port INTEGER NOT NULL,
                status TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                capabilities TEXT,
                performance_metrics TEXT,
                data_info TEXT,
                hardware_info TEXT,
                created_at TEXT NOT NULL,
                total_rounds INTEGER DEFAULT 0,
                successful_rounds INTEGER DEFAULT 0,
                failed_rounds INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS client_sessions (
                session_id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                project_id TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                rounds_completed INTEGER DEFAULT 0,
                avg_accuracy REAL DEFAULT 0.0,
                avg_loss REAL DEFAULT 0.0,
                FOREIGN KEY (client_id) REFERENCES clients (client_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS client_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                round_number INTEGER,
                accuracy REAL,
                loss REAL,
                training_time REAL,
                data_samples INTEGER,
                FOREIGN KEY (client_id) REFERENCES clients (client_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_client(self, client: ClientInfo):
        """Save or update client information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO clients 
            (client_id, name, ip_address, port, status, last_seen, capabilities, 
             performance_metrics, data_info, hardware_info, created_at, 
             total_rounds, successful_rounds, failed_rounds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            client.client_id,
            client.name,
            client.ip_address,
            client.port,
            client.status.value,
            client.last_seen.isoformat(),
            json.dumps(client.capabilities),
            json.dumps(client.performance_metrics),
            json.dumps(client.data_info),
            json.dumps(client.hardware_info),
            client.created_at.isoformat(),
            client.total_rounds,
            client.successful_rounds,
            client.failed_rounds
        ))
        
        conn.commit()
        conn.close()
    
    def get_client(self, client_id: str) -> Optional[ClientInfo]:
        """Get client information by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM clients WHERE client_id = ?', (client_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return ClientInfo(
                client_id=row[0],
                name=row[1],
                ip_address=row[2],
                port=row[3],
                status=ClientStatus(row[4]),
                last_seen=datetime.fromisoformat(row[5]),
                capabilities=json.loads(row[6]),
                performance_metrics=json.loads(row[7]),
                data_info=json.loads(row[8]),
                hardware_info=json.loads(row[9]),
                created_at=datetime.fromisoformat(row[10]),
                total_rounds=row[11],
                successful_rounds=row[12],
                failed_rounds=row[13]
            )
        return None
    
    def get_all_clients(self) -> List[ClientInfo]:
        """Get all clients"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM clients ORDER BY last_seen DESC')
        rows = cursor.fetchall()
        conn.close()
        
        clients = []
        for row in rows:
            clients.append(ClientInfo(
                client_id=row[0],
                name=row[1],
                ip_address=row[2],
                port=row[3],
                status=ClientStatus(row[4]),
                last_seen=datetime.fromisoformat(row[5]),
                capabilities=json.loads(row[6]),
                performance_metrics=json.loads(row[7]),
                data_info=json.loads(row[8]),
                hardware_info=json.loads(row[9]),
                created_at=datetime.fromisoformat(row[10]),
                total_rounds=row[11],
                successful_rounds=row[12],
                failed_rounds=row[13]
            ))
        
        return clients
    
    def save_client_metrics(self, client_id: str, round_number: int, 
                           accuracy: float, loss: float, training_time: float, 
                           data_samples: int):
        """Save client training metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO client_metrics 
            (client_id, timestamp, round_number, accuracy, loss, training_time, data_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            client_id,
            datetime.now().isoformat(),
            round_number,
            accuracy,
            loss,
            training_time,
            data_samples
        ))
        
        conn.commit()
        conn.close()

class ClientManager:
    """Advanced client management system"""
    
    def __init__(self, server_address: str = "localhost", server_port: int = 8080):
        self.server_address = server_address
        self.server_port = server_port
        self.db = ClientDatabase()
        self.active_clients: Dict[str, ClientInfo] = {}
        self.client_sessions: Dict[str, str] = {}  # session_id -> client_id
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Load existing clients from database
        self.load_clients_from_db()
        
        logger.info(f"ClientManager initialized with {len(self.active_clients)} clients")
    
    def load_clients_from_db(self):
        """Load clients from database into memory"""
        clients = self.db.get_all_clients()
        for client in clients:
            self.active_clients[client.client_id] = client
    
    def register_client(self, client_data: Dict[str, Any]) -> str:
        """Register a new client"""
        client_id = client_data.get('client_id') or str(uuid.uuid4())
        
        # Get client IP (in real deployment, this would come from request)
        ip_address = client_data.get('ip_address', 'localhost')
        
        client = ClientInfo(
            client_id=client_id,
            name=client_data.get('name', f'Client-{client_id[:8]}'),
            ip_address=ip_address,
            port=client_data.get('port', 0),
            status=ClientStatus.ONLINE,
            last_seen=datetime.now(),
            capabilities=client_data.get('capabilities', {}),
            performance_metrics=client_data.get('performance_metrics', {}),
            data_info=client_data.get('data_info', {}),
            hardware_info=client_data.get('hardware_info', {}),
            created_at=datetime.now()
        )
        
        self.active_clients[client_id] = client
        self.db.save_client(client)
        
        logger.info(f"Client {client_id} registered successfully")
        return client_id
    
    def update_client_status(self, client_id: str, status: ClientStatus):
        """Update client status"""
        if client_id in self.active_clients:
            self.active_clients[client_id].status = status
            self.active_clients[client_id].last_seen = datetime.now()
            self.db.save_client(self.active_clients[client_id])
            logger.info(f"Client {client_id} status updated to {status.value}")
    
    def get_available_clients(self, min_clients: int = 1) -> List[ClientInfo]:
        """Get available clients for training"""
        available = []
        current_time = datetime.now()
        
        for client in self.active_clients.values():
            # Consider client available if seen within last 5 minutes
            if (current_time - client.last_seen).seconds < 300:
                if client.status in [ClientStatus.ONLINE, ClientStatus.TRAINING]:
                    available.append(client)
        
        logger.info(f"Found {len(available)} available clients (min required: {min_clients})")
        return available[:min_clients] if len(available) >= min_clients else []
    
    def start_training_session(self, project_id: str, client_ids: List[str]) -> str:
        """Start a training session with selected clients"""
        session_id = str(uuid.uuid4())
        
        for client_id in client_ids:
            if client_id in self.active_clients:
                self.update_client_status(client_id, ClientStatus.TRAINING)
                self.client_sessions[session_id] = client_id
        
        # Save session to database
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        for client_id in client_ids:
            cursor.execute('''
                INSERT INTO client_sessions (session_id, client_id, project_id, start_time)
                VALUES (?, ?, ?, ?)
            ''', (session_id, client_id, project_id, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Training session {session_id} started with {len(client_ids)} clients")
        return session_id
    
    def end_training_session(self, session_id: str):
        """End a training session"""
        if session_id in self.client_sessions:
            client_id = self.client_sessions[session_id]
            self.update_client_status(client_id, ClientStatus.ONLINE)
            del self.client_sessions[session_id]
            
            # Update session end time in database
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE client_sessions 
                SET end_time = ? 
                WHERE session_id = ?
            ''', (datetime.now().isoformat(), session_id))
            conn.commit()
            conn.close()
            
            logger.info(f"Training session {session_id} ended")
    
    def record_training_metrics(self, client_id: str, round_number: int, 
                               metrics: Dict[str, float]):
        """Record training metrics for a client"""
        if client_id in self.active_clients:
            client = self.active_clients[client_id]
            client.total_rounds += 1
            
            # Update performance metrics
            accuracy = metrics.get('accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            training_time = metrics.get('training_time', 0.0)
            data_samples = metrics.get('data_samples', 0)
            
            if accuracy > 0:
                client.successful_rounds += 1
            else:
                client.failed_rounds += 1
            
            # Update running averages
            if 'avg_accuracy' not in client.performance_metrics:
                client.performance_metrics['avg_accuracy'] = accuracy
            else:
                client.performance_metrics['avg_accuracy'] = (
                    client.performance_metrics['avg_accuracy'] * 0.9 + accuracy * 0.1
                )
            
            if 'avg_loss' not in client.performance_metrics:
                client.performance_metrics['avg_loss'] = loss
            else:
                client.performance_metrics['avg_loss'] = (
                    client.performance_metrics['avg_loss'] * 0.9 + loss * 0.1
                )
            
            # Save to database
            self.db.save_client(client)
            self.db.save_client_metrics(client_id, round_number, accuracy, loss, 
                                      training_time, data_samples)
            
            logger.info(f"Metrics recorded for client {client_id}, round {round_number}")
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get overall client statistics"""
        total_clients = len(self.active_clients)
        online_clients = sum(1 for c in self.active_clients.values() 
                           if c.status == ClientStatus.ONLINE)
        training_clients = sum(1 for c in self.active_clients.values() 
                             if c.status == ClientStatus.TRAINING)
        
        total_rounds = sum(c.total_rounds for c in self.active_clients.values())
        successful_rounds = sum(c.successful_rounds for c in self.active_clients.values())
        
        return {
            'total_clients': total_clients,
            'online_clients': online_clients,
            'training_clients': training_clients,
            'offline_clients': total_clients - online_clients - training_clients,
            'total_rounds': total_rounds,
            'successful_rounds': successful_rounds,
            'success_rate': successful_rounds / total_rounds if total_rounds > 0 else 0
        }
    
    def start_monitoring(self):
        """Start client monitoring thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_clients)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Client monitoring started")
    
    def stop_monitoring(self):
        """Stop client monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Client monitoring stopped")
    
    def _monitor_clients(self):
        """Monitor client health and connectivity"""
        while self.is_monitoring:
            current_time = datetime.now()
            
            for client_id, client in self.active_clients.items():
                # Check if client is offline (no heartbeat for 5 minutes)
                if (current_time - client.last_seen).seconds > 300:
                    if client.status != ClientStatus.OFFLINE:
                        logger.warning(f"Client {client_id} appears to be offline")
                        self.update_client_status(client_id, ClientStatus.OFFLINE)
            
            time.sleep(30)  # Check every 30 seconds
    
    def export_client_data(self, format: str = 'json') -> str:
        """Export client data for analysis"""
        clients_data = []
        for client in self.active_clients.values():
            client_dict = asdict(client)
            client_dict['status'] = client.status.value
            client_dict['last_seen'] = client.last_seen.isoformat()
            client_dict['created_at'] = client.created_at.isoformat()
            clients_data.append(client_dict)
        
        if format.lower() == 'json':
            return json.dumps(clients_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize client manager
    client_manager = ClientManager()
    
    # Register some test clients
    test_clients = [
        {
            'name': 'Hospital-A',
            'ip_address': '192.168.1.100',
            'port': 8081,
            'capabilities': {'gpu': True, 'memory': '16GB'},
            'data_info': {'samples': 1000, 'classes': 10},
            'hardware_info': {'cpu': 'Intel i7', 'gpu': 'RTX 3080'}
        },
        {
            'name': 'Hospital-B',
            'ip_address': '192.168.1.101',
            'port': 8082,
            'capabilities': {'gpu': False, 'memory': '8GB'},
            'data_info': {'samples': 800, 'classes': 10},
            'hardware_info': {'cpu': 'Intel i5', 'gpu': 'None'}
        }
    ]
    
    for client_data in test_clients:
        client_id = client_manager.register_client(client_data)
        print(f"Registered client: {client_id}")
    
    # Start monitoring
    client_manager.start_monitoring()
    
    # Get statistics
    stats = client_manager.get_client_statistics()
    print(f"Client statistics: {stats}")
    
    # Get available clients
    available = client_manager.get_available_clients(min_clients=2)
    print(f"Available clients: {len(available)}")
    
    # Export data
    exported_data = client_manager.export_client_data()
    print("Exported client data (first 200 chars):")
    print(exported_data[:200] + "...")
