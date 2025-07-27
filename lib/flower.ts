import { spawn } from 'child_process';

export function runFlowerServer(numRounds: number, numClients: number, projectId: string): Promise<void> {
  return new Promise((resolve, reject) => {
    // Start the Flower server
    const serverProcess = spawn('python', [
      'scripts/flower_server.py',
      '--rounds', numRounds.toString(),
      '--min-clients', numClients.toString(),
      '--min-available-clients', numClients.toString(),
      '--project-id', projectId,
    ], {
      detached: true,
      stdio: 'ignore',
    });

    serverProcess.unref(); // Allow the Node.js event loop to exit independently of the child

    serverProcess.on('error', (err) => {
      console.error('Failed to start Flower server process:', err);
      reject(new Error('Failed to start Flower server'));
    });

    console.log(`Flower server process started with PID: ${serverProcess.pid}`);

    // Give the server a moment to start up before launching clients
    setTimeout(() => {
      resolve();
    }, 5000); // Wait 5 seconds for the server to initialize and for clients to connect manually
  });
}