import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import { v4 as uuidv4 } from 'uuid';
import clientPromise from '@/lib/mongodb';

const execPromise = promisify(exec);

export async function POST(req: Request) {
  try {
    const { projectId, clientIds, config } = await req.json();
    const { datasetType } = config;

    if (!projectId || !clientIds || !Array.isArray(clientIds) || clientIds.length === 0 || !datasetType) {
      return NextResponse.json({ error: 'Missing required fields: projectId, clientIds, or datasetType' }, { status: 400 });
    }

    const sessionId = uuidv4();
    const simulationLogs: { clientId: string; log: string }[] = [];

    const client = await clientPromise;
    const db = client.db("fed_learning");
    const sessionsCollection = db.collection("trainingSessions");

    // Store session details in MongoDB
    await sessionsCollection.insertOne({
      sessionId,
      projectId,
      clientIds,
      config,
      status: 'running',
      startTime: new Date(),
      logs: [], // Initialize with empty logs
    });

    // Simulate training for each client
    for (const clientId of clientIds) {
      const command = `python scripts/simulate_client_training.py --client_id ${clientId} --dataset_type ${datasetType}`;
      console.log(`Executing command for client ${clientId}: ${command}`);

      try {
        const { stdout, stderr } = await execPromise(command, { cwd: process.cwd() });
        const log = `Client ${clientId} Simulation Output:
${stdout}
${stderr}`;
        simulationLogs.push({ clientId, log });

        // Update logs in MongoDB for the current session
        await sessionsCollection.updateOne(
          { sessionId },
          { $push: { logs: { clientId, log, timestamp: new Date() } } }
        );

      } catch (error: any) {
        const log = `Client ${clientId} Simulation Error:
${error.stdout}
${error.stderr}
${error.message}`;
        simulationLogs.push({ clientId, log });

        // Update logs in MongoDB for the current session
        await sessionsCollection.updateOne(
          { sessionId },
          { $push: { logs: { clientId, log, timestamp: new Date(), error: true } } }
        );
        console.error(`Error simulating training for client ${clientId}:`, error);
      }
    }

    // Update session status to completed (or failed if any errors)
    const hasErrors = simulationLogs.some(log => log.log.includes('Simulation Error'));
    await sessionsCollection.updateOne(
      { sessionId },
      { $set: { status: hasErrors ? 'completed_with_errors' : 'completed', endTime: new Date() } }
    );

    return NextResponse.json({ sessionId, simulationLogs });

  } catch (error) {
    console.error('Failed to start training simulation:', error);
    return NextResponse.json({ error: 'Failed to start training simulation' }, { status: 500 });
  }
}
