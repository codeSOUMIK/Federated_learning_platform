import { type NextRequest, NextResponse } from "next/server";
import { MongoClient, ObjectId } from "mongodb";

// MongoDB Connection Details
const uri = "mongodb+srv://Finovate:Soumik1234@finovate.unx3ozt.mongodb.net/?retryWrites=true&w=majority&appName=Finovate";

// const uri = process.env.DB_URI
const client = new MongoClient(uri);
const dbName = "federated_learning"; // You can choose your database name

let clientPromise: Promise<MongoClient>;

if (process.env.NODE_ENV === "development") {
  // In development mode, use a global variable so that the client is not recreated on every hot reload
  if (!(global as any)._mongoClientPromise) {
    (global as any)._mongoClientPromise = client.connect();
  }
  clientPromise = (global as any)._mongoClientPromise;
} else {
  // In production mode, it's best to not use a global variable.
  clientPromise = client.connect();
}

export async function GET() {
  try {
    const client = await clientPromise;
    const db = client.db(dbName);
    const clientsCollection = db.collection("clients");

    const clients = await clientsCollection.find({}).toArray();

    // Simulate checking client connectivity and updating performance (optional, can be removed)
    // For demonstration, we'll ensure clients are always reported as 'online'
    const updatedClients = clients.map((client: any) => ({
      ...client,
      status: "online",
      lastSeen: new Date().toISOString(), // Update last seen to current time
    }));

    return NextResponse.json({ clients: updatedClients });
  } catch (error) {
    console.error("Error fetching clients:", error);
    return NextResponse.json({ error: "Failed to fetch clients" }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const client = await clientPromise;
    const db = client.db(dbName);
    const clientsCollection = db.collection("clients");

    const clientDataArray = await request.json();

    if (!Array.isArray(clientDataArray)) {
      return NextResponse.json({ error: "Request body must be an array of clients" }, { status: 400 });
    }

    const results: { success: boolean; client?: any; error?: string }[] = [];

    for (const clientData of clientDataArray) {
      try {
        // Validate required fields
        if (!clientData.name || !clientData.ipAddress) {
          results.push({ success: false, error: "Name and IP address are required for a client" });
          continue;
        }

        // Check if client already exists
        const existingClient = await clientsCollection.findOne({
          ipAddress: clientData.ipAddress,
          port: clientData.port || 8080,
        });

        if (existingClient) {
          results.push({ success: false, error: `Client with IP ${clientData.ipAddress} and port ${clientData.port || 8080} already exists` });
          continue;
        }

        const newClient = {
          id: `client-${Date.now()}`,
          name: clientData.name,
          ipAddress: clientData.ipAddress,
          port: clientData.port || 8080,
          status: "online", // Default to online for immediate visibility
          lastSeen: new Date().toISOString(),
          capabilities: clientData.capabilities || {
            gpu: false,
            memory: "Unknown",
            cpuCores: 0,
          },
          dataInfo: clientData.dataInfo || {
            samples: 0,
            classes: 0,
            datasetType: "Unknown",
          },
          performance: {
            avgAccuracy: 0,
            avgLoss: 0,
            totalRounds: 0,
            successfulRounds: 0,
          },
          hardwareInfo: clientData.hardwareInfo || {
            cpu: "Unknown",
            gpu: "Unknown",
            platform: "Unknown",
          },
          registeredAt: new Date().toISOString(),
          connectionKey: generateConnectionKey(),
        };

        const result = await clientsCollection.insertOne(newClient);
        results.push({ success: true, client: { ...newClient, _id: result.insertedId } });
      } catch (innerError: any) {
        console.error("Error processing individual client:", innerError);
        results.push({ success: false, error: innerError.message });
      }
    }

    return NextResponse.json({ results });
  } catch (error: any) {
    console.error("Error creating clients:", error);
    return NextResponse.json({ error: "Failed to create clients" }, { status: 400 });
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const client = await clientPromise;
    const db = client.db(dbName);
    const clientsCollection = db.collection("clients");

    const { searchParams } = new URL(request.url);
    const clientId = searchParams.get("id");

    if (!clientId) {
      return NextResponse.json({ error: "Client ID is required" }, { status: 400 });
    }

    // Assuming client.id in the frontend maps to the MongoDB _id
    const result = await clientsCollection.deleteOne({ _id: new ObjectId(clientId) });

    if (result.deletedCount === 0) {
      return NextResponse.json({ error: "Client not found" }, { status: 404 });
    }

    return NextResponse.json({ message: "Client removed successfully" });
  } catch (error) {
    console.error("Error deleting client:", error);
    return NextResponse.json({ error: "Failed to delete client" }, { status: 500 });
  }
}

// Helper functions (connectivity simulation and key generation remain)
function generateConnectionKey(): string {
  return `client-key-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// The checkClientConnectivity, registerClientWithFlowerServer, unregisterClientFromFlowerServer
// functions are no longer directly used for persistence, but their simulation logic can remain
// if you want to keep the UI's dynamic status updates.
// For a real system, you'd replace checkClientConnectivity with actual pings/API calls to clients.
async function checkClientConnectivity(ipAddress: string, port: number): Promise<boolean> {
  // For hackathon demo, always return true to show clients as online
  return true;
}

async function registerClientWithFlowerServer(client: any): Promise<void> {
  console.log(`Simulating registration of client ${client.id} with Flower server`);
  await new Promise((resolve) => setTimeout(resolve, 100));
}

async function unregisterClientFromFlowerServer(client: any): Promise<void> {
  console.log(`Simulating unregistration of client ${client.id} from Flower server`);
  await new Promise((resolve) => setTimeout(resolve, 100));
}