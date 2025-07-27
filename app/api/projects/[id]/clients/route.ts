import { type NextRequest, NextResponse } from "next/server";
import * as fs from 'fs/promises';
import * as path from 'path';
import { MongoClient, ObjectId } from "mongodb";

const PROJECTS_FILE = path.join(process.cwd(), 'data', 'projects.json');

// MongoDB Connection Details (re-using from clients API)
const uri = "mongodb+srv://Finovate:Soumik1234@finovate.unx3ozt.mongodb.net/?retryWrites=true&w=majority&appName=Finovate";
const client = new MongoClient(uri);
const dbName = "federated_learning";

let clientPromise: Promise<MongoClient>;

if (process.env.NODE_ENV === "development") {
  if (!(global as any)._mongoClientPromise) {
    (global as any)._mongoClientPromise = client.connect();
  }
  clientPromise = (global as any)._mongoClientPromise;
} else {
  clientPromise = client.connect();
}

// Helper to read projects from JSON file (for now, will transition to DB)
async function readProjectsFromFile(): Promise<any[]> {
  try {
    const data = await fs.readFile(PROJECTS_FILE, 'utf-8');
    return JSON.parse(data);
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      await fs.writeFile(PROJECTS_FILE, JSON.stringify([]));
      return [];
    }
    console.error("Error reading projects file:", error);
    throw new Error("Failed to read projects");
  }
}

// Helper to write projects to JSON file (for now, will transition to DB)
async function writeProjectsToFile(projects: any[]): Promise<void> {
  try {
    await fs.writeFile(PROJECTS_FILE, JSON.stringify(projects, null, 2));
  } catch (error) {
    console.error("Error writing projects file:", error);
    throw new Error("Failed to write projects");
  }
}

export async function GET(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const projectId = params.id;
    const mongoClient = await clientPromise;
    const db = mongoClient.db(dbName);
    const projectsCollection = db.collection("projects");
    const clientsCollection = db.collection("clients");

    // Query project by _id (ObjectId)
    const project = await projectsCollection.findOne({ _id: new ObjectId(projectId) });

    if (!project) {
      return NextResponse.json({ error: "Project not found" }, { status: 404 });
    }

    // Fetch full client details for the client IDs stored in the project
    const projectClientIds = project.associatedClientIds || [];
    const associatedClients = await clientsCollection.find({ _id: { $in: projectClientIds.map((id: string) => new ObjectId(id)) } }).toArray();

    return NextResponse.json({ clients: associatedClients });
  } catch (error) {
    console.error("Error fetching project clients:", error);
    return NextResponse.json({ error: "Failed to fetch project clients" }, { status: 500 });
  }
}

export async function POST(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const projectId = params.id;
    const mongoClient = await clientPromise;
    const db = mongoClient.db(dbName);
    const projectsCollection = db.collection("projects");

    const { clientId } = await request.json();

    if (!clientId) {
      return NextResponse.json({ error: "Client ID is required" }, { status: 400 });
    }

    // Query project by _id (ObjectId)
    const project = await projectsCollection.findOne({ _id: new ObjectId(projectId) });

    if (!project) {
      return NextResponse.json({ error: "Project not found" }, { status: 404 });
    }

    // Ensure associatedClientIds is an array
    const associatedClientIds = project.associatedClientIds || [];

    // Check if client is already associated
    if (associatedClientIds.includes(new ObjectId(clientId))) { // Compare ObjectIds
      return NextResponse.json({ message: "Client already associated with this project" }, { status: 200 });
    }

    // Add client ID (as ObjectId) to the project's associatedClientIds array
    const updateResult = await projectsCollection.updateOne(
      { _id: new ObjectId(projectId) },
      { $addToSet: { associatedClientIds: new ObjectId(clientId) }, $inc: { clients: 1 } } // Increment clients count
    );

    if (updateResult.modifiedCount === 0) {
      return NextResponse.json({ error: "Failed to add client to project or client already added" }, { status: 400 });
    }

    return NextResponse.json({ message: "Client added to project successfully" });
  } catch (error) {
    console.error("Error adding client to project:", error);
    return NextResponse.json({ error: "Failed to add client to project" }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const projectId = params.id;
    const mongoClient = await clientPromise;
    const db = mongoClient.db(dbName);
    const projectsCollection = db.collection("projects");

    const { searchParams } = new URL(request.url);
    const clientId = searchParams.get("clientId");

    if (!clientId) {
      return NextResponse.json({ error: "Client ID is required" }, { status: 400 });
    }

    // Query project by _id (ObjectId)
    const project = await projectsCollection.findOne({ _id: new ObjectId(projectId) });

    if (!project) {
      return NextResponse.json({ error: "Project not found" }, { status: 404 });
    }

    // Remove client ID from the project's associatedClientIds array
    const updateResult = await projectsCollection.updateOne(
      { _id: new ObjectId(projectId) },
      { $pull: { associatedClientIds: new ObjectId(clientId) }, $inc: { clients: -1 } } // Decrement clients count
    );

    if (updateResult.modifiedCount === 0) {
      return NextResponse.json({ error: "Failed to remove client from project or client not found in project" }, { status: 400 });
    }

    return NextResponse.json({ message: "Client removed from project successfully" });
  } catch (error) {
    console.error("Error removing client from project:", error);
    return NextResponse.json({ error: "Failed to remove client from project" }, { status: 500 });
  }
}
