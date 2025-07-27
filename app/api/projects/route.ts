import { type NextRequest, NextResponse } from "next/server";
import { ProjectManager } from "@/lib/project-manager";
import { MongoClient, ObjectId } from "mongodb";

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

export async function GET() {
  try {
    const mongoClient = await clientPromise;
    const db = mongoClient.db(dbName);
    const projectsCollection = db.collection("projects");

    const projects = await projectsCollection.find({}).toArray();
    return NextResponse.json({ projects });
  } catch (error: any) {
    console.error("Error fetching projects:", error);
    return NextResponse.json({ error: "Failed to fetch projects" }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const mongoClient = await clientPromise;
    const db = mongoClient.db(dbName);
    const projectsCollection = db.collection("projects");

    const projectData = await request.json();

    const errors = ProjectManager.validateProject(projectData);
    if (errors.length > 0) {
      return NextResponse.json({ errors }, { status: 400 });
    }

    // Create project data, let MongoDB generate _id
    const newProject = {
      ...projectData,
      status: "created",
      currentRound: 0,
      accuracy: 0,
      clients: 0, // This will be updated when clients are associated
      createdAt: new Date().toISOString().split("T")[0],
      associatedClientIds: [], // Initialize with an empty array for client ObjectIds
    };

    const result = await projectsCollection.insertOne(newProject);

    // Return the inserted document, including the MongoDB _id
    return NextResponse.json({ project: { ...newProject, _id: result.insertedId } }, { status: 201 });
  } catch (error: any) {
    console.error("Error creating project:", error);
    return NextResponse.json({ error: "Failed to create project" }, { status: 400 });
  }
}