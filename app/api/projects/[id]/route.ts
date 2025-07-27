import { type NextRequest, NextResponse } from "next/server";
import { MongoClient, ObjectId } from "mongodb";

// MongoDB Connection Details (re-using from other API files)
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

export async function GET(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const projectId = params.id;

    // Validate if projectId is a valid MongoDB ObjectId
    if (!ObjectId.isValid(projectId)) {
      return NextResponse.json({ error: "Invalid Project ID format" }, { status: 400 });
    }

    const mongoClient = await clientPromise;
    const db = mongoClient.db(dbName);
    const projectsCollection = db.collection("projects");

    // Fetch project by its MongoDB _id
    const project = await projectsCollection.findOne({ _id: new ObjectId(projectId) });

    if (!project) {
      return NextResponse.json({ error: "Project not found" }, { status: 404 });
    }

    return NextResponse.json({ project });
  } catch (error) {
    console.error("Error fetching project:", error);
    return NextResponse.json({ error: "Failed to fetch project" }, { status: 500 });
  }
}

export async function PUT(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const projectId = params.id;
    if (!ObjectId.isValid(projectId)) {
      return NextResponse.json({ error: "Invalid Project ID format" }, { status: 400 });
    }

    const mongoClient = await clientPromise;
    const db = mongoClient.db(dbName);
    const projectsCollection = db.collection("projects");

    const updatedData = await request.json();

    // Only allow specific fields to be updated by the server
    const updateFields: { [key: string]: any } = {};
    if (updatedData.status) updateFields.status = updatedData.status;
    if (updatedData.currentRound !== undefined) updateFields.currentRound = updatedData.currentRound;
    if (updatedData.accuracy !== undefined) updateFields.accuracy = updatedData.accuracy;
    if (updatedData.loss !== undefined) updateFields.loss = updatedData.loss;
    if (updatedData.clients !== undefined) updateFields.clients = updatedData.clients;

    if (Object.keys(updateFields).length === 0) {
      return NextResponse.json({ message: "No valid fields to update" }, { status: 200 });
    }

    const result = await projectsCollection.updateOne(
      { _id: new ObjectId(projectId) },
      { $set: updateFields }
    );

    if (result.modifiedCount === 0) {
      return NextResponse.json({ error: "Project not found or no changes made" }, { status: 404 });
    }

    const updatedProject = await projectsCollection.findOne({ _id: new ObjectId(projectId) });
    return NextResponse.json({ project: updatedProject });
  } catch (error) {
    console.error("Error updating project:", error);
    return NextResponse.json({ error: "Failed to update project" }, { status: 500 });
  }
}

/*
export async function DELETE(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const projectId = params.id;
    const mongoClient = await clientPromise;
    const db = mongoClient.db(dbName);
    const projectsCollection = db.collection("projects");

    const result = await projectsCollection.deleteOne({ _id: new ObjectId(projectId) });

    if (result.deletedCount === 0) {
      return NextResponse.json({ error: "Project not found" }, { status: 404 });
    }

    return NextResponse.json({ message: "Project deleted successfully" });
  } catch (error) {
    console.error("Error deleting project:", error);
    return NextResponse.json({ error: "Failed to delete project" }, { status: 500 });
  }
}
*/