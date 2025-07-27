import { type NextRequest, NextResponse } from "next/server"

// Mock model versions data
const mockModelVersions = {
  "project-1": [
    {
      id: "model-v1-round10",
      version: "1.0.0",
      round: 10,
      accuracy: 0.87,
      loss: 0.23,
      size: "45.2 MB",
      format: "pytorch",
      status: "ready",
      createdAt: "2024-01-15T10:30:00Z",
    },
    {
      id: "model-v2-round20",
      version: "2.0.0",
      round: 20,
      accuracy: 0.91,
      loss: 0.18,
      size: "47.8 MB",
      format: "pytorch",
      status: "ready",
      createdAt: "2024-01-16T14:45:00Z",
    },
    {
      id: "model-v3-round30",
      version: "3.0.0",
      round: 30,
      accuracy: 0.94,
      loss: 0.15,
      size: "49.1 MB",
      format: "pytorch",
      status: "ready",
      createdAt: "2024-01-17T09:15:00Z",
    },
  ],
  "project-2": [
    {
      id: "model-v1-round5",
      version: "1.0.0",
      round: 5,
      accuracy: 0.82,
      loss: 0.28,
      size: "38.7 MB",
      format: "pytorch",
      status: "ready",
      createdAt: "2024-01-14T16:20:00Z",
    },
  ],
}

export async function GET(request: NextRequest, { params }: { params: { projectId: string } }) {
  try {
    const projectId = params.projectId

    // Get model versions for the project
    const versions = mockModelVersions[projectId as keyof typeof mockModelVersions] || []

    // Sort by creation date (newest first)
    const sortedVersions = versions.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())

    return NextResponse.json({
      projectId,
      versions: sortedVersions,
      totalVersions: sortedVersions.length,
    })
  } catch (error) {
    console.error("Error fetching model versions:", error)
    return NextResponse.json({ error: "Failed to fetch model versions" }, { status: 500 })
  }
}
