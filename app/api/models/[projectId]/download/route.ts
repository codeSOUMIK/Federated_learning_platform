import { type NextRequest, NextResponse } from "next/server"

// Mock model data
const mockModels = {
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
  ],
}

export async function POST(request: NextRequest, { params }: { params: { projectId: string } }) {
  try {
    const { versionId, format } = await request.json()
    const projectId = params.projectId

    if (!versionId || !format) {
      return NextResponse.json({ error: "Version ID and format are required" }, { status: 400 })
    }

    // Find the model version
    const projectModels = mockModels[projectId as keyof typeof mockModels] || []
    const modelVersion = projectModels.find((m) => m.id === versionId)

    if (!modelVersion) {
      return NextResponse.json({ error: "Model version not found" }, { status: 404 })
    }

    // Simulate model file generation based on format
    let fileContent: string
    let mimeType: string
    let fileExtension: string

    switch (format) {
      case "pytorch":
        fileContent = "# PyTorch Model File\n# This is a simulated model file\nmodel_state_dict = {...}"
        mimeType = "application/octet-stream"
        fileExtension = "pth"
        break
      case "onnx":
        fileContent = "# ONNX Model File\n# This is a simulated ONNX model\nonnx_model = {...}"
        mimeType = "application/octet-stream"
        fileExtension = "onnx"
        break
      case "tensorflow":
        fileContent = "# TensorFlow Model File\n# This is a simulated TensorFlow model\ntf_model = {...}"
        mimeType = "application/octet-stream"
        fileExtension = "pb"
        break
      default:
        return NextResponse.json({ error: "Unsupported format" }, { status: 400 })
    }

    // Create a blob with the simulated model content
    const blob = new Blob([fileContent], { type: mimeType })

    // Convert blob to array buffer for response
    const buffer = await blob.arrayBuffer()

    return new NextResponse(buffer, {
      status: 200,
      headers: {
        "Content-Type": mimeType,
        "Content-Disposition": `attachment; filename="model-${versionId}.${fileExtension}"`,
        "Content-Length": buffer.byteLength.toString(),
      },
    })
  } catch (error) {
    console.error("Error downloading model:", error)
    return NextResponse.json({ error: "Failed to download model" }, { status: 500 })
  }
}
