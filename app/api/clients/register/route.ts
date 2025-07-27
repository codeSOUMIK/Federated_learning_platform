import { type NextRequest, NextResponse } from "next/server"

// This endpoint is called by clients to register themselves
export async function POST(request: NextRequest) {
  try {
    const registrationData = await request.json()

    // Extract client information from the registration request
    const { name, capabilities, dataInfo, hardwareInfo, port = 8080 } = registrationData

    // Get client IP from request headers
    const forwarded = request.headers.get("x-forwarded-for")
    const ipAddress = forwarded ? forwarded.split(",")[0] : "localhost"

    // Validate required fields
    if (!name) {
      return NextResponse.json({ error: "Client name is required" }, { status: 400 })
    }

    // Create client registration payload
    const clientData = {
      name,
      ipAddress,
      port,
      capabilities: capabilities || {
        gpu: false,
        memory: "Unknown",
        cpuCores: 1,
      },
      dataInfo: dataInfo || {
        samples: 0,
        classes: 0,
        datasetType: "Unknown",
      },
      hardwareInfo: hardwareInfo || {
        cpu: "Unknown",
        gpu: "Unknown",
        platform: "Unknown",
      },
    }

    // Register the client using the main clients API
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000"}/api/clients`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(clientData),
    })

    if (!response.ok) {
      const error = await response.json()
      return NextResponse.json(error, { status: response.status })
    }

    const result = await response.json()

    return NextResponse.json({
      success: true,
      clientId: result.client.id,
      connectionKey: result.client.connectionKey,
      message: "Client registered successfully",
    })
  } catch (error) {
    console.error("Error in client registration:", error)
    return NextResponse.json({ error: "Failed to register client" }, { status: 500 })
  }
}
