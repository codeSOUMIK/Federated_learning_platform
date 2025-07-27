import { type NextRequest, NextResponse } from "next/server"

// Mock clients data - in production, this would come from a database
const clients = [
  {
    id: "client-1",
    name: "Hospital A - Main Server",
    ipAddress: "192.168.1.100",
    port: 8081,
    status: "online",
    lastSeen: new Date().toISOString(),
  },
]

export async function GET(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const clientId = params.id
    const client = clients.find((c) => c.id === clientId)

    if (!client) {
      return NextResponse.json({ error: "Client not found" }, { status: 404 })
    }

    return NextResponse.json({ client })
  } catch (error) {
    return NextResponse.json({ error: "Failed to get client status" }, { status: 500 })
  }
}

export async function PUT(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const clientId = params.id
    const updateData = await request.json()

    const clientIndex = clients.findIndex((c) => c.id === clientId)

    if (clientIndex === -1) {
      return NextResponse.json({ error: "Client not found" }, { status: 404 })
    }

    // Update client status
    clients[clientIndex] = {
      ...clients[clientIndex],
      ...updateData,
      lastSeen: new Date().toISOString(),
    }

    return NextResponse.json({ client: clients[clientIndex] })
  } catch (error) {
    return NextResponse.json({ error: "Failed to update client status" }, { status: 500 })
  }
}
