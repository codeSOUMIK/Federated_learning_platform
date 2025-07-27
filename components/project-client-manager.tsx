"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Users, Activity, AlertCircle, Cpu, HardDrive, Trash2, Wifi, WifiOff, RefreshCw } from "lucide-react"
import AddClientToProjectDialog from "./add-client-to-project-dialog"
import { useToast } from "@/hooks/use-toast"

interface Client {
  _id: string; // MongoDB ObjectId as string
  id?: string; // Keep original id if it exists, but prefer _id
  name: string
  ipAddress: string
  port: number
  status: "online" | "offline" | "training"
  capabilities: {
    gpu: boolean
    memory: string
    cpu_cores: number
  }
  dataInfo: {
    samples: number
    classes: number
    datasetType: string
  }
  hardwareInfo: {
    cpu: string
    gpu: string
    memory: string
  }
  lastSeen: string
  createdAt: string
}

interface ProjectClientManagerProps {
  projectId: string
  projectName: string
}

export default function ProjectClientManager({ projectId, projectName }: ProjectClientManagerProps) {
  const [clients, setClients] = useState<Client[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [stats, setStats] = useState({
    total: 0,
    online: 0,
    training: 0,
    offline: 0,
    totalSamples: 0,
  })
  const { toast } = useToast()

  useEffect(() => {
    fetchProjectClients()

    // Set up auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchProjectClients(true)
    }, 30000)

    return () => clearInterval(interval)
  }, [projectId])

  const fetchProjectClients = async (isAutoRefresh = false) => {
    if (!isAutoRefresh) {
      setLoading(true)
    } else {
      setRefreshing(true)
    }

    try {
      const response = await fetch(`/api/projects/${projectId}/clients`)
      if (response.ok) {
        const data = await response.json()
        const clientList = data.clients || []
        setClients(clientList)

        // Calculate stats
        const stats = {
          total: clientList.length,
          online: clientList.filter((c: Client) => c.status === "online").length,
          training: clientList.filter((c: Client) => c.status === "training").length,
          offline: clientList.filter((c: Client) => c.status === "offline").length,
          totalSamples: clientList.reduce((sum: number, c: Client) => sum + c.dataInfo.samples, 0),
        }
        setStats(stats)
      } else {
        throw new Error("Failed to fetch project clients")
      }
    } catch (error) {
      console.error("Failed to fetch project clients:", error)
      if (!isAutoRefresh) {
        toast({
          title: "Error",
          description: "Failed to fetch project clients",
          variant: "destructive",
        })
      }
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  const handleRemoveClient = async (clientId: string, clientName: string) => {
    if (!confirm(`Are you sure you want to remove "${clientName}" from this project?`)) {
      return
    }

    try {
      const response = await fetch(`/api/projects/${projectId}/clients?clientId=${clientId}`, {
        method: "DELETE",
      })

      if (response.ok) {
        toast({
          title: "Client Removed",
          description: `${clientName} has been removed from the project`,
        })
        fetchProjectClients()
      } else {
        const error = await response.json()
        toast({
          title: "Error",
          description: error.error || "Failed to remove client from project",
          variant: "destructive",
        })
      }
    } catch (error) {
      console.error("Failed to remove client:", error)
      toast({
        title: "Error",
        description: "Failed to remove client from project",
        variant: "destructive",
      })
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "online":
        return "bg-green-100 text-green-800"
      case "training":
        return "bg-blue-100 text-blue-800"
      case "offline":
        return "bg-gray-100 text-gray-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "online":
        return <Wifi className="h-3 w-3" />
      case "training":
        return <Activity className="h-3 w-3" />
      case "offline":
        return <WifiOff className="h-3 w-3" />
      default:
        return <AlertCircle className="h-3 w-3" />
    }
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Users className="h-5 w-5 mr-2" />
            Project Clients
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading project clients...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Client Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-blue-600">{stats.total}</div>
            <p className="text-xs text-gray-600">Total Clients</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-green-600">{stats.online}</div>
            <p className="text-xs text-gray-600">Online</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-blue-600">{stats.training}</div>
            <p className="text-xs text-gray-600">Training</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-gray-600">{stats.offline}</div>
            <p className="text-xs text-gray-600">Offline</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-purple-600">{stats.totalSamples.toLocaleString()}</div>
            <p className="text-xs text-gray-600">Total Samples</p>
          </CardContent>
        </Card>
      </div>

      {/* Client Management */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center">
                <Users className="h-5 w-5 mr-2" />
                Project Clients ({clients.length})
                {refreshing && <RefreshCw className="h-4 w-4 ml-2 animate-spin text-blue-600" />}
              </CardTitle>
              <CardDescription>Manage clients assigned to this project</CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm" onClick={() => fetchProjectClients()} disabled={refreshing}>
                <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? "animate-spin" : ""}`} />
                Refresh
              </Button>
              <AddClientToProjectDialog
                projectId={projectId}
                projectName={projectName}
                onClientAdded={fetchProjectClients}
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {clients.length === 0 ? (
            <div className="text-center py-12">
              <Users className="h-16 w-16 mx-auto mb-4 text-gray-300" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Clients Assigned</h3>
              <p className="text-gray-600 mb-6">
                This project doesn't have any clients assigned yet. Add clients to start federated learning.
              </p>
              <AddClientToProjectDialog
                projectId={projectId}
                projectName={projectName}
                onClientAdded={fetchProjectClients}
              />
            </div>
          ) : (
            <div className="space-y-4">
              {clients.map((client) => (
                <div
                  key={client._id}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <div className="font-medium">{client.name}</div>
                        <Badge className={getStatusColor(client.status)}>
                          {getStatusIcon(client.status)}
                          <span className="ml-1">{client.status}</span>
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">
                        {client.ipAddress}:{client.port}
                      </p>
                      <div className="flex items-center space-x-6 mt-2">
                        <div className="flex items-center space-x-1 text-xs text-gray-500">
                          <Cpu className="h-3 w-3" />
                          <span>{client.hardwareInfo.cpu}</span>
                        </div>
                        <div className="flex items-center space-x-1 text-xs text-gray-500">
                          <HardDrive className="h-3 w-3" />
                          <span>{client.hardwareInfo.memory}</span>
                        </div>
                        <div className="text-xs text-gray-500">GPU: {client.capabilities.gpu ? "Yes" : "No"}</div>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <div className="text-sm font-medium">{client.dataInfo.samples.toLocaleString()} samples</div>
                      <div className="text-xs text-gray-600">{client.dataInfo.datasetType}</div>
                      <div className="text-xs text-gray-500 mt-1">
                        Last seen: {new Date(client.lastSeen).toLocaleString()}
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleRemoveClient(client._id, client.name)}
                      className="text-red-600 hover:text-red-700 hover:bg-red-50"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}