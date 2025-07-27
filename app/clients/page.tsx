"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  ArrowLeft,
  Users,
  Server,
  Activity,
  Clock,
  HardDrive,
  Cpu,
  MemoryStick,
  Wifi,
  WifiOff,
  CheckCircle,
  AlertCircle,
  Trash2,
  RefreshCw,
  Search,
} from "lucide-react"
import Link from "next/link"
import { ClientRegistrationForm } from "@/components/client-registration-form"
import { useToast } from "@/hooks/use-toast"

interface Client {
  id: string
  name: string
  ipAddress: string
  port: number
  status: "online" | "offline" | "training" | "error"
  lastSeen: string
  capabilities: {
    gpu: boolean
    memory: string
    cpuCores: number
  }
  dataInfo: {
    samples: number
    classes: number
    datasetType: string
  }
  performance: {
    avgAccuracy: number
    avgLoss: number
    totalRounds: number
    successfulRounds: number
  }
  hardwareInfo: {
    cpu: string
    gpu: string
    platform: string
  }
  registeredAt: string
  connectionKey?: string
}

export default function ClientManagement() {
  const [clients, setClients] = useState<Client[]>([])
  const [selectedClient, setSelectedClient] = useState<Client | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const { toast } = useToast()

  const fetchClients = async () => {
    try {
      const response = await fetch("/api/clients")
      if (!response.ok) throw new Error("Failed to fetch clients")

      const data = await response.json()
      setClients(data.clients)
    } catch (error) {
      console.error("Error fetching clients:", error)
      toast({
        title: "Error",
        description: "Failed to fetch clients",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchClients()

    // Set up auto-refresh every 10 seconds
    const interval = setInterval(fetchClients, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleDeleteClient = async (clientId: string) => {
    try {
      const response = await fetch(`/api/clients?id=${clientId}`, {
        method: "DELETE",
      })

      if (!response.ok) throw new Error("Failed to delete client")

      setClients(clients.filter((client) => client.id !== clientId))
      toast({
        title: "Client Deleted",
        description: "Client has been successfully removed",
      })
    } catch (error) {
      console.error("Error deleting client:", error)
      toast({
        title: "Error",
        description: "Failed to delete client",
        variant: "destructive",
      })
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "online":
        return <Wifi className="h-4 w-4 text-green-500" />
      case "training":
        return <Activity className="h-4 w-4 text-blue-500" />
      case "offline":
        return <WifiOff className="h-4 w-4 text-gray-500" />
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
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
      case "error":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  const getOverallStats = () => {
    const totalClients = clients.length
    const onlineClients = clients.filter((c) => c.status === "online").length
    const trainingClients = clients.filter((c) => c.status === "training").length
    const totalSamples = clients.reduce((sum, c) => sum + c.dataInfo.samples, 0)
    const avgAccuracy =
      totalClients > 0 ? clients.reduce((sum, c) => sum + c.performance.avgAccuracy, 0) / totalClients : 0

    return {
      totalClients,
      onlineClients,
      trainingClients,
      offlineClients: totalClients - onlineClients - trainingClients,
      totalSamples,
      avgAccuracy,
    }
  }

  const filteredClients = clients.filter((client) => {
    const matchesSearch =
      client.name.toLowerCase().includes(searchTerm.toLowerCase()) || client.ipAddress.includes(searchTerm)
    const matchesStatus = statusFilter === "all" || client.status === statusFilter
    return matchesSearch && matchesStatus
  })

  const stats = getOverallStats()

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Loading clients...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-6">
            <div className="flex items-center">
              <Link href="/" className="flex items-center text-gray-600 hover:text-gray-900 mr-4">
                <ArrowLeft className="h-5 w-5 mr-2" />
                Back to Dashboard
              </Link>
              <div className="flex items-center">
                <Users className="h-8 w-8 text-blue-600 mr-3" />
                <h1 className="text-2xl font-bold text-gray-900">Client Management</h1>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="outline" onClick={fetchClients}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
              <ClientRegistrationForm onClientAdded={fetchClients} />
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Clients</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalClients}</div>
              <p className="text-xs text-muted-foreground">Registered clients</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Online</CardTitle>
              <Wifi className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{stats.onlineClients}</div>
              <p className="text-xs text-muted-foreground">Available for training</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Training</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">{stats.trainingClients}</div>
              <p className="text-xs text-muted-foreground">Currently training</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Samples</CardTitle>
              <HardDrive className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalSamples.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground">Across all clients</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Accuracy</CardTitle>
              <CheckCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(stats.avgAccuracy * 100).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">Overall performance</p>
            </CardContent>
          </Card>
        </div>

        {/* Search and Filter */}
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex items-center space-x-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
                  <Input
                    placeholder="Search clients by name or IP address..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10"
                  />
                </div>
              </div>
              <div className="flex space-x-2">
                <Button
                  variant={statusFilter === "all" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setStatusFilter("all")}
                >
                  All
                </Button>
                <Button
                  variant={statusFilter === "online" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setStatusFilter("online")}
                >
                  Online
                </Button>
                <Button
                  variant={statusFilter === "training" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setStatusFilter("training")}
                >
                  Training
                </Button>
                <Button
                  variant={statusFilter === "offline" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setStatusFilter("offline")}
                >
                  Offline
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Client List */}
        <Card>
          <CardHeader>
            <CardTitle>Federated Learning Clients</CardTitle>
            <CardDescription>
              Manage and monitor your federated learning participants ({filteredClients.length} of {clients.length}{" "}
              clients)
            </CardDescription>
          </CardHeader>
          <CardContent>
            {filteredClients.length === 0 ? (
              <div className="text-center py-8">
                <Server className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No clients found</h3>
                <p className="text-gray-500 mb-4">
                  {clients.length === 0
                    ? "Get started by registering your first federated learning client."
                    : "No clients match your current search criteria."}
                </p>
                {clients.length === 0 && <ClientRegistrationForm onClientAdded={fetchClients} />}
              </div>
            ) : (
              <div className="space-y-4">
                {filteredClients.map((client) => (
                  <div key={client.id} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <Server className="h-8 w-8 text-gray-400" />
                        <div>
                          <h3 className="font-semibold text-lg">{client.name}</h3>
                          <p className="text-sm text-gray-600">
                            {client.ipAddress}:{client.port} • Last seen: {new Date(client.lastSeen).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          <div className="flex items-center space-x-2">
                            {getStatusIcon(client.status)}
                            <Badge className={getStatusColor(client.status)}>{client.status}</Badge>
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button variant="outline" size="sm" onClick={() => setSelectedClient(client)}>
                                View Details
                              </Button>
                            </DialogTrigger>
                            <DialogContent className="max-w-4xl">
                              <DialogHeader>
                                <DialogTitle>{client.name}</DialogTitle>
                                <DialogDescription>
                                  Detailed client information and performance metrics
                                </DialogDescription>
                              </DialogHeader>
                              {selectedClient && (
                                <Tabs defaultValue="overview" className="w-full">
                                  <TabsList className="grid w-full grid-cols-4">
                                    <TabsTrigger value="overview">Overview</TabsTrigger>
                                    <TabsTrigger value="hardware">Hardware</TabsTrigger>
                                    <TabsTrigger value="data">Data</TabsTrigger>
                                    <TabsTrigger value="performance">Performance</TabsTrigger>
                                  </TabsList>

                                  <TabsContent value="overview" className="space-y-4">
                                    <div className="grid grid-cols-2 gap-4">
                                      <Card>
                                        <CardHeader>
                                          <CardTitle className="text-sm">Connection Info</CardTitle>
                                        </CardHeader>
                                        <CardContent className="space-y-2">
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">IP Address:</span>
                                            <span className="text-sm font-medium">
                                              {selectedClient.ipAddress}:{selectedClient.port}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Status:</span>
                                            <Badge className={getStatusColor(selectedClient.status)}>
                                              {selectedClient.status}
                                            </Badge>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Last Seen:</span>
                                            <span className="text-sm font-medium">
                                              {new Date(selectedClient.lastSeen).toLocaleString()}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Registered:</span>
                                            <span className="text-sm font-medium">
                                              {new Date(selectedClient.registeredAt).toLocaleString()}
                                            </span>
                                          </div>
                                        </CardContent>
                                      </Card>

                                      <Card>
                                        <CardHeader>
                                          <CardTitle className="text-sm">Quick Stats</CardTitle>
                                        </CardHeader>
                                        <CardContent className="space-y-2">
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Total Rounds:</span>
                                            <span className="text-sm font-medium">
                                              {selectedClient.performance.totalRounds}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Success Rate:</span>
                                            <span className="text-sm font-medium">
                                              {selectedClient.performance.totalRounds > 0
                                                ? (
                                                    (selectedClient.performance.successfulRounds /
                                                      selectedClient.performance.totalRounds) *
                                                    100
                                                  ).toFixed(1)
                                                : 0}
                                              %
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Avg Accuracy:</span>
                                            <span className="text-sm font-medium">
                                              {(selectedClient.performance.avgAccuracy * 100).toFixed(1)}%
                                            </span>
                                          </div>
                                        </CardContent>
                                      </Card>
                                    </div>
                                  </TabsContent>

                                  <TabsContent value="hardware" className="space-y-4">
                                    <div className="grid grid-cols-1 gap-4">
                                      <Card>
                                        <CardHeader>
                                          <CardTitle className="text-sm flex items-center">
                                            <Cpu className="h-4 w-4 mr-2" />
                                            Hardware Specifications
                                          </CardTitle>
                                        </CardHeader>
                                        <CardContent className="space-y-3">
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">CPU:</span>
                                            <span className="text-sm font-medium">
                                              {selectedClient.hardwareInfo.cpu}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">GPU:</span>
                                            <span className="text-sm font-medium">
                                              {selectedClient.hardwareInfo.gpu}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Memory:</span>
                                            <span className="text-sm font-medium">
                                              {selectedClient.capabilities.memory}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">CPU Cores:</span>
                                            <span className="text-sm font-medium">
                                              {selectedClient.capabilities.cpuCores}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Platform:</span>
                                            <span className="text-sm font-medium">
                                              {selectedClient.hardwareInfo.platform}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">GPU Available:</span>
                                            <Badge variant={selectedClient.capabilities.gpu ? "default" : "secondary"}>
                                              {selectedClient.capabilities.gpu ? "Yes" : "No"}
                                            </Badge>
                                          </div>
                                        </CardContent>
                                      </Card>
                                    </div>
                                  </TabsContent>

                                  <TabsContent value="data" className="space-y-4">
                                    <Card>
                                      <CardHeader>
                                        <CardTitle className="text-sm flex items-center">
                                          <HardDrive className="h-4 w-4 mr-2" />
                                          Dataset Information
                                        </CardTitle>
                                      </CardHeader>
                                      <CardContent className="space-y-3">
                                        <div className="flex justify-between">
                                          <span className="text-sm text-gray-600">Dataset Type:</span>
                                          <span className="text-sm font-medium">
                                            {selectedClient.dataInfo.datasetType}
                                          </span>
                                        </div>
                                        <div className="flex justify-between">
                                          <span className="text-sm text-gray-600">Total Samples:</span>
                                          <span className="text-sm font-medium">
                                            {selectedClient.dataInfo.samples.toLocaleString()}
                                          </span>
                                        </div>
                                        <div className="flex justify-between">
                                          <span className="text-sm text-gray-600">Number of Classes:</span>
                                          <span className="text-sm font-medium">{selectedClient.dataInfo.classes}</span>
                                        </div>
                                      </CardContent>
                                    </Card>
                                  </TabsContent>

                                  <TabsContent value="performance" className="space-y-4">
                                    <div className="grid grid-cols-2 gap-4">
                                      <Card>
                                        <CardHeader>
                                          <CardTitle className="text-sm">Training Performance</CardTitle>
                                        </CardHeader>
                                        <CardContent className="space-y-2">
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Avg Accuracy:</span>
                                            <span className="text-sm font-medium">
                                              {(selectedClient.performance.avgAccuracy * 100).toFixed(2)}%
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Avg Loss:</span>
                                            <span className="text-sm font-medium">
                                              {selectedClient.performance.avgLoss.toFixed(4)}
                                            </span>
                                          </div>
                                        </CardContent>
                                      </Card>

                                      <Card>
                                        <CardHeader>
                                          <CardTitle className="text-sm">Training History</CardTitle>
                                        </CardHeader>
                                        <CardContent className="space-y-2">
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Total Rounds:</span>
                                            <span className="text-sm font-medium">
                                              {selectedClient.performance.totalRounds}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Successful:</span>
                                            <span className="text-sm font-medium text-green-600">
                                              {selectedClient.performance.successfulRounds}
                                            </span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-sm text-gray-600">Failed:</span>
                                            <span className="text-sm font-medium text-red-600">
                                              {selectedClient.performance.totalRounds -
                                                selectedClient.performance.successfulRounds}
                                            </span>
                                          </div>
                                        </CardContent>
                                      </Card>
                                    </div>
                                  </TabsContent>
                                </Tabs>
                              )}
                            </DialogContent>
                          </Dialog>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleDeleteClient(client.id)}
                            className="text-red-600 hover:text-red-700"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>

                    {/* Quick info row */}
                    <div className="mt-3 grid grid-cols-4 gap-4 text-sm">
                      <div className="flex items-center">
                        <MemoryStick className="h-4 w-4 mr-2 text-gray-400" />
                        <span className="text-gray-600">Memory: {client.capabilities.memory}</span>
                      </div>
                      <div className="flex items-center">
                        <Cpu className="h-4 w-4 mr-2 text-gray-400" />
                        <span className="text-gray-600">Cores: {client.capabilities.cpuCores}</span>
                      </div>
                      <div className="flex items-center">
                        <HardDrive className="h-4 w-4 mr-2 text-gray-400" />
                        <span className="text-gray-600">Samples: {client.dataInfo.samples.toLocaleString()}</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="h-4 w-4 mr-2 text-gray-400" />
                        <span className="text-gray-600">
                          Accuracy: {(client.performance.avgAccuracy * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
