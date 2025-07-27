"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Plus, Users, Search, AlertCircle, Cpu, HardDrive, Wifi, WifiOff, Activity, Loader2 } from "lucide-react"
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

interface AddClientToProjectDialogProps {
  projectId: string
  projectName: string
  onClientAdded?: () => void
}

export default function AddClientToProjectDialog({
  projectId,
  projectName,
  onClientAdded,
}: AddClientToProjectDialogProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [availableClients, setAvailableClients] = useState<Client[]>([])
  const [projectClients, setProjectClients] = useState<Client[]>([])
  const [selectedClients, setSelectedClients] = useState<string[]>([])
  const [searchTerm, setSearchTerm] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [loading, setLoading] = useState(false)
  const [adding, setAdding] = useState(false)
  const [activeTab, setActiveTab] = useState("add-clients"); // New state for active tab
  const { toast } = useToast()

  useEffect(() => {
    if (isOpen) {
      fetchClients()
      fetchProjectClients()
      setActiveTab("add-clients"); // Reset to add-clients tab when dialog opens
    }
  }, [isOpen, projectId])

  const fetchClients = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/clients")
      if (response.ok) {
        const data = await response.json()
        setAvailableClients(data.clients || [])
      }
    } catch (error) {
      console.error("Failed to fetch clients:", error)
      toast({
        title: "Error",
        description: "Failed to fetch available clients",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const fetchProjectClients = async () => {
    try {
      const response = await fetch(`/api/projects/${projectId}/clients`)
      if (response.ok) {
        const data = await response.json()
        setProjectClients(data.clients || [])
        console.log("Fetched project clients:", data.clients); // LOGGING
      }
    } catch (error) {
      console.error("Failed to fetch project clients:", error)
    }
  }

  const handleClientToggle = (clientId: string) => {
    setSelectedClients((prev) => (prev.includes(clientId) ? prev.filter((id) => id !== clientId) : [...prev, clientId]))
  }

  const handleAddClients = async () => {
    if (selectedClients.length === 0) return

    setAdding(true)
    try {
      const promises = selectedClients.map(async (clientId) => {
        console.log(`Attempting to add client ${clientId} to project ${projectId}`); // LOGGING
        const response = await fetch(`/api/projects/${projectId}/clients`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ clientId }),
        })
        const responseData = await response.json(); // LOGGING
        console.log(`Response for client ${clientId}:`, response.status, responseData); // LOGGING
        return { clientId, success: response.ok, responseData }
      })

      const results = await Promise.all(promises)
      const successful = results.filter((r) => r.success).length
      const failed = results.length - successful

      if (successful > 0) {
        toast({
          title: "Clients Added",
          description: `Successfully added ${successful} client${successful !== 1 ? "s" : ""} to the project!`,
        })
        setSelectedClients([])
        fetchProjectClients() // Re-fetch project clients to update the list
        onClientAdded?.()
        setActiveTab("current-clients"); // Switch to current clients tab
      }

      if (failed > 0) {
        toast({
          title: "Partial Success",
          description: `${failed} client${failed !== 1 ? "s" : ""} could not be added (may already be assigned). Check console for details.`,
          variant: "destructive",
        })
      }
    } catch (error) {
      console.error("Failed to add clients:", error)
      toast({
        title: "Error",
        description: "Failed to add clients to project",
        variant: "destructive",
      })
    } finally {
      setAdding(false)
    }
  }

  const handleRemoveClient = async (clientId: string) => {
    try {
      const response = await fetch(`/api/projects/${projectId}/clients?clientId=${clientId}`, {
        method: "DELETE",
      })

      if (response.ok) {
        toast({
          title: "Client Removed",
          description: "Client removed from project successfully!",
        })
        fetchProjectClients()
        onClientAdded?.()
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

  const filteredClients = availableClients.filter((client) => {
    const matchesSearch =
      client.name.toLowerCase().includes(searchTerm.toLowerCase()) || client.ipAddress.includes(searchTerm)
    const matchesStatus = statusFilter === "all" || client.status === statusFilter
    // Use client._id for comparison with projectClients
    const notInProject = !projectClients.some((pc) => pc._id === client._id)

    return matchesSearch && matchesStatus && notInProject
  })

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

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="bg-blue-50 hover:bg-blue-100 border-blue-200">
          <Plus className="h-4 w-4 mr-2" />
          Add Clients
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center">
            <Users className="h-5 w-5 mr-2" />
            Manage Project Clients
          </DialogTitle>
          <DialogDescription>
            Add or remove clients for project: <strong>{projectName}</strong>
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="add-clients">Add Clients</TabsTrigger>
            <TabsTrigger value="current-clients">Current Clients ({projectClients.length})</TabsTrigger>
          </TabsList>

          <TabsContent value="add-clients" className="space-y-6">
            {/* Search and Filter */}
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <Label htmlFor="search">Search Clients</Label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    id="search"
                    placeholder="Search by name or IP address..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10"
                  />
                </div>
              </div>
              <div className="w-full sm:w-48">
                <Label htmlFor="status-filter">Filter by Status</Label>
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger>
                    <SelectValue placeholder="All statuses" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Statuses</SelectItem>
                    <SelectItem value="online">Online</SelectItem>
                    <SelectItem value="offline">Offline</SelectItem>
                    <SelectItem value="training">Training</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Available Clients */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Available Clients ({filteredClients.length})</span>
                  <span className="text-sm font-normal text-gray-600">Selected: {selectedClients.length}</span>
                </CardTitle>
                <CardDescription>
                  Select clients to add to this project. Only clients not already assigned are shown.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-blue-600" />
                    <p className="text-gray-600">Loading available clients...</p>
                  </div>
                ) : filteredClients.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <Users className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>No available clients found matching your criteria.</p>
                    <p className="text-sm mt-2">Try adjusting your search or filter settings.</p>
                  </div>
                ) : (
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {filteredClients.map((client) => (
                      <div
                        key={client._id}
                        className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        <Checkbox
                          id={client._id}
                          checked={selectedClients.includes(client._id)}
                          onCheckedChange={() => handleClientToggle(client._id)}
                        />
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <label htmlFor={client._id} className="font-medium cursor-pointer block">
                                {client.name}
                              </label>
                              <p className="text-sm text-gray-600">
                                {client.ipAddress}:{client.port}
                              </p>
                              <div className="flex items-center space-x-4 mt-2">
                                <div className="flex items-center space-x-1 text-xs text-gray-500">
                                  <Cpu className="h-3 w-3" />
                                  <span>{client.hardwareInfo.cpu}</span>
                                </div>
                                <div className="flex items-center space-x-1 text-xs text-gray-500">
                                  <HardDrive className="h-3 w-3" />
                                  <span>{client.hardwareInfo.memory}</span>
                                </div>
                                <div className="text-xs text-gray-500">
                                  GPU: {client.capabilities.gpu ? "Yes" : "No"}
                                </div>
                              </div>
                            </div>
                            <div className="text-right space-y-2">
                              <Badge className={getStatusColor(client.status)}>
                                {getStatusIcon(client.status)}
                                <span className="ml-1">{client.status}</span>
                              </Badge>
                              <div className="text-xs text-gray-600">
                                {client.dataInfo.samples.toLocaleString()} samples
                              </div>
                              <div className="text-xs text-gray-500">{client.dataInfo.datasetType}</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3">
              <Button variant="outline" onClick={() => setIsOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleAddClients}
                disabled={selectedClients.length === 0 || adding}
                className="bg-blue-600 hover:bg-blue-700"
              >
                {adding ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Adding...
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4 mr-2" />
                    Add {selectedClients.length} Client{selectedClients.length !== 1 ? "s" : ""}
                  </>
                )}
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="current-clients" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Currently Assigned Clients</CardTitle>
                <CardDescription>
                  Clients currently assigned to this project. You can remove them if needed.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {projectClients.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <Users className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>No clients assigned to this project yet.</p>
                    <p className="text-sm mt-2">Switch to the "Add Clients" tab to assign clients.</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {projectClients.map((client) => (
                      <div
                        key={client._id}
                        className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex items-center space-x-4">
                          <div className="flex-1">
                            <div className="font-medium">{client.name}</div>
                            <p className="text-sm text-gray-600">
                              {client.ipAddress}:{client.port}
                            </p>
                            <div className="flex items-center space-x-4 mt-1">
                              <div className="flex items-center space-x-1 text-xs text-gray-500">
                                <Cpu className="h-3 w-3" />
                                <span>{client.hardwareInfo.cpu}</span>
                              </div>
                              <div className="flex items-center space-x-1 text-xs text-gray-500">
                                <HardDrive className="h-3 w-3" />
                                <span>{client.hardwareInfo.memory}</span>
                              </div>
                              <div className="text-xs text-gray-500">
                                GPU: {client.capabilities.gpu ? "Yes" : "No"}</div>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="text-right">
                            <Badge className={getStatusColor(client.status)}>
                              {getStatusIcon(client.status)}
                              <span className="ml-1">{client.status}</span>
                            </Badge>
                            <div className="text-xs text-gray-600 mt-1">
                              {client.dataInfo.samples.toLocaleString()} samples
                            </div>
                            <div className="text-xs text-gray-500">{client.dataInfo.datasetType}</div>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleRemoveClient(client._id)}
                            className="text-red-600 hover:text-red-700 hover:bg-red-50"
                          >
                            Remove
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
