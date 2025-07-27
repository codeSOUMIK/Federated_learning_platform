"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Play, Users, Cpu, Wifi, Database, AlertTriangle, CheckCircle, Clock, Zap, RefreshCw } from "lucide-react"
import { useRouter } from "next/navigation"

interface Client {
  _id: string; // MongoDB ObjectId as string
  id?: string; // Keep original id if it exists, but prefer _id
  name: string
  ipAddress: string
  port: number
  status: "online" | "offline" | "training"
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
  hardwareInfo: {
    cpu: string
    gpu: string
    platform: string
  }
  performance?: {
    avgAccuracy: number
    avgLoss: number
    totalRounds: number
    successfulRounds: number
  }
}

interface TrainingConfig {
  rounds: number
  learningRate: number
  batchSize: number
  minClients: number
  aggregationStrategy: string
  clientFraction: number
  convergenceThreshold: number
  datasetType: string // Added for simulated client data
}

interface PreflightCheck {
  readinessScore: number
  issues: string[]
  recommendations: string[]
  clientChecks: {
    [clientId: string]: {
      online: boolean
      networkLatency: number
      resourceCompatible: boolean
      dataCompatible: boolean
    }
  }
  estimatedTrainingTime: number
  totalSamples: number
}

export default function TrainingStarter({ projectId }: { projectId: string }) {
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [clients, setClients] = useState<Client[]>([])
  const [selectedClients, setSelectedClients] = useState<string[]>([])
  const [config, setConfig] = useState<TrainingConfig>({
    rounds: 10,
    learningRate: 0.01,
    batchSize: 32,
    minClients: 2,
    aggregationStrategy: "fedavg",
    clientFraction: 1.0,
    convergenceThreshold: 0.001,
    datasetType: "mnist", // Default dataset type
  })
  const [preflightCheck, setPreflightCheck] = useState<PreflightCheck | null>(null)
  const [clientFilter, setClientFilter] = useState<"all" | "online" | "high-performance">("all")
  const [refreshing, setRefreshing] = useState(false)

  // Load project clients
  useEffect(() => {
    if (open) {
      loadProjectClients()
    }
  }, [open, projectId])

  // Auto-refresh clients every 30 seconds
  useEffect(() => {
    if (!open) return

    const interval = setInterval(() => {
      loadProjectClients(true)
    }, 30000)

    return () => clearInterval(interval)
  }, [open])

  const loadProjectClients = async (silent = false) => {
    if (!silent) setRefreshing(true)

    try {
      const response = await fetch(`/api/projects/${projectId}/clients`)
      if (response.ok) {
        const data = await response.json()
        setClients(data.clients || [])
      }
    } catch (error) {
      console.error("Error loading clients:", error)
    } finally {
      if (!silent) setRefreshing(false)
    }
  }

  const calculateClientPerformance = (client: Client): number => {
    let score = 50 // Base score

    // Hardware bonuses
    if (client.capabilities.gpu) score += 30
    if (client.capabilities.cpuCores >= 4) score += 10
    if (client.capabilities.memory.includes("16GB") || client.capabilities.memory.includes("32GB")) score += 10

    // Status bonus
    if (client.status === "online") score += 20

    // Performance history
    if (client.performance) {
      const successRate = client.performance.successfulRounds / Math.max(client.performance.totalRounds, 1)
      score += successRate * 20
      if (client.performance.avgAccuracy > 0.8) score += 10
    }

    // Network quality (based on last seen)
    const lastSeenTime = new Date(client.lastSeen).getTime()
    const now = Date.now()
    const minutesAgo = (now - lastSeenTime) / (1000 * 60)
    if (minutesAgo < 5) score += 10
    else if (minutesAgo > 60) score -= 20

    return Math.min(100, Math.max(0, score))
  }

  const getFilteredClients = () => {
    let filtered = clients

    switch (clientFilter) {
      case "online":
        filtered = clients.filter((c) => c.status === "online")
        break
      case "high-performance":
        filtered = clients.filter((c) => calculateClientPerformance(c) >= 70)
        break
      default:
        break
    }

    return filtered
  }

  const runPreflightCheck = async () => {
    if (selectedClients.length === 0) return

    setLoading(true)
    try {
      const response = await fetch("/api/training/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          projectId,
          clientIds: selectedClients,
          config,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setPreflightCheck(data)
      }
    } catch (error) {
      console.error("Preflight check failed:", error)
    } finally {
      setLoading(false)
    }
  }

  const startTraining = async () => {
    if (!preflightCheck || preflightCheck.readinessScore < 70) {
      return
    }

    setLoading(true)
    try {
      const response = await fetch("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          projectId,
          clientIds: selectedClients,
          config: {
            ...config, // Ensure all config properties including datasetType are sent
          },
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setOpen(false)
        router.push(`/training/${data.sessionId}`)
      }
    } catch (error) {
      console.error("Failed to start training:", error)
    } finally {
      setLoading(false)
    }
  }

  const selectAllOnline = () => {
    const onlineClients = clients.filter((c) => c.status === "online").map((c) => c._id)
    setSelectedClients(onlineClients)
  }

  const clearSelection = () => {
    setSelectedClients([])
  }

  const toggleClientSelection = (clientId: string) => {
    setSelectedClients((prev) => (prev.includes(clientId) ? prev.filter((id) => id !== clientId) : [...prev, clientId]))
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "online":
        return <Wifi className="h-4 w-4 text-green-500" />
      case "training":
        return <Zap className="h-4 w-4 text-blue-500" />
      default:
        return <Wifi className="h-4 w-4 text-gray-400" />
    }
  }

  const getStatusBadge = (status: string) => {
    const variants = {
      online: "default",
      training: "secondary",
      offline: "outline",
    } as const

    return <Badge variant={variants[status as keyof typeof variants] || "outline"}>{status}</Badge>
  }

  const filteredClients = getFilteredClients()
  const selectedClientData = clients.filter((c) => selectedClients.includes(c._id))
  const onlineClients = clients.filter((c) => c.status === "online")

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="lg" className="gap-2">
          <Play className="h-4 w-4" />
          Start Training
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Play className="h-5 w-5" />
            Start Federated Learning Training
          </DialogTitle>
          <DialogDescription>
            Configure training parameters and select clients to participate in federated learning.
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="clients" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="clients">Client Selection</TabsTrigger>
            <TabsTrigger value="config">Training Config</TabsTrigger>
            <TabsTrigger value="preflight">Pre-flight Check</TabsTrigger>
          </TabsList>

          <TabsContent value="clients" className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Label>Filter Clients:</Label>
                <Select value={clientFilter} onValueChange={(value: any) => setClientFilter(value)}>
                  <SelectTrigger className="w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Clients ({clients.length})</SelectItem>
                    <SelectItem value="online">Online Only ({onlineClients.length})</SelectItem>
                    <SelectItem value="high-performance">High Performance</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={selectAllOnline}>
                  Select All Online
                </Button>
                <Button variant="outline" size="sm" onClick={clearSelection}>
                  Clear Selection
                </Button>
                <Button variant="outline" size="sm" onClick={() => loadProjectClients()} disabled={refreshing}>
                  <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
                </Button>
              </div>
            </div>

            {selectedClients.length > 0 && (
              <Alert>
                <Users className="h-4 w-4" />
                <AlertDescription>{selectedClients.length} clients selected for training</AlertDescription>
              </Alert>
            )}

            <div className="grid gap-3 max-h-96 overflow-y-auto">
              {filteredClients.map((client) => {
                const isSelected = selectedClients.includes(client._id)
                const performanceScore = calculateClientPerformance(client)

                return (
                  <Card
                    key={client._id}
                    className={`cursor-pointer transition-colors ${isSelected ? "ring-2 ring-primary" : ""}`}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <Checkbox checked={isSelected} onCheckedChange={() => toggleClientSelection(client._id)} />
                          <div className="flex items-center gap-2">
                            {getStatusIcon(client.status)}
                            <div>
                              <div className="font-medium">{client.name}</div>
                              <div className="text-sm text-muted-foreground">
                                {client.ipAddress}:{client.port}
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary">Performance: {performanceScore}%</Badge>
                          {getStatusBadge(client.status)}
                        </div>
                      </div>

                      <div className="mt-3 grid grid-cols-3 gap-4 text-sm">
                        <div className="flex items-center gap-1">
                          <Cpu className="h-3 w-3" />
                          <span>{client.capabilities.cpuCores} cores</span>
                          {client.capabilities.gpu && (
                            <Badge variant="outline" className="ml-1 text-xs">
                              GPU
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-1">
                          <Database className="h-3 w-3" />
                          <span>{client.dataInfo.samples.toLocaleString()} samples</span>
                        </div>
                        <div className="text-muted-foreground">{client.capabilities.memory}</div>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>

            {filteredClients.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                No clients match the current filter criteria.
              </div>
            )}
          </TabsContent>

          <TabsContent value="config" className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="rounds">Training Rounds</Label>
                  <Input
                    id="rounds"
                    type="number"
                    value={config.rounds}
                    onChange={(e) => setConfig((prev) => ({ ...prev, rounds: Number.parseInt(e.target.value) || 1 }))}
                    min="1"
                    max="100"
                  />
                </div>

                <div>
                  <Label htmlFor="learningRate">Learning Rate</Label>
                  <Input
                    id="learningRate"
                    type="number"
                    step="0.001"
                    value={config.learningRate}
                    onChange={(e) =>
                      setConfig((prev) => ({ ...prev, learningRate: Number.parseFloat(e.target.value) || 0.01 }))
                    }
                    min="0.001"
                    max="1"
                  />
                </div>

                <div>
                  <Label htmlFor="batchSize">Batch Size</Label>
                  <Select
                    value={config.batchSize.toString()}
                    onValueChange={(value) => setConfig((prev) => ({ ...prev, batchSize: Number.parseInt(value) }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="16">16</SelectItem>
                      <SelectItem value="32">32</SelectItem>
                      <SelectItem value="64">64</SelectItem>
                      <SelectItem value="128">128</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="minClients">Minimum Clients</Label>
                  <Input
                    id="minClients"
                    type="number"
                    value={config.minClients}
                    onChange={(e) =>
                      setConfig((prev) => ({ ...prev, minClients: Number.parseInt(e.target.value) || 1 }))
                    }
                    min="1"
                    max={clients.length}
                  />
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <Label htmlFor="aggregation">Aggregation Strategy</Label>
                  <Select
                    value={config.aggregationStrategy}
                    onValueChange={(value) => setConfig((prev) => ({ ...prev, aggregationStrategy: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fedavg">FedAvg (Federated Averaging)</SelectItem>
                      <SelectItem value="fedprox">FedProx (Proximal)</SelectItem>
                      <SelectItem value="fedopt">FedOpt (Optimized)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Client Participation: {Math.round(config.clientFraction * 100)}%</Label>
                  <Slider
                    value={[config.clientFraction]}
                    onValueChange={([value]) => setConfig((prev) => ({ ...prev, clientFraction: value }))}
                    min={0.1}
                    max={1}
                    step={0.1}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label htmlFor="convergence">Convergence Threshold</Label>
                  <Input
                    id="convergence"
                    type="number"
                    step="0.0001"
                    value={config.convergenceThreshold}
                    onChange={(e) =>
                      setConfig((prev) => ({
                        ...prev,
                        convergenceThreshold: Number.parseFloat(e.target.value) || 0.001,
                      }))
                    }
                    min="0.0001"
                    max="0.1"
                  />
                </div>

                <div>
                  <Label htmlFor="datasetType">Dataset Type</Label>
                  <Select
                    value={config.datasetType}
                    onValueChange={(value) => setConfig((prev) => ({ ...prev, datasetType: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a dataset type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="mnist">MNIST</SelectItem>
                      <SelectItem value="cifar10">CIFAR-10</SelectItem>
                      <SelectItem value="fashion_mnist">Fashion MNIST</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>

            <Separator />

            <div className="space-y-2">
              <h4 className="font-medium">Training Estimation</h4>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Estimated Time</div>
                  <div className="font-medium">
                    {selectedClients.length > 0
                      ? `~${Math.round(config.rounds * 2 + selectedClients.length * 0.5)} minutes`
                      : "Select clients first"}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Total Samples</div>
                  <div className="font-medium">
                    {selectedClientData.reduce((sum, client) => sum + client.dataInfo.samples, 0).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Expected Accuracy</div>
                  <div className="font-medium">{selectedClients.length > 0 ? "85-92%" : "N/A"}</div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="preflight" className="space-y-4">
            {!preflightCheck ? (
              <div className="text-center py-8">
                <Button onClick={runPreflightCheck} disabled={selectedClients.length === 0 || loading}>
                  {loading ? "Running Checks..." : "Run Pre-flight Check"}
                </Button>
                <p className="text-sm text-muted-foreground mt-2">
                  {selectedClients.length === 0
                    ? "Select clients first to run pre-flight checks"
                    : "Verify system readiness before starting training"}
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {preflightCheck.readinessScore >= 80 ? (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      ) : (
                        <AlertTriangle className="h-5 w-5 text-yellow-500" />
                      )}
                      System Readiness: {preflightCheck.readinessScore}%
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Progress value={preflightCheck.readinessScore} className="mb-4" />

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">Estimated Training Time</div>
                        <div className="font-medium flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {Math.round(preflightCheck.estimatedTrainingTime)} minutes
                        </div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Total Training Samples</div>
                        <div className="font-medium flex items-center gap-1">
                          <Database className="h-3 w-3" />
                          {preflightCheck.totalSamples.toLocaleString()}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {preflightCheck.issues.length > 0 && (
                  <Alert>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="font-medium mb-2">Issues Detected:</div>
                      <ul className="list-disc list-inside space-y-1">
                        {preflightCheck.issues.map((issue, index) => (
                          <li key={index} className="text-sm">
                            {issue}
                          </li>
                        ))}
                      </ul>
                    </AlertDescription>
                  </Alert>
                )}

                {preflightCheck.recommendations.length > 0 && (
                  <Alert>
                    <CheckCircle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="font-medium mb-2">Recommendations:</div>
                      <ul className="list-disc list-inside space-y-1">
                        {preflightCheck.recommendations.map((rec, index) => (
                          <li key={index} className="text-sm">
                            {rec}
                          </li>
                        ))}
                      </ul>
                    </AlertDescription>
                  </Alert>
                )}

                <div className="grid gap-2">
                  <h4 className="font-medium">Client Status</h4>
                  {Object.entries(preflightCheck.clientChecks).map(([clientId, check]) => {
                    const client = clients.find((c) => c._id === clientId) // Use _id here
                    if (!client) return null

                    return (
                      <div key={clientId} className="flex items-center justify-between p-2 border rounded">
                        <span className="font-medium">{client.name}</span>
                        <div className="flex items-center gap-2">
                          {check.online && <Badge variant="outline">Online</Badge>}
                          {check.resourceCompatible && <Badge variant="outline">Compatible</Badge>}
                          {check.dataCompatible && <Badge variant="outline">Data OK</Badge>}
                          <span className="text-sm text-muted-foreground">{check.networkLatency}ms</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={startTraining}
            disabled={selectedClients.length === 0 || loading || (preflightCheck && preflightCheck.readinessScore < 70)}
          >
            {loading ? "Starting..." : "Start Training"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}