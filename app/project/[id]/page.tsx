"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ArrowLeft, Activity, Users, Download, Settings, BarChart3, Clock, Brain, RefreshCw } from "lucide-react"
import Link from "next/link"
import TrainingStarter from "@/components/training-starter"
import ProjectClientManager from "@/components/project-client-manager"
import ModelDownloadDialog from "@/components/model-download-dialog"

export default function ProjectDetails({ params }: { params: { id: string } }) {
  const [project, setProject] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showDownloadDialog, setShowDownloadDialog] = useState(false)

  const fetchProject = async () => {
    console.log(`Attempting to fetch project with ID: ${params.id}`);
    try {
      // Fetch project by its MongoDB _id
      const response = await fetch(`/api/projects/${params.id}`);
      if (response.ok) {
        const data = await response.json();
        console.log("Project data fetched successfully:", data.project);
        setProject(data.project);
      } else {
        const errorData = await response.json();
        const errorMessage = errorData.error || "Project not found";
        console.error("Failed to fetch project:", response.status, errorMessage);
        throw new Error(errorMessage);
      }
    } catch (err) {
      console.error("Error in fetchProject:", err);
      setError(err instanceof Error ? err.message : "Failed to load project");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProject();

    // Refresh project data every 10 seconds
    const interval = setInterval(fetchProject, 10000);
    return () => clearInterval(interval);
  }, [params.id]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading project details...</p>
        </div>
      </div>
    )
  }

  if (error || !project) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 mb-4">{error || "Project not found"}</p>
          <Link href="/">
            <Button variant="outline">Back to Dashboard</Button>
          </Link>
        </div>
      </div>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "running":
        return "bg-green-100 text-green-800"
      case "completed":
        return "bg-blue-100 text-blue-800"
      case "paused":
        return "bg-yellow-100 text-yellow-800"
      case "failed":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
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
                <Brain className="h-8 w-8 text-blue-600 mr-3" />
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">{project.name}</h1>
                  <p className="text-sm text-gray-600">{project.description}</p>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Badge className={getStatusColor(project.status)}>{project.status}</Badge>
              <TrainingStarter projectId={project._id} projectName={project.name} />
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
              <Button variant="outline" size="sm" onClick={() => setShowDownloadDialog(true)}>
                <Download className="h-4 w-4 mr-2" />
                Download Model
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Progress</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{Math.round((project.currentRound / project.rounds) * 100)}%</div>
              <Progress value={(project.currentRound / project.rounds) * 100} className="mt-2" />
              <p className="text-xs text-muted-foreground mt-1">
                Round {project.currentRound} of {project.rounds}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Accuracy</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(project.accuracy * 100).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">Current model accuracy</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Loss</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{project.loss || 0.23}</div>
              <p className="text-xs text-muted-foreground">Current training loss</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Clients</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{project.clients}</div>
              <p className="text-xs text-muted-foreground">Participating clients</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Runtime</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">2h 34m</div>
              <p className="text-xs text-muted-foreground">Current session</p>
            </CardContent>
          </Card>
        </div>

        {/* Project Details */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="clients">Clients</TabsTrigger>
            <TabsTrigger value="configuration">Configuration</TabsTrigger>
            <TabsTrigger value="history">Training History</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Project Information</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Model Architecture:</span>
                    <span className="text-sm font-medium">{project.model}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Dataset Type:</span>
                    <span className="text-sm font-medium">{project.datasetType || "Medical Images"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Created:</span>
                    <span className="text-sm font-medium">{project.createdAt}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Status:</span>
                    <Badge className={getStatusColor(project.status)}>{project.status}</Badge>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Training Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Learning Rate:</span>
                    <span className="text-sm font-medium">{project.settings?.learningRate || 0.01}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Batch Size:</span>
                    <span className="text-sm font-medium">{project.settings?.batchSize || 32}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Min Clients:</span>
                    <span className="text-sm font-medium">{project.settings?.minClients || 2}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Aggregation:</span>
                    <span className="text-sm font-medium">{project.settings?.aggregation || "FedAvg"}</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="clients" className="space-y-6">
            <ProjectClientManager projectId={project._id} projectName={project.name} onProjectDataUpdated={fetchProject} />
          </TabsContent>

          <TabsContent value="configuration" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Model Configuration</CardTitle>
                <CardDescription>Current model and training parameters</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium">Model Architecture</label>
                      <p className="text-sm text-gray-600">{project.model}</p>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Training Rounds</label>
                      <p className="text-sm text-gray-600">{project.rounds}</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium">Learning Rate</label>
                      <p className="text-sm text-gray-600">{project.settings?.learningRate || 0.01}</p>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Batch Size</label>
                      <p className="text-sm text-gray-600">{project.settings?.batchSize || 32}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Training History</CardTitle>
                <CardDescription>Historical training performance and metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8 text-gray-500">
                  Training history will appear here once training begins.
                  <br />
                  <Link href="/training/demo-session">
                    <Button variant="outline" className="mt-4 bg-transparent">
                      View Demo Training Session
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="logs" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>System Logs</CardTitle>
                <CardDescription>Real-time system logs and events</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm h-64 overflow-y-auto">
                  <div>
                    [{new Date().toLocaleTimeString()}] INFO: Project {project.name} initialized
                  </div>
                  <div>
                    [{new Date().toLocaleTimeString()}] INFO: Model architecture: {project.model}
                  </div>
                  <div>[{new Date().toLocaleTimeString()}] INFO: Configuration loaded successfully</div>
                  <div>[{new Date().toLocaleTimeString()}] INFO: Waiting for training to begin...</div>
                  <div className="animate-pulse">
                    [{new Date().toLocaleTimeString()}] INFO: System ready for federated learning
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Model Download Dialog */}
      <ModelDownloadDialog
        projectId={project._id}
        projectName={project.name}
        isOpen={showDownloadDialog}
        onOpenChange={setShowDownloadDialog}
      />
    </div>
  )
}