"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Activity,
  Brain,
  Users,
  Server,
  Plus,
  Play,
  Pause,
  BarChart3,
  Settings,
  Clock,
  CheckCircle,
  AlertCircle,
} from "lucide-react"
import Link from "next/link"
import { useRealTimeClients, useRealTimeProjects } from "@/hooks/use-real-time-data"
export default function HomePage() {
  const { clients, loading: clientsLoading, error: clientsError } = useRealTimeClients();
  const { projects, loading: projectsLoading, error: projectsError } = useRealTimeProjects();

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return <Activity className="h-4 w-4 text-green-500" />
      case "completed":
        return <CheckCircle className="h-4 w-4 text-blue-500" />
      case "paused":
        return <Pause className="h-4 w-4 text-yellow-500" />
      case "failed":
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
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
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">Federated Learning Platform</h1>
            </div>
            <div className="flex items-center space-x-4">
              <Link href="/create-project">
                <Button>
                  <Plus className="h-4 w-4 mr-2" />
                  New Project
                </Button>
              </Link>
              
              <Link href="/settings">
                <Button variant="outline">
                  <Settings className="h-4 w-4 mr-2" />
                  Settings
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Projects</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{projects.length}</div>
              <p className="text-xs text-muted-foreground">Currently training</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Clients</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{clients.length}</div>
              <p className="text-xs text-muted-foreground">Across all projects</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Completed Projects</CardTitle>
              <CheckCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{projects.filter((p) => p.status === "completed").length}</div>
              <p className="text-xs text-muted-foreground">Successfully finished</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Global Model Accuracy</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {((projects.reduce((sum, p) => sum + p.accuracy, 0) / projects.length) * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground">Across all models</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="projects" className="space-y-6">
          <TabsList>
            <TabsTrigger value="projects">Projects</TabsTrigger>
            <TabsTrigger value="clients">Clients</TabsTrigger>
            <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          </TabsList>

          <TabsContent value="projects" className="space-y-6">
            <div className="grid gap-6">
              {projects.map((project) => (
                <Card key={project._id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(project.status)}
                        <div>
                          <CardTitle className="text-lg">{project.name}</CardTitle>
                          <CardDescription>
                            Model: {project.model} • Created: {project.createdAt}
                          </CardDescription>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge className={getStatusColor(project.status)}>{project.status}</Badge>
                        <Link href={`/project/${project._id}`}>
                          <Button variant="outline" size="sm">
                            View Details
                          </Button>
                        </Link>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      <div className="space-y-2">
                        <p className="text-sm font-medium">Progress</p>
                        <Progress value={(project.currentRound / project.rounds) * 100} className="w-full" />
                        <p className="text-xs text-muted-foreground">
                          Round {project.currentRound} of {project.rounds}
                        </p>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm font-medium">Clients</p>
                        <div className="flex items-center">
                          <Users className="h-4 w-4 mr-2 text-muted-foreground" />
                          <span className="text-lg font-semibold">{project.clients}</span>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm font-medium">Current Accuracy</p>
                        <div className="flex items-center">
                          <BarChart3 className="h-4 w-4 mr-2 text-muted-foreground" />
                          <span className="text-lg font-semibold">{(project.accuracy * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm font-medium">Actions</p>
                        <div className="flex space-x-2">
                          {project.status === "running" ? (
                            <Button size="sm" variant="outline">
                              <Pause className="h-4 w-4 mr-1" />
                              Pause
                            </Button>
                          ) : (
                            <Button
                              size="sm"
                              onClick={async () => {
                                try {
                                  const response = await fetch('/api/flower', {
                                    method: 'POST',
                                    headers: {
                                      'Content-Type': 'application/json',
                                    },
                                    body: JSON.stringify({ numRounds: project.rounds, numClients: project.clients, projectId: project._id }),
                                  });
                                  const data = await response.json();
                                  console.log(data);
                                  // Optionally, redirect to a training session page or update UI
                                  window.location.href = `/training/${project._id}`;
                                } catch (error) {
                                  console.error(error);
                                  alert("Failed to start training. Check console for details.");
                                }
                              }}
                            >
                              <Play className="h-4 w-4 mr-1" />
                              Start Training
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="clients" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Client Management</CardTitle>
                <CardDescription>Manage federated learning clients and their configurations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8">
                  <Server className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Client Management</h3>
                  <p className="text-muted-foreground mb-4">Configure and monitor your federated learning clients</p>
                  <Link href="/clients">
                    <Button>
                      <Users className="h-4 w-4 mr-2" />
                      Manage Clients
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="monitoring" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Training Monitoring</CardTitle>
                <CardDescription>Real-time monitoring of federated learning training progress</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8">
                  <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Training Metrics</h3>
                  <p className="text-muted-foreground mb-4">
                    Monitor accuracy, loss, and other training metrics in real-time
                  </p>
                  <Link href="/monitoring">
                    <Button>
                      <Activity className="h-4 w-4 mr-2" />
                      View Monitoring
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
        </div>
    </div>
  )
}
