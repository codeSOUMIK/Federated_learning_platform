"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Checkbox } from "@/components/ui/checkbox"
import { Brain, Loader2, ArrowLeft } from "lucide-react"
import Link from "next/link"
import { useToast } from "@/hooks/use-toast"

export default function CreateProject() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const { toast } = useToast()

  const [formData, setFormData] = useState({
    name: "",
    description: "",
    model: "",
    datasetType: "",
    rounds: 10,
    settings: {
      minClients: 2,
      learningRate: 0.01,
      batchSize: 32,
      aggregation: "FedAvg",
      privacy: "None",
    },
  })

  const handleInputChange = (field: string, value: any) => {
    if (field.includes(".")) {
      const [parent, child] = field.split(".")
      setFormData((prev) => ({
        ...prev,
        [parent]: {
          ...prev[parent as keyof typeof prev],
          [child]: value,
        },
      }))
    } else {
      setFormData((prev) => ({
        ...prev,
        [field]: value,
      }))
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      const response = await fetch("/api/projects", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.errors ? errorData.errors.join(", ") : "Failed to create project")
      }

      const result = await response.json()
      toast({
        title: "Project Created",
        description: `${result.project.name} has been successfully created.`,
      })

      // Redirect to the new project's detail page using its MongoDB _id
      router.push(`/project/${result.project._id}`)
    } catch (error) {
      console.error("Error creating project:", error)
      toast({
        title: "Project Creation Failed",
        description: error instanceof Error ? error.message : "Failed to create project",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
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
                <h1 className="text-2xl font-bold text-gray-900">Create New Project</h1>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Card>
          <CardHeader>
            <CardTitle>Project Details</CardTitle>
            <CardDescription>Define the parameters for your federated learning project.</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Basic Information */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Project Name *</Label>
                  <Input
                    id="name"
                    placeholder="e.g., Medical Image Classification"
                    value={formData.name}
                    onChange={(e) => handleInputChange("name", e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Input
                    id="description"
                    placeholder="Brief description of the project"
                    value={formData.description}
                    onChange={(e) => handleInputChange("description", e.target.value)}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="model">Model Architecture *</Label>
                  <Select
                    value={formData.model}
                    onValueChange={(value) => handleInputChange("model", value)}
                    required
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select model architecture" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="CNN">Convolutional Neural Network (CNN)</SelectItem>
                      <SelectItem value="ResNet">ResNet</SelectItem>
                      <SelectItem value="MobileNet">MobileNet</SelectItem>
                      <SelectItem value="Custom">Custom Model</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="datasetType">Dataset Type *</Label>
                  <Select
                    value={formData.datasetType}
                    onValueChange={(value) => handleInputChange("datasetType", value)}
                    required
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select dataset type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Medical Images">Medical Images</SelectItem>
                      <SelectItem value="Text Data">Text Data</SelectItem>
                      <SelectItem value="Tabular Data">Tabular Data</SelectItem>
                      <SelectItem value="Time Series">Time Series</SelectItem>
                      <SelectItem value="Audio Data">Audio Data</SelectItem>
                      <SelectItem value="Other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Training Parameters */}
              <div className="space-y-2">
                <Label htmlFor="rounds">Training Rounds: {formData.rounds}</Label>
                <Slider
                  id="rounds"
                  min={1}
                  max={100}
                  step={1}
                  value={[formData.rounds]}
                  onValueChange={(value) => handleInputChange("rounds", value[0])}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="minClients">Minimum Clients: {formData.settings.minClients}</Label>
                  <Slider
                    id="minClients"
                    min={1}
                    max={10}
                    step={1}
                    value={[formData.settings.minClients]}
                    onValueChange={(value) => handleInputChange("settings.minClients", value[0])}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="learningRate">Learning Rate: {formData.settings.learningRate}</Label>
                  <Slider
                    id="learningRate"
                    min={0.0001}
                    max={0.1}
                    step={0.0001}
                    value={[formData.settings.learningRate]}
                    onValueChange={(value) => handleInputChange("settings.learningRate", value[0])}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="batchSize">Batch Size</Label>
                  <Select
                    value={formData.settings.batchSize.toString()}
                    onValueChange={(value) => handleInputChange("settings.batchSize", Number.parseInt(value))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select batch size" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="16">16</SelectItem>
                      <SelectItem value="32">32</SelectItem>
                      <SelectItem value="64">64</SelectItem>
                      <SelectItem value="128">128</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="aggregation">Aggregation Strategy</Label>
                  <Select
                    value={formData.settings.aggregation}
                    onValueChange={(value) => handleInputChange("settings.aggregation", value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select aggregation strategy" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="FedAvg">Federated Averaging (FedAvg)</SelectItem>
                      <SelectItem value="FedProx">Federated Proximal (FedProx)</SelectItem>
                      <SelectItem value="FedOpt">Federated Optimization (FedOpt)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="privacy">Privacy Mechanism</Label>
                <Select
                  value={formData.settings.privacy}
                  onValueChange={(value) => handleInputChange("settings.privacy", value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select privacy mechanism" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="None">None</SelectItem>
                    <SelectItem value="DP">Differential Privacy (DP)</SelectItem>
                    <SelectItem value="Homomorphic Encryption">Homomorphic Encryption</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex justify-end">
                <Button type="submit" disabled={isLoading}>
                  {isLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Creating Project...
                    </>
                  ) : (
                    "Create Project"
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}