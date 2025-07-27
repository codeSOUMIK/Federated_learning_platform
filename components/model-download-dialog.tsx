"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Brain, CheckCircle, AlertCircle, Loader2, Copy } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface ModelVersion {
  id: string
  version: string
  round: number
  accuracy: number
  loss: number
  size: string
  createdAt: string
  format: "pytorch" | "onnx" | "tensorflow"
  status: "ready" | "processing" | "failed"
}

interface ModelDownloadDialogProps {
  projectId: string
  projectName: string
  isOpen: boolean
  onOpenChange: (open: boolean) => void
}

export default function ModelDownloadDialog({
  projectId,
  projectName,
  isOpen,
  onOpenChange,
}: ModelDownloadDialogProps) {
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [selectedVersion, setSelectedVersion] = useState<string>("")
  const [selectedFormat, setSelectedFormat] = useState<string>("pytorch")
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState(0)
  const { toast } = useToast()

  useEffect(() => {
    if (isOpen) {
      fetchModelVersions()
    }
  }, [isOpen, projectId])

  const fetchModelVersions = async () => {
    setLoading(true)
    try {
      const response = await fetch(`/api/models/${projectId}/versions`)
      if (response.ok) {
        const data = await response.json()
        setVersions(data.versions || [])
        if (data.versions && data.versions.length > 0) {
          setSelectedVersion(data.versions[0].id)
        }
      }
    } catch (error) {
      console.error("Failed to fetch model versions:", error)
      toast({
        title: "Error",
        description: "Failed to fetch model versions",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async () => {
    if (!selectedVersion) {
      toast({
        title: "Error",
        description: "Please select a model version to download",
        variant: "destructive",
      })
      return
    }

    setDownloading(true)
    setDownloadProgress(0)

    try {
      // Simulate download progress
      const progressInterval = setInterval(() => {
        setDownloadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + Math.random() * 10
        })
      }, 200)

      const response = await fetch(`/api/models/${projectId}/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          versionId: selectedVersion,
          format: selectedFormat,
        }),
      })

      clearInterval(progressInterval)
      setDownloadProgress(100)

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = `${projectName}-model-${selectedVersion}.${selectedFormat === "pytorch" ? "pth" : selectedFormat === "onnx" ? "onnx" : "pb"}`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)

        toast({
          title: "Download Complete",
          description: "Model has been downloaded successfully",
        })
      } else {
        throw new Error("Download failed")
      }
    } catch (error) {
      console.error("Download failed:", error)
      toast({
        title: "Download Failed",
        description: "Failed to download the model",
        variant: "destructive",
      })
    } finally {
      setDownloading(false)
      setTimeout(() => setDownloadProgress(0), 1000)
    }
  }

  const copyDownloadCommand = (version: ModelVersion) => {
    const command = `curl -X POST "${window.location.origin}/api/models/${projectId}/download" \\
  -H "Content-Type: application/json" \\
  -d '{"versionId": "${version.id}", "format": "${selectedFormat}"}' \\
  --output "${projectName}-model-${version.version}.${selectedFormat === "pytorch" ? "pth" : selectedFormat === "onnx" ? "onnx" : "pb"}"`

    navigator.clipboard.writeText(command)
    toast({
      title: "Copied",
      description: "Download command copied to clipboard",
    })
  }

  const selectedVersionData = versions.find((v) => v.id === selectedVersion)

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center">
            <Download className="h-5 w-5 mr-2" />
            Download Model
          </DialogTitle>
          <DialogDescription>
            Download trained models from project: <strong>{projectName}</strong>
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="download" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="download">Download Model</TabsTrigger>
            <TabsTrigger value="api">API Access</TabsTrigger>
          </TabsList>

          <TabsContent value="download" className="space-y-6">
            {loading ? (
              <div className="text-center py-8">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-blue-600" />
                <p className="text-gray-600">Loading model versions...</p>
              </div>
            ) : versions.length === 0 ? (
              <Card>
                <CardContent className="text-center py-8">
                  <Brain className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No Models Available</h3>
                  <p className="text-gray-600">
                    No trained models are available for download yet. Start training to generate models.
                  </p>
                </CardContent>
              </Card>
            ) : (
              <>
                {/* Model Selection */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Select Model Version</CardTitle>
                      <CardDescription>Choose which model version to download</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <label className="text-sm font-medium mb-2 block">Model Version</label>
                        <Select value={selectedVersion} onValueChange={setSelectedVersion}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select a version" />
                          </SelectTrigger>
                          <SelectContent>
                            {versions.map((version) => (
                              <SelectItem key={version.id} value={version.id}>
                                <div className="flex items-center justify-between w-full">
                                  <span>
                                    v{version.version} (Round {version.round})
                                  </span>
                                  <Badge variant="outline" className="ml-2">
                                    {(version.accuracy * 100).toFixed(1)}% acc
                                  </Badge>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <label className="text-sm font-medium mb-2 block">Export Format</label>
                        <Select value={selectedFormat} onValueChange={setSelectedFormat}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="pytorch">PyTorch (.pth)</SelectItem>
                            <SelectItem value="onnx">ONNX (.onnx)</SelectItem>
                            <SelectItem value="tensorflow">TensorFlow (.pb)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Model Details</CardTitle>
                      <CardDescription>Information about the selected model</CardDescription>
                    </CardHeader>
                    <CardContent>
                      {selectedVersionData ? (
                        <div className="space-y-3">
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Version:</span>
                            <span className="text-sm font-medium">v{selectedVersionData.version}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Training Round:</span>
                            <span className="text-sm font-medium">{selectedVersionData.round}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Accuracy:</span>
                            <span className="text-sm font-medium">
                              {(selectedVersionData.accuracy * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Loss:</span>
                            <span className="text-sm font-medium">{selectedVersionData.loss.toFixed(4)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">File Size:</span>
                            <span className="text-sm font-medium">{selectedVersionData.size}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Created:</span>
                            <span className="text-sm font-medium">
                              {new Date(selectedVersionData.createdAt).toLocaleDateString()}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Status:</span>
                            <Badge variant={selectedVersionData.status === "ready" ? "default" : "secondary"}>
                              {selectedVersionData.status === "ready" ? (
                                <CheckCircle className="h-3 w-3 mr-1" />
                              ) : (
                                <AlertCircle className="h-3 w-3 mr-1" />
                              )}
                              {selectedVersionData.status}
                            </Badge>
                          </div>
                        </div>
                      ) : (
                        <p className="text-sm text-gray-500">Select a model version to view details</p>
                      )}
                    </CardContent>
                  </Card>
                </div>

                {/* Download Progress */}
                {downloading && (
                  <Card>
                    <CardContent className="pt-6">
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Downloading model...</span>
                          <span>{Math.round(downloadProgress)}%</span>
                        </div>
                        <Progress value={downloadProgress} className="w-full" />
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Download Button */}
                <div className="flex justify-end space-x-3">
                  <Button variant="outline" onClick={() => onOpenChange(false)}>
                    Cancel
                  </Button>
                  <Button
                    onClick={handleDownload}
                    disabled={!selectedVersion || downloading}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    {downloading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Downloading...
                      </>
                    ) : (
                      <>
                        <Download className="h-4 w-4 mr-2" />
                        Download Model
                      </>
                    )}
                  </Button>
                </div>
              </>
            )}
          </TabsContent>

          <TabsContent value="api" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">API Download</CardTitle>
                <CardDescription>Use these API endpoints to download models programmatically</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Download Command</label>
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
                    <div className="flex justify-between items-start">
                      <pre className="flex-1 whitespace-pre-wrap">
                        {`curl -X POST "${window.location.origin}/api/models/${projectId}/download" \\
  -H "Content-Type: application/json" \\
  -d '{"versionId": "${selectedVersion}", "format": "${selectedFormat}"}' \\
  --output "${projectName}-model.${selectedFormat === "pytorch" ? "pth" : selectedFormat === "onnx" ? "onnx" : "pb"}"`}
                      </pre>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => selectedVersionData && copyDownloadCommand(selectedVersionData)}
                        className="ml-2"
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Available Endpoints</label>
                  <div className="space-y-2">
                    <div className="bg-gray-50 p-3 rounded border">
                      <code className="text-sm">GET /api/models/{projectId}/versions</code>
                      <p className="text-xs text-gray-600 mt-1">List all available model versions</p>
                    </div>
                    <div className="bg-gray-50 p-3 rounded border">
                      <code className="text-sm">POST /api/models/{projectId}/download</code>
                      <p className="text-xs text-gray-600 mt-1">Download a specific model version</p>
                    </div>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Response Format</label>
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
                    <pre>{`{
  "versions": [
    {
      "id": "version-123",
      "version": "1.0.0",
      "round": 10,
      "accuracy": 0.87,
      "loss": 0.23,
      "size": "45.2 MB",
      "format": "pytorch",
      "status": "ready",
      "createdAt": "2024-01-15T10:30:00Z"
    }
  ]
}`}</pre>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
