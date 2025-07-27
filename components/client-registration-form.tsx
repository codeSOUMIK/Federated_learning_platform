import type React from "react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Plus, Server, Loader2, CheckCircle, AlertCircle, Copy, X } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface ClientRegistrationFormProps {
  onClientAdded: () => void;
}

interface ClientFormData {
  name: string;
  ipAddress: string;
  port: string;
  description: string;
  capabilities: {
    gpu: boolean;
    memory: string;
    cpuCores: string;
  };
  dataInfo: {
    samples: string;
    classes: string;
    datasetType: string;
  };
  hardwareInfo: {
    cpu: string;
    gpu: string;
    platform: string;
  };
}

const initialClientFormData: ClientFormData = {
  name: "",
  ipAddress: "",
  port: "8080",
  description: "",
  capabilities: {
    gpu: false,
    memory: "",
    cpuCores: "",
  },
  dataInfo: {
    samples: "",
    classes: "",
    datasetType: "",
  },
  hardwareInfo: {
    cpu: "",
    gpu: "",
    platform: "",
  },
};

export function ClientRegistrationForm({ onClientAdded }: ClientRegistrationFormProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [clientForms, setClientForms] = useState<ClientFormData[]>([initialClientFormData]);
  const [registrationResults, setRegistrationResults] = useState<any[] | null>(null);
  const { toast } = useToast();

  const handleAddClientForm = () => {
    setClientForms((prev) => [...prev, { ...initialClientFormData, id: `temp-${Date.now()}` }]);
  };

  const handleRemoveClientForm = (index: number) => {
    setClientForms((prev) => prev.filter((_, i) => i !== index));
  };

  const handleInputChange = (index: number, field: string, value: any) => {
    setClientForms((prev) =>
      prev.map((form, i) => {
        if (i === index) {
          if (field.includes(".")) {
            const [parent, child] = field.split(".");
            return {
              ...form,
              [parent]: {
                ...(form as any)[parent],
                [child]: value,
              },
            };
          } else {
            return {
              ...form,
              [field]: value,
            };
          }
        }
        return form;
      })
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setRegistrationResults(null);

    const clientsToRegister = clientForms.map((formData) => ({
      name: formData.name,
      ipAddress: formData.ipAddress,
      port: Number.parseInt(formData.port),
      description: formData.description,
      capabilities: {
        gpu: formData.capabilities.gpu,
        memory: formData.capabilities.memory || "Unknown",
        cpuCores: Number.parseInt(formData.capabilities.cpuCores) || 0,
      },
      dataInfo: {
        samples: Number.parseInt(formData.dataInfo.samples) || 0,
        classes: Number.parseInt(formData.dataInfo.classes) || 0,
        datasetType: formData.dataInfo.datasetType || "Unknown",
      },
      hardwareInfo: {
        cpu: formData.hardwareInfo.cpu || "Unknown",
        gpu: formData.hardwareInfo.gpu || "Unknown",
        platform: formData.hardwareInfo.platform || "Unknown",
      },
    }));

    try {
      const response = await fetch("/api/clients", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(clientsToRegister),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Failed to register clients");
      }

      const result = await response.json();
      setRegistrationResults(result.results);

      const successfulRegistrations = result.results.filter((r: any) => r.success).length;
      const failedRegistrations = result.results.filter((r: any) => !r.success).length;

      if (successfulRegistrations > 0) {
        toast({
          title: "Registration Complete",
          description: `${successfulRegistrations} client(s) registered successfully. ${failedRegistrations} failed.`,
        });
        onClientAdded(); // Refresh client list in parent component
      } else {
        toast({
          title: "Registration Failed",
          description: "No clients were registered. Check errors for details.",
          variant: "destructive",
        });
      }

      // Reset forms after submission
      setClientForms([initialClientFormData]);
    } catch (error: any) {
      console.error("Error registering clients:", error);
      toast({
        title: "Registration Failed",
        description: error instanceof Error ? error.message : "Failed to register clients",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const renderClientForm = (formData: ClientFormData, index: number) => (
    <Card key={index} className="relative">
      {clientForms.length > 1 && (
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-2 right-2 text-gray-400 hover:text-red-500"
          onClick={() => handleRemoveClientForm(index)}
        >
          <X className="h-4 w-4" />
        </Button>
      )}
      <CardHeader>
        <CardTitle className="text-lg">Client {index + 1} Information</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Basic Information */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor={`name-${index}`}>Client Name *</Label>
            <Input
              id={`name-${index}`}
              placeholder="e.g., Hospital A - Main Server"
              value={formData.name}
              onChange={(e) => handleInputChange(index, "name", e.target.value)}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={`description-${index}`}>Description</Label>
            <Input
              id={`description-${index}`}
              placeholder="Brief description of the client"
              value={formData.description}
              onChange={(e) => handleInputChange(index, "description", e.target.value)}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor={`ipAddress-${index}`}>IP Address *</Label>
            <Input
              id={`ipAddress-${index}`}
              placeholder="192.168.1.100"
              value={formData.ipAddress}
              onChange={(e) => handleInputChange(index, "ipAddress", e.target.value)}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={`port-${index}`}>Port</Label>
            <Input
              id={`port-${index}`}
              placeholder="8080"
              value={formData.port}
              onChange={(e) => handleInputChange(index, "port", e.target.value)}
            />
          </div>
        </div>

        {/* Hardware Capabilities */}
        <h4 className="font-semibold mt-6 mb-2">Hardware Capabilities</h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor={`cpu-${index}`}>CPU</Label>
            <Input
              id={`cpu-${index}`}
              placeholder="Intel Xeon E5-2680"
              value={formData.hardwareInfo.cpu}
              onChange={(e) => handleInputChange(index, "hardwareInfo.cpu", e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={`memory-${index}`}>Memory</Label>
            <Input
              id={`memory-${index}`}
              placeholder="16GB"
              value={formData.capabilities.memory}
              onChange={(e) => handleInputChange(index, "capabilities.memory", e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={`cpuCores-${index}`}>CPU Cores</Label>
            <Input
              id={`cpuCores-${index}`}
              type="number"
              placeholder="8"
              value={formData.capabilities.cpuCores}
              onChange={(e) => handleInputChange(index, "capabilities.cpuCores", e.target.value)}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor={`gpu-${index}`}>GPU</Label>
            <Input
              id={`gpu-${index}`}
              placeholder="NVIDIA RTX 3080"
              value={formData.hardwareInfo.gpu}
              onChange={(e) => handleInputChange(index, "hardwareInfo.gpu", e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={`platform-${index}`}>Platform</Label>
            <Select
              value={formData.hardwareInfo.platform}
              onValueChange={(value) => handleInputChange(index, "hardwareInfo.platform", value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select platform" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Linux Ubuntu 20.04">Linux Ubuntu 20.04</SelectItem>
                <SelectItem value="Linux Ubuntu 22.04">Linux Ubuntu 22.04</SelectItem>
                <SelectItem value="Linux CentOS 8">Linux CentOS 8</SelectItem>
                <SelectItem value="Windows 10">Windows 10</SelectItem>
                <SelectItem value="Windows 11">Windows 11</SelectItem>
                <SelectItem value="macOS">macOS</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <Checkbox
            id={`gpu-available-${index}`}
            checked={formData.capabilities.gpu}
            onCheckedChange={(checked) => handleInputChange(index, "capabilities.gpu", checked)}
          />
          <Label htmlFor={`gpu-available-${index}`}>GPU Available for Training</Label>
        </div>

        {/* Dataset Information */}
        <h4 className="font-semibold mt-6 mb-2">Dataset Information</h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor={`datasetType-${index}`}>Dataset Type</Label>
            <Select
              value={formData.dataInfo.datasetType}
              onValueChange={(value) => handleInputChange(index, "dataInfo.datasetType", value)}
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
          <div className="space-y-2">
            <Label htmlFor={`samples-${index}`}>Number of Samples</Label>
            <Input
              id={`samples-${index}`}
              type="number"
              placeholder="1000"
              value={formData.dataInfo.samples}
              onChange={(e) => handleInputChange(index, "dataInfo.samples", e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={`classes-${index}`}>Number of Classes</Label>
            <Input
              id={`classes-${index}`}
              type="number"
              placeholder="10"
              value={formData.dataInfo.classes}
              onChange={(e) => handleInputChange(index, "dataInfo.classes", e.target.value)}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          Register New Client
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center">
            <Server className="h-5 w-5 mr-2" />
            Register Federated Learning Client(s)
          </DialogTitle>
          <DialogDescription>Add one or more clients to participate in federated learning training sessions</DialogDescription>
        </DialogHeader>

        {registrationResults ? (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center text-green-600">
                <CheckCircle className="h-5 w-5 mr-2" />
                Registration Summary
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {registrationResults.map((result, idx) => (
                <div key={idx} className="border rounded-lg p-3">
                  {result.success ? (
                    <p className="text-green-600">
                      <CheckCircle className="inline-block h-4 w-4 mr-2" />
                      Client "{result.client.name}" registered successfully! ID: {result.client._id}
                    </p>
                  ) : (
                    <p className="text-red-600">
                      <AlertCircle className="inline-block h-4 w-4 mr-2" />
                      Failed to register client: {result.error}
                    </p>
                  )}
                </div>
              ))}
              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={() => setIsOpen(false)}>
                  Close
                </Button>
                <Button
                  onClick={() => {
                    setRegistrationResults(null);
                    setClientForms([initialClientFormData]); // Reset forms
                  }}
                >
                  Register More Clients
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            {clientForms.map((formData, index) => renderClientForm(formData, index))}

            <div className="flex justify-between items-center">
              <Button type="button" variant="outline" onClick={handleAddClientForm}>
                <Plus className="h-4 w-4 mr-2" />
                Add Another Client
              </Button>
              <div className="flex space-x-2">
                <Button type="button" variant="outline" onClick={() => setIsOpen(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={isLoading}>
                  {isLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Registering...
                    </>
                  ) : (
                    "Register Clients"
                  )}
                </Button>
              </div>
            </div>
          </form>
        )}
      </DialogContent>
    </Dialog>
  );
}