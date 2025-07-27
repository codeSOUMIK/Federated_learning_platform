export interface Project {
  _id: string; // MongoDB ObjectId as string
  name: string
  description?: string
  model: string
  datasetType?: string
  status: "created" | "running" | "completed" | "paused" | "failed"
  clients: number
  rounds: number
  currentRound: number
  accuracy: number
  loss?: number
  createdAt: string
  settings?: {
    minClients: number
    learningRate: number
    batchSize?: number
    aggregation?: string
    privacy?: string
  }
  associatedClientIds?: string[]; // Array of client _ids
}

export class ProjectManager {
  static validateProject(project: Partial<Project>): string[] {
    const errors: string[] = []

    if (!project.name?.trim()) {
      errors.push("Project name is required")
    }

    if (!project.model) {
      errors.push("Model architecture is required")
    }

    if (!project.datasetType) {
      errors.push("Dataset type is required")
    }

    if (project.rounds && (project.rounds < 1 || project.rounds > 100)) {
      errors.push("Number of rounds must be between 1 and 100")
    }

    if (project.settings?.minClients && project.settings.minClients < 1) {
      errors.push("Minimum clients must be at least 1")
    }

    if (project.settings?.learningRate && (project.settings.learningRate <= 0 || project.settings.learningRate > 1)) {
      errors.push("Learning rate must be between 0 and 1")
    }

    return errors
  }

  // createProject will no longer generate an ID, MongoDB will do it
  static createProject(
    projectData: Omit<Project, "_id" | "createdAt" | "status" | "currentRound" | "accuracy" | "clients" | "associatedClientIds">,
  ): Omit<Project, "_id"> {
    return {
      ...projectData,
      status: "created",
      currentRound: 0,
      accuracy: 0,
      clients: 0,
      createdAt: new Date().toISOString().split("T")[0],
      associatedClientIds: [],
    }
  }

  // These static methods are less relevant now that persistence is handled by API routes directly with MongoDB
  // They might be removed or refactored later if not used.
  static updateProject(projects: Project[], updatedProject: Project): Project[] {
    const index = projects.findIndex(p => p._id === updatedProject._id);
    if (index !== -1) {
      projects[index] = updatedProject;
    }
    return projects;
  }

  static deleteProject(projects: Project[], id: string): Project[] {
    return projects.filter(p => p._id !== id);
  }

  static getProject(projects: Project[], id: string): Project | null {
    return projects.find(p => p._id === id) || null;
  }
}