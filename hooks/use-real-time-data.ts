"use client"

import { useState, useEffect, useCallback } from "react"

export function useRealTimeClients() {
  const [clients, setClients] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchClients = useCallback(async () => {
    try {
      const response = await fetch("/api/clients")
      if (!response.ok) throw new Error("Failed to fetch clients")
      const data = await response.json()
      setClients(data.clients)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchClients()

    // Poll for updates every 5 seconds
    const interval = setInterval(fetchClients, 5000)

    return () => clearInterval(interval)
  }, [fetchClients])

  return { clients, loading, error, refetch: fetchClients }
}

export function useRealTimeProjects() {
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchProjects = useCallback(async () => {
    try {
      const response = await fetch("/api/projects")
      if (!response.ok) throw new Error("Failed to fetch projects")
      const data = await response.json()
      setProjects(data.projects)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchProjects()

    // Poll for updates every 10 seconds
    const interval = setInterval(fetchProjects, 10000)

    return () => clearInterval(interval)
  }, [fetchProjects])

  const createProject = async (projectData: any) => {
    try {
      const response = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(projectData),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to create project")
      }

      const data = await response.json()
      await fetchProjects() // Refresh the list
      return data.project
    } catch (err) {
      throw err
    }
  }

  return { projects, loading, error, refetch: fetchProjects, createProject }
}

export function useTrainingSession(sessionId: string | null) {
  const [progress, setProgress] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!sessionId) return

    setLoading(true)

    const fetchProgress = async () => {
      try {
        const response = await fetch(`/api/training/status/${sessionId}`)
        if (response.ok) {
          const data = await response.json()
          setProgress(data.progress)
        }
      } catch (err) {
        console.error("Failed to fetch training progress:", err)
      }
    }

    fetchProgress()
    const interval = setInterval(fetchProgress, 2000) // Update every 2 seconds

    return () => {
      clearInterval(interval)
      setLoading(false)
    }
  }, [sessionId])

  return { progress, loading }
}
