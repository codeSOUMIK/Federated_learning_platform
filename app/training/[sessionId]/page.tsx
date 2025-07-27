'use client'

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Terminal, CheckCircle, XCircle, Loader2 } from 'lucide-react';

interface TrainingLog {
  clientId: string;
  log: string;
  timestamp: string;
  error?: boolean;
}

interface TrainingSession {
  sessionId: string;
  projectId: string;
  clientIds: string[];
  config: any;
  status: string;
  startTime: string;
  endTime?: string;
  logs: TrainingLog[];
}

export default function TrainingSessionPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;
  const [session, setSession] = useState<TrainingSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (sessionId) {
      const fetchSession = async () => {
        try {
          setLoading(true);
          const response = await fetch(`/api/training/session/${sessionId}`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          setSession(data);
        } catch (e: any) {
          setError(e.message);
        } finally {
          setLoading(false);
        }
      };
      fetchSession();

      // Optional: Poll for updates if session is still running
      const interval = setInterval(() => {
        if (session && (session.status === 'running' || session.status === 'pending')) {
          fetchSession();
        }
      }, 5000); // Poll every 5 seconds

      return () => clearInterval(interval);
    }
  }, [sessionId, session]); // Added session to dependency array for polling logic

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin mr-2" /> Loading Training Session...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen text-red-500">
        <XCircle className="h-8 w-8 mr-2" /> Error: {error}
      </div>
    );
  }

  if (!session) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        No training session found for ID: {sessionId}
      </div>
    );
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'running':
        return <Badge variant="secondary" className="bg-blue-500 text-white">Running</Badge>;
      case 'completed':
        return <Badge variant="default" className="bg-green-500 text-white">Completed</Badge>;
      case 'completed_with_errors':
        return <Badge variant="destructive">Completed with Errors</Badge>;
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6 flex items-center gap-2">
        <Terminal className="h-8 w-8" /> Training Session: {sessionId.substring(0, 8)}...
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader>
            <CardTitle>Session Details</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            <p><strong>Project ID:</strong> {session.projectId}</p>
            <p><strong>Status:</strong> {getStatusBadge(session.status)}</p>
            <p><strong>Start Time:</strong> {new Date(session.startTime).toLocaleString()}</p>
            {session.endTime && <p><strong>End Time:</strong> {new Date(session.endTime).toLocaleString()}</p>}
            <p><strong>Clients:</strong> {session.clientIds.length}</p>
            <p><strong>Dataset Type:</strong> {session.config.datasetType}</p>
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Simulation Logs</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[500px] w-full rounded-md border p-4 font-mono text-sm bg-gray-900 text-green-400">
              {session.logs.length === 0 && <p className="text-gray-500">No logs yet...</p>}
              {session.logs.map((entry, index) => (
                <div key={index} className={`mb-2 ${entry.error ? 'text-red-400' : ''}`}>
                  <span className="text-gray-500">[{new Date(entry.timestamp).toLocaleTimeString()}]</span>
                  <span className="font-bold"> {entry.clientId}:</span>
                  <pre className="whitespace-pre-wrap">{entry.log}</pre>
                </div>
              ))}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}