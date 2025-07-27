import { NextResponse } from 'next/server';
import clientPromise from '@/lib/mongodb';

export async function GET(
  request: Request,
  { params }: { params: { sessionId: string } }
) {
  try {
    const { sessionId } = params;

    if (!sessionId) {
      return NextResponse.json({ error: 'Session ID is required' }, { status: 400 });
    }

    const client = await clientPromise;
    const db = client.db("fed_learning");
    const sessionsCollection = db.collection("trainingSessions");

    const session = await sessionsCollection.findOne({ sessionId });

    if (!session) {
      return NextResponse.json({ error: 'Training session not found' }, { status: 404 });
    }

    return NextResponse.json(session);
  } catch (error) {
    console.error('Failed to fetch training session:', error);
    return NextResponse.json({ error: 'Failed to fetch training session' }, { status: 500 });
  }
}
