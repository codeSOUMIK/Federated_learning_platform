import { NextResponse } from 'next/server';
import { runFlowerServer } from '../../../lib/flower';

export async function POST(req: Request) {
  try {
    const { numRounds, numClients, projectId } = await req.json();
    await runFlowerServer(numRounds, numClients, projectId);
    return NextResponse.json({ message: 'Flower server started' });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
