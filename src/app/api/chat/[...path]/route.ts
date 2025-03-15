import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const resolvedParams = await params;
  const pathSegments = resolvedParams.path;
  return handleApiRequest(request, pathSegments, 'GET');
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const resolvedParams = await params;
  const pathSegments = resolvedParams.path;
  return handleApiRequest(request, pathSegments, 'POST');
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const resolvedParams = await params;
  const pathSegments = resolvedParams.path;
  return handleApiRequest(request, pathSegments, 'PUT');
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const resolvedParams = await params;
  const pathSegments = resolvedParams.path;
  return handleApiRequest(request, pathSegments, 'PATCH');
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const resolvedParams = await params;
  const pathSegments = resolvedParams.path;
  return handleApiRequest(request, pathSegments, 'DELETE');
}

async function handleApiRequest(
  request: NextRequest,
  pathSegments: string[],
  method: string
) {
  try {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    const path = pathSegments.join('/');
    const url = new URL(`${backendUrl}/api/v1/chat/${path}`);
    
    // Log the URL for debugging
    console.log(`Forwarding ${method} request to: ${url.toString()}`);
    
    // Add query parameters
    const searchParams = new URL(request.url).searchParams;
    searchParams.forEach((value, key) => {
      url.searchParams.append(key, value);
    });

    // Get request body if it exists
    let body = null;
    if (method !== 'GET' && method !== 'HEAD') {
      try {
        body = await request.json();
      } catch (e) {
        // No body or not JSON
      }
    }

    // Forward the request to the backend
    const headers = new Headers();
    
    // Always copy authorization header
    const authHeader = request.headers.get('authorization');
    if (authHeader) {
      headers.set('authorization', authHeader);
    }
    
    // Set content type for JSON requests
    if (body) {
      headers.set('content-type', 'application/json');
    }

    const backendResponse = await fetch(url.toString(), {
      method,
      headers,
      body: body ? JSON.stringify(body) : null,
    });

    // Handle streaming responses
    if (backendResponse.headers.get('content-type')?.includes('text/event-stream')) {
      const { readable, writable } = new TransformStream();
      backendResponse.body?.pipeTo(writable);
      return new NextResponse(readable, {
        headers: {
          'content-type': 'text/event-stream',
          'cache-control': 'no-cache',
          'connection': 'keep-alive',
        },
      });
    }

    // Handle regular responses
    const data = await backendResponse.text();
    console.log(`Backend response status: ${backendResponse.status}`);
    console.log(`Backend response data: ${data}`);
    
    const responseHeaders = new Headers();
    
    // Copy content type
    const contentType = backendResponse.headers.get('content-type');
    if (contentType) {
      responseHeaders.set('content-type', contentType);
    }

    return new NextResponse(data, {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error('API proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to proxy request to backend' },
      { status: 500 }
    );
  }
}
