import { NextRequest, NextResponse } from 'next/server';

// Handle all HTTP methods
export async function GET(request: NextRequest) {
  return proxyRequest(request, 'GET');
}

export async function POST(request: NextRequest) {
  return proxyRequest(request, 'POST');
}

export async function PUT(request: NextRequest) {
  return proxyRequest(request, 'PUT');
}

export async function PATCH(request: NextRequest) {
  return proxyRequest(request, 'PATCH');
}

export async function DELETE(request: NextRequest) {
  return proxyRequest(request, 'DELETE');
}

async function proxyRequest(request: NextRequest, method: string) {
  try {
    // Extract the path from the request URL
    const requestUrl = new URL(request.url);
    const path = requestUrl.pathname.replace('/api/proxy', '');
    
    // Construct the backend URL
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    const url = new URL(`${backendUrl}${path}`);
    
    // Copy query parameters
    requestUrl.searchParams.forEach((value, key) => {
      url.searchParams.append(key, value);
    });

    console.log(`Proxying ${method} request to: ${url.toString()}`);

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
    
    // Copy authorization header
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
