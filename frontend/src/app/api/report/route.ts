import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'

export async function POST(request: NextRequest) {
  try {
    const { report_path } = await request.json()
    
    if (!report_path) {
      return NextResponse.json(
        { error: 'Report path is required' },
        { status: 400 }
      )
    }

    // Forward the request to the Python backend
    const response = await fetch(`${BACKEND_URL}/api/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ report_path }),
    })

    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json(
        { error: errorData.detail || 'Failed to read report' },
        { status: response.status }
      )
    }

    const report = await response.json()
    return NextResponse.json(report)

  } catch (error) {
    console.error('Error reading report:', error)
    return NextResponse.json(
      { error: 'Failed to read report' },
      { status: 500 }
    )
  }
}
