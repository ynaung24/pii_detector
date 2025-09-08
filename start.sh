#!/bin/bash

# PII Detection Agent Start Script
echo "🚀 Starting PII Detection Agent..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run './setup.sh' first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating Python virtual environment..."
source venv/bin/activate

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "⚠️  .env.local not found. Creating from template..."
    cp env_template.txt .env.local
    echo "⚠️  Please edit .env.local and add your OpenAI API key before continuing."
    echo "Press Enter to continue or Ctrl+C to exit..."
    read
fi

# Start backend server in background
echo "🔧 Starting Python backend server on port 3001..."
python backend_server.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend development server
echo "🔧 Starting TypeScript frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo ""
echo "✅ PII Detection Agent is running!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:3001"
echo "📚 API Docs: http://localhost:3001/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for processes
wait
