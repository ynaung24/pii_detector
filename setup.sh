#!/bin/bash

# PII Detection Agent Setup Script
echo "🚀 Setting up PII Detection Agent with TypeScript Frontend..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed. Please install Node.js 18+ and try again."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed. Please install npm and try again."
    exit 1
fi

echo "✅ Python and Node.js are installed"

# Create virtual environment for Python
echo "📦 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
echo "📦 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Setup frontend
echo "📦 Setting up TypeScript frontend..."
cd frontend
npm install
cd ..

# Create environment file from template
if [ ! -f .env.local ]; then
    echo "📝 Creating environment file..."
    cp env_template.txt .env.local
    echo "⚠️  Please edit .env.local and add your OpenAI API key"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads
mkdir -p frontend/uploads

echo "✅ Setup completed successfully!"
echo ""
echo "🔧 Next steps:"
echo "1. Edit .env.local and add your OpenAI API key"
echo "2. Run './start.sh' to start both backend and frontend"
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "📚 For more information, see README.md"
