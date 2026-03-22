#!/bin/bash

# Credit AI Backend Server Startup Script

echo "🚀 Starting Credit AI Backend Server..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Please create a .env file with OPENAI_API_KEY=your_key_here"
    echo ""
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "✅ Virtual environment activated"
echo "✅ Starting FastAPI server on http://localhost:8000"
echo ""
echo "📊 API Endpoints:"
echo "   - Health Check: http://localhost:8000/health"
echo "   - Analyze: POST http://localhost:8000/v1/analyze"
echo "   - Export Excel: POST http://localhost:8000/v1/export/excel"
echo ""
echo "🌐 Frontend: Open frontend.html in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the server
uvicorn main:app --reload --port 8000
