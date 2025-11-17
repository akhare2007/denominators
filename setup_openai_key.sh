#!/bin/bash
# Script to set OpenAI API key and start model server

export OPENAI_API_KEY="sk-proj-***************"
echo "✓ OpenAI API key set"
echo "Starting model server..."

# Kill any existing server
pkill -f "python.*model_server" 2>/dev/null
sleep 2

# Start server
cd /Users/aayankhare/Desktop/D-en-ominators
python3 model_server.py > model_server.log 2>&1 &

echo "✓ Model server started"
echo "Check status: tail -f model_server.log"
echo "Health check: curl http://localhost:5001/health"

