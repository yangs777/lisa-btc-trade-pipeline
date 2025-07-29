#!/bin/bash
# Start Data Collection Pipeline

echo "🚀 Starting Bitcoin Orderbook Data Collection Pipeline"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate || source venv/Scripts/activate

# Install dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch .deps_installed
fi

# Check environment variables
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠️  Warning: GOOGLE_APPLICATION_CREDENTIALS not set"
    echo "GCS upload will not work without credentials"
    echo "Set with: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
fi

# Create necessary directories
mkdir -p data/raw data/temp logs

# Start processes
echo ""
echo "Starting components:"
echo "1. Orderbook Collector (Port: WebSocket)"
echo "2. GCS Uploader (Interval: 5 min)"
echo ""

# Start collector in background
echo "Starting orderbook collector..."
python -m data_collection.data_collector > logs/collector.log 2>&1 &
COLLECTOR_PID=$!
echo "Collector PID: $COLLECTOR_PID"

# Wait a bit for collector to start
sleep 5

# Start GCS uploader if credentials are available
if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Starting GCS uploader..."
    python -m data_collection.gcs_uploader > logs/uploader.log 2>&1 &
    UPLOADER_PID=$!
    echo "Uploader PID: $UPLOADER_PID"
else
    echo "Skipping GCS uploader (no credentials)"
    UPLOADER_PID=""
fi

# Save PIDs
echo $COLLECTOR_PID > .collector.pid
if [ ! -z "$UPLOADER_PID" ]; then
    echo $UPLOADER_PID > .uploader.pid
fi

echo ""
echo "✅ Data collection pipeline started!"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/collector.log"
echo "  tail -f logs/uploader.log"
echo ""
echo "Stop with: ./scripts/stop_collector.sh"
echo ""

# Keep script running and handle signals
trap 'echo "Stopping services..."; kill $COLLECTOR_PID $UPLOADER_PID 2>/dev/null; exit' INT TERM

# Wait for processes
wait