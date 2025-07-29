#!/bin/bash
# Stop Data Collection Pipeline

echo "🛑 Stopping Bitcoin Orderbook Data Collection Pipeline"
echo "===================================================="

# Read PIDs
if [ -f ".collector.pid" ]; then
    COLLECTOR_PID=$(cat .collector.pid)
    echo "Stopping collector (PID: $COLLECTOR_PID)..."
    kill -TERM $COLLECTOR_PID 2>/dev/null
    rm .collector.pid
fi

if [ -f ".uploader.pid" ]; then
    UPLOADER_PID=$(cat .uploader.pid)
    echo "Stopping uploader (PID: $UPLOADER_PID)..."
    kill -TERM $UPLOADER_PID 2>/dev/null
    rm .uploader.pid
fi

# Wait a bit
sleep 2

# Check if processes are still running
if [ ! -z "$COLLECTOR_PID" ] && ps -p $COLLECTOR_PID > /dev/null 2>&1; then
    echo "Force stopping collector..."
    kill -9 $COLLECTOR_PID 2>/dev/null
fi

if [ ! -z "$UPLOADER_PID" ] && ps -p $UPLOADER_PID > /dev/null 2>&1; then
    echo "Force stopping uploader..."
    kill -9 $UPLOADER_PID 2>/dev/null
fi

echo "✅ Data collection pipeline stopped"