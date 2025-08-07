#!/bin/bash

echo "🔍 Starting Anomaly Detection Web UI..."
echo "=" * 50

# Kill any existing streamlit processes
pkill -f streamlit 2>/dev/null || true

# Wait a moment
sleep 2

# Start streamlit on port 8504 (to avoid conflicts)
echo "🚀 Launching on http://localhost:8504"
echo "📱 Copy and paste this URL into your browser:"
echo ""
echo "    http://localhost:8504"
echo ""
echo "💡 Press Ctrl+C to stop the application"
echo "=" * 50

# Launch with explicit settings to avoid issues
python3 -m streamlit run anomaly_detection_app.py \
    --server.port 8504 \
    --server.address 127.0.0.1 \
    --browser.serverAddress 127.0.0.1 \
    --server.headless false \
    --server.runOnSave false