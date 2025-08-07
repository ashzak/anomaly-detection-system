#!/usr/bin/env python3
"""
Launch script for the Anomaly Detection Web UI
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    
    print("🔍 Starting Anomaly Detection Web UI...")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        'anomaly_detection_app.py',
        'data_utils.py',
        'anomaly_models.py',
        'evaluation.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all files are in the current directory.")
        return
    
    try:
        # Launch Streamlit app
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "anomaly_detection_app.py",
            "--server.port", "8502",
            "--server.address", "localhost",
            "--server.headless", "true"
        ]
        
        print("🚀 Launching web interface...")
        print("📱 Open this URL in your browser:")
        print("🌐 URL: http://localhost:8502")
        print("\n💡 To stop the app, press Ctrl+C in this terminal")
        print("=" * 50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down the app...")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("\n🔧 Try running manually:")
        print("python3 -m streamlit run anomaly_detection_app.py")

if __name__ == "__main__":
    main()