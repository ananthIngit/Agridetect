"""
Startup script to run both Flask backend (API + ML) and React frontend (Vite)
"""
import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def check_dependencies(silent=False):
    """Check if required Python dependencies are installed"""
    # Check for core packages used by the ML and API parts
    required = ["flask", "torch", "timm", "opencv-python", "flask_cors"]
    missing = []
    
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        if not silent:
            print(f"❌ Missing Python dependencies: {', '.join(missing)}")
            print("Please run: pip install -r requirements.txt to install them.")
        return False
    
    if not silent:
        print("✅ Python dependencies found")
    return True

def start_flask():
    """Start Flask backend server"""
    print("🚀 Starting Flask API backend server (port 5000)...")
    # Run the Flask application, relying on the 'if __name__ == "__main__":' block in app.py
    subprocess.run([sys.executable, "app.py"], env=os.environ.copy())

def start_react():
    """Start React frontend using Vite"""
    print("🚀 Starting React frontend server (port 5173)...")
    # Check if node_modules exists
    if not os.path.exists("node_modules"):
        print("📦 Installing npm dependencies...")
        npm_command = ["npm", "ci"] if os.path.exists("package-lock.json") else ["npm", "install"]
        try:
            # Use shell=True on Windows to handle npm commands better
            subprocess.run(npm_command, check=True, shell=sys.platform.startswith('win'))
            print("✅ Node.js dependencies installed successfully.")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Node.js dependencies.")
            return

    # FIX: Add shell=True for Windows compatibility
    subprocess.run(["npm", "run", "dev", "--", "--port", "5173"], shell=sys.platform.startswith('win'))

def open_browser():
    """Open browser after a delay"""
    time.sleep(7)  # Wait longer for both servers to stabilize
    print("🌐 Opening browser...")
    webbrowser.open("http://localhost:5173")

if __name__ == "__main__":
    print("=" * 50)
    print("🌱 AgriDetect - Starting Unified Servers")
    print("=" * 50)
    
    # Optional: Run initial dependency check (can be removed if user relies solely on manual install)
    if not check_dependencies():
        print("Continuing startup, but please ensure you've installed all Python dependencies from requirements.txt.")
    
    # 1. Start Flask (ML/API) in a separate thread
    flask_thread = Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    # 2. Start browser opener in a separate thread
    browser_thread = Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # 3. Start React (Frontend) - this will block the main thread until interrupted
    try:
        start_react()
    except KeyboardInterrupt:
        print("\n👋 Shutting down servers...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)