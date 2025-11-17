#!/usr/bin/env python3
"""
Setup and Launch Script for Hieroglyphic Scriptorium
Installs dependencies, starts the model server, and launches the app in browser
"""

import os
import sys
import subprocess
import time
import webbrowser
import signal
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

# Configuration
OPENAI_API_KEY = "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"
MODEL_SERVER_PORT = 5001
MODEL_SERVER_URL = f"http://localhost:{MODEL_SERVER_PORT}"
APP_HTML_PATH = Path(__file__).parent / "App.html"
REQUIREMENTS_FILE = Path(__file__).parent / "requirements_model.txt"
MODEL_SERVER_SCRIPT = Path(__file__).parent / "model_server.py"
MODEL_SERVER_LOG = Path(__file__).parent / "model_server.log"

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, color=Colors.CYAN):
    """Print colored status message"""
    print(f"{color}{Colors.BOLD}✓{Colors.RESET} {message}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}{Colors.BOLD}✗{Colors.RESET} {message}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠{Colors.RESET} {message}")

def print_header(message):
    """Print header message"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def check_python_version():
    """Check if Python version is 3.7+"""
    print_header("Checking Python Version")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print_error(f"Python 3.7+ required. Found Python {version.major}.{version.minor}")
        sys.exit(1)
    print_status(f"Python {version.major}.{version.minor}.{version.micro} detected")

def install_dependencies():
    """Install Python dependencies from requirements file"""
    print_header("Installing Dependencies")
    
    if not REQUIREMENTS_FILE.exists():
        print_warning(f"Requirements file not found: {REQUIREMENTS_FILE}")
        print_status("Installing basic dependencies...")
        packages = [
            "flask>=2.0.0",
            "flask-cors>=3.0.0",
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "timm>=0.6.0",
            "albumentations>=1.1.0",
            "pillow>=8.0.0",
            "numpy>=1.21.0",
            "openai>=1.0.0"
        ]
    else:
        print_status(f"Found requirements file: {REQUIREMENTS_FILE}")
        packages = None
    
    try:
        if packages:
            for package in packages:
                print_status(f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"], 
                            check=True, capture_output=True)
        else:
            print_status("Installing from requirements file...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE), "--quiet"],
                         check=True, capture_output=True)
        print_status("All dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        print_warning("You may need to install dependencies manually:")
        print(f"  pip install -r {REQUIREMENTS_FILE}")
        return False
    return True

def check_model_files():
    """Check if required model files exist"""
    print_header("Checking Model Files")
    
    model_path = Path(__file__).parent / "ResNeXT-0.8133.pth"
    if not model_path.exists():
        print_warning(f"Model file not found: {model_path}")
        print_warning("The app will work but ResNeXT predictions won't be available")
    else:
        print_status(f"Model file found: {model_path}")
    
    if not MODEL_SERVER_SCRIPT.exists():
        print_error(f"Model server script not found: {MODEL_SERVER_SCRIPT}")
        return False
    
    print_status(f"Model server script found: {MODEL_SERVER_SCRIPT}")
    return True

def stop_existing_server():
    """Stop any existing model server processes"""
    print_header("Checking for Existing Servers")
    try:
        # Kill processes on the port
        result = subprocess.run(
            ["lsof", "-ti", f":{MODEL_SERVER_PORT}"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print_status(f"Stopped existing server (PID: {pid})")
                    except ProcessLookupError:
                        pass
            time.sleep(2)
        else:
            print_status("No existing server found")
    except Exception as e:
        print_warning(f"Could not check for existing servers: {e}")

def start_model_server():
    """Start the model server"""
    print_header("Starting Model Server")
    
    # Set environment variables
    env = os.environ.copy()
    env['OPENAI_API_KEY'] = OPENAI_API_KEY
    
    try:
        # Start server in background
        log_file = open(MODEL_SERVER_LOG, 'w')
        process = subprocess.Popen(
            [sys.executable, str(MODEL_SERVER_SCRIPT)],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent)
        )
        
        print_status(f"Model server starting (PID: {process.pid})")
        print_status(f"Log file: {MODEL_SERVER_LOG}")
        
        # Wait for server to start
        print_status("Waiting for server to initialize...")
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = urlopen(f"{MODEL_SERVER_URL}/health", timeout=2)
                if response.getcode() == 200:
                    print_status("Model server is running!")
                    return process
            except (URLError, OSError):
                if attempt < max_attempts - 1:
                    time.sleep(1)
                else:
                    print_error("Server failed to start within timeout period")
                    print_warning("Check the log file for errors:")
                    print(f"  tail -f {MODEL_SERVER_LOG}")
                    return None
        
        return process
    except Exception as e:
        print_error(f"Failed to start model server: {e}")
        return None

def launch_browser():
    """Launch App.html in default browser"""
    print_header("Launching Application")
    
    if not APP_HTML_PATH.exists():
        print_error(f"App.html not found: {APP_HTML_PATH}")
        return False
    
    # Use file:// URL for local file
    file_url = f"file://{APP_HTML_PATH.absolute()}"
    
    try:
        print_status(f"Opening {APP_HTML_PATH.name} in default browser...")
        webbrowser.open(file_url)
        print_status("Browser launched successfully!")
        return True
    except Exception as e:
        print_error(f"Failed to launch browser: {e}")
        print_warning(f"Please manually open: {file_url}")
        return False

def print_summary(server_process):
    """Print summary and instructions"""
    print_header("Setup Complete!")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}Application Status:{Colors.RESET}")
    print(f"  • Model Server: {Colors.GREEN}Running{Colors.RESET} on {MODEL_SERVER_URL}")
    print(f"  • App HTML: {Colors.GREEN}Opened{Colors.RESET} in browser")
    print(f"  • OpenAI API: {Colors.GREEN}Configured{Colors.RESET}")
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}Useful Commands:{Colors.RESET}")
    print(f"  • View server logs: tail -f {MODEL_SERVER_LOG}")
    print(f"  • Stop server: pkill -f model_server.py")
    print(f"  • Restart server: python3 {MODEL_SERVER_SCRIPT}")
    
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Note:{Colors.RESET}")
    print(f"  • Keep this terminal open to keep the server running")
    print(f"  • Press Ctrl+C to stop the server and exit")
    print(f"  • The server will continue running in the background")
    
    if server_process:
        print(f"\n{Colors.BLUE}Server PID: {server_process.pid}{Colors.RESET}")

def main():
    """Main setup and launch function"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   Hieroglyphic Scriptorium - Setup & Launch Script       ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(Colors.RESET)
    
    try:
        # Step 1: Check Python version
        check_python_version()
        
        # Step 2: Install dependencies
        if not install_dependencies():
            print_warning("Continuing despite dependency installation issues...")
        
        # Step 3: Check model files
        check_model_files()
        
        # Step 4: Stop existing servers
        stop_existing_server()
        
        # Step 5: Start model server
        server_process = start_model_server()
        if not server_process:
            print_error("Failed to start model server")
            print_warning("You can try starting it manually:")
            print(f"  export OPENAI_API_KEY='{OPENAI_API_KEY}'")
            print(f"  python3 {MODEL_SERVER_SCRIPT}")
            sys.exit(1)
        
        # Step 6: Launch browser
        launch_browser()
        
        # Step 7: Print summary
        print_summary(server_process)
        
        # Keep script running
        print(f"\n{Colors.CYAN}Server is running. Press Ctrl+C to stop...{Colors.RESET}\n")
        
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Stopping server...{Colors.RESET}")
            server_process.terminate()
            server_process.wait()
            print_status("Server stopped")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup interrupted by user{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

