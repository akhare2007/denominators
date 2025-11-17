# Setup and Launch Guide

## Quick Start

Run the setup script to install dependencies, start the server, and launch the app:

```bash
python3 setup_and_launch.py
```

Or make it executable and run directly:

```bash
chmod +x setup_and_launch.py
./setup_and_launch.py
```

## What the Script Does

1. **Checks Python Version** - Ensures Python 3.7+ is installed
2. **Installs Dependencies** - Installs all required packages from `requirements_model.txt`
3. **Checks Model Files** - Verifies model files exist
4. **Stops Existing Servers** - Cleans up any running servers on port 5001
5. **Starts Model Server** - Launches the Flask server with OpenAI API key configured
6. **Launches Browser** - Opens `App.html` in your default web browser

## Manual Setup

If you prefer to set up manually:

### 1. Install Dependencies

```bash
pip install -r requirements_model.txt
```

Or install individually:
```bash
pip install flask flask-cors torch torchvision timm albumentations pillow numpy openai
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="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"
```

### 3. Start Model Server

```bash
python3 model_server.py
```

### 4. Open App.html

Open `App.html` in your web browser:
- **macOS**: `open App.html`
- **Linux**: `xdg-open App.html`
- **Windows**: `start App.html`

Or manually navigate to the file in your browser.

## Troubleshooting

### Port Already in Use

If port 5001 is already in use:

```bash
# Find and kill the process
lsof -ti:5001 | xargs kill -9

# Or use the script's built-in cleanup
python3 setup_and_launch.py
```

### Dependencies Not Installing

Try upgrading pip first:
```bash
pip install --upgrade pip
pip install -r requirements_model.txt
```

### Model Server Not Starting

Check the log file:
```bash
tail -f model_server.log
```

### Browser Not Opening

Manually open `App.html`:
- Double-click the file in Finder/File Explorer
- Or drag it into your browser

## Features

- ✅ Automatic dependency installation
- ✅ Server health checking
- ✅ Background server process management
- ✅ Cross-platform browser launching
- ✅ Colored terminal output
- ✅ Error handling and recovery
- ✅ Server log management

## Server Management

### View Logs
```bash
tail -f model_server.log
```

### Stop Server
```bash
pkill -f model_server.py
```

### Restart Server
```bash
python3 setup_and_launch.py
```

## Requirements

- Python 3.7+
- pip (Python package manager)
- Default web browser
- Internet connection (for initial dependency installation and OpenAI API)

## Notes

- The script keeps running to maintain the server process
- Press Ctrl+C to stop the server and exit
- The OpenAI API key is embedded in the script for convenience
- Server runs on `http://localhost:5001`
- App.html uses `file://` protocol (no local server needed for HTML)

