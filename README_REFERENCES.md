# References in README Files

This document lists all references, URLs, file paths, and citations found in the README documentation files.

## External URLs and Web Resources

### Google Fonts
- **URL**: `https://fonts.googleapis.com`
- **URL**: `https://fonts.gstatic.com`
- **Font CSS**: `https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=MedievalSharp&family=Uncial+Antiqua&display=swap`
- **Fonts Used**:
  - Cinzel (400, 600, 700 weights)
  - MedievalSharp
  - Uncial Antiqua
- **Location**: `App.html` (referenced in README_SETUP.md)

## Local Server URLs

### Model Server
- **Development Server**: `http://localhost:5001`
  - Health check endpoint: `http://localhost:5001/health`
  - Prediction endpoint: `http://localhost:5001/predict`
  - Base64 prediction endpoint: `http://localhost:5001/predict_base64`
  - OpenAI Vision endpoint: `http://localhost:5001/predict_openai_vision`
- **Legacy Reference** (outdated): `http://localhost:5000` (mentioned in README_MODEL.md but should be 5001)
- **Location**: `README_SETUP.md`, `README_MODEL.md`, `setup_and_launch.py`, `setup_openai_key.sh`

### Local HTTP Server (for CORS)
- **URL**: `http://localhost:8000/App.html`
- **Command**: `python3 -m http.server 8000`
- **Location**: `App.html` (error message), `README_SETUP.md`

## File Paths and References

### Core Application Files
- `App.html` - Main application file
- `model_server.py` - Flask backend server
- `setup_and_launch.py` - Setup and launch script
- `requirements_model.txt` - Python dependencies

### Model Files
- `ResNeXT-0.8133.pth` - Trained ResNeXT model (94 MB)
- `model_vals.json` - Class mapping file (769 classes)
- `training_class_list.json` - Fallback class mapping file

### Data Files
- `archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json` - Hieroglyph database
- `archaeohack-starterpack/data/utf-pngs/` - PNG images directory
- `archaeohack-starterpack/data/utf-pngs/{gardiner_num}.png` - Individual glyph images

### Configuration Files
- `setup_openai_key.sh` - OpenAI API key setup script

### Log Files
- `model_server.log` - Server log file

### Notebook Files (Referenced but Deleted)
- `non-vit (1).ipynb` - Training notebook (referenced in README_MODEL_VALS.md)
- `extract_782_classes_from_notebook.py` - Helper script (referenced but deleted)

## API Endpoints

### Model Server Endpoints
1. **GET** `/health`
   - Health check endpoint
   - Returns: `{"status": "ok", "message": "Model server is running"}`

2. **POST** `/predict`
   - Upload image file for prediction
   - Request: FormData with `image` file
   - Response: JSON with top 3 predictions

3. **POST** `/predict_base64`
   - Send base64 encoded image for prediction
   - Request: `{"image": "data:image/png;base64,..."}`
   - Response: JSON with top 3 predictions

4. **POST** `/predict_openai_vision`
   - Use OpenAI Vision API for prediction
   - Request: `{"image": "data:image/png;base64,..."}`
   - Response: JSON with OpenAI Vision analysis

## Python Packages and Dependencies

### Core Dependencies
- `flask` (>=2.0.0)
- `flask-cors` (>=3.0.0)
- `torch` (>=1.9.0)
- `torchvision` (>=0.10.0)
- `timm` (>=0.6.0)
- `albumentations` (>=1.1.0)
- `pillow` (>=8.0.0)
- `numpy` (>=1.21.0)
- `openai` (>=1.0.0)

## System Requirements

### Python
- **Version**: Python 3.7+
- **Package Manager**: pip

### Operating System Commands
- **macOS**: `open App.html`
- **Linux**: `xdg-open App.html`
- **Windows**: `start App.html`

### Shell Profiles
- `~/.zshrc` - Zsh configuration file
- `~/.bashrc` - Bash configuration file

## Environment Variables

### OpenAI API Key
- **Variable**: `OPENAI_API_KEY`
- **Example**: `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`
- **Location**: `README_SETUP.md`, `setup_and_launch.py`

## Model Specifications

### ResNeXT Model
- **Architecture**: ResNeXT-50 (resnext50_32x4d)
- **Accuracy**: 81.33% validation accuracy
- **Classes**: 782 hieroglyph classes
- **Model File Size**: 94 MB
- **Location**: `README_MODEL.md`

### Class Mapping
- **Current Classes**: 769 (in `model_vals.json`)
- **Expected Classes**: 782
- **Missing Classes**: 13 (from HuggingFace dataset)
- **Ordering**: Lexicographic string sort
- **Location**: `README_MODEL_VALS.md`

## API Models and Services

### OpenAI Vision API
- **Model**: GPT-4o (or gpt-4-vision-preview)
- **Service**: OpenAI Vision API
- **Location**: `README_OPENAI_VISION.md`

## File Protocols

### Local File Access
- **Protocol**: `file://` protocol
- **Usage**: Opening App.html directly in browser
- **Location**: `README_SETUP.md`

## Terminal Commands

### Server Management
- `python3 model_server.py` - Start model server
- `pkill -f model_server.py` - Stop model server
- `tail -f model_server.log` - View server logs
- `lsof -ti:5001 | xargs kill -9` - Kill process on port 5001

### Setup Commands
- `python3 setup_and_launch.py` - Run setup script
- `pip install -r requirements_model.txt` - Install dependencies
- `pip install --upgrade pip` - Upgrade pip

## Port Numbers

- **5001** - Model server port (current)
- **5000** - Legacy model server port (outdated reference)
- **8000** - Local HTTP server port (for CORS)

## Directory Structure References

### archaeohack-starterpack/
- `data/gardiner_hieroglyphs_with_unicode_hex.json` - JSON array of glyph objects with UTF hex codes, characters, Gardiner numbers, descriptions, and priority flags
- `data/utf-pngs/` - PNG images generated from UTF hex codes (769 files)
- `data/me-sign-examples-pjb/` - Drawn sample images for testing glyph recognition, organized by Gardiner number
- `docs/Alan Gardiner's List of Hieroglyphic Signs.pdf` - Additional documentation
- `lib/` - Scripts and resources for cleaning & producing the data
- `lib/font/NotoSansEgyptianHieroglyphs-Regular.ttf` - Egyptian hieroglyph font file
- **Location**: `archaeohack-starterpack/README.md`

## Code References

### Python Code Snippets
- Class extraction code (README_MODEL_VALS.md)
- JSON dump example (README_MODEL_VALS.md)
- API request examples (README_OPENAI_VISION.md)

### JavaScript References
- Fetch API calls to `http://localhost:5001`
- Canvas to base64 conversion
- File upload handling

## Image References

### Example Glyph Image
- **File**: `data/utf-pngs/A46.png`
- **Description**: King holding flail and wearing red crown of Lower Egypt
- **Gardiner Number**: A46
- **Unicode**: U+13037
- **Location**: `archaeohack-starterpack/README.md`

## Data Structure References

### JSON Glyph Object Structure
Each glyph object in `gardiner_hieroglyphs_with_unicode_hex.json` contains:
- `gardiner_num` - Gardiner classification number
- `hieroglyph` - UTF character representation
- `unicode_hex` - UTF hex code
- `description` - Descriptive name
- `details` - Detailed information
- `is_priority` - Boolean flag for priority status

## Notes

- Some references point to deleted files (notebooks, helper scripts)
- Port 5000 is mentioned in README_MODEL.md but should be 5001
- OpenAI API key is embedded in setup scripts for convenience
- All file paths use absolute paths in `model_server.py` but relative paths in documentation
- The archaeohack-starterpack contains sample images for testing glyph recognition

