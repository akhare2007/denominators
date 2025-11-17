# ResNeXT Model Integration

The ResNeXT model (81.33% validation accuracy) has been integrated into the App.html web application.

## Setup

1. **Install required Python packages:**
```bash
pip install flask flask-cors torch torchvision timm albumentations pillow
```

2. **Start the model server:**
```bash
python3 model_server.py
```

The server will start on `http://localhost:5000`

## Features

### 1. Upload Image Analysis
- Go to the "Upload Image" tab
- Upload or drag-and-drop a hieroglyph image
- Click "𓎔 Analyze Image" button
- Get top 5 predictions with confidence scores

### 2. Drawing Analysis
- Draw a hieroglyph on the canvas in the "Draw" tab
- Click "𓎔 Analyze Drawing" button
- Get top 5 predictions with confidence scores

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Upload image file for prediction
- `POST /predict_base64` - Send base64 encoded image for prediction

## Model Details

- **Architecture**: ResNeXT-50 (resnext50_32x4d)
- **Accuracy**: 81.33% validation accuracy
- **Classes**: 782 hieroglyph classes
- **Model File**: `ResNeXT-0.8133.pth` (94 MB)

## Troubleshooting

If you see "Error: Failed to fetch" or connection errors:
1. Make sure the model server is running (`python3 model_server.py`)
2. Check that the server is accessible at `http://localhost:5000`
3. Verify the model file exists at the path specified in `model_server.py`

