#!/usr/bin/env python3
"""
Flask server for ResNeXT hieroglyph classification model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import io
import base64
import json
import os
from pathlib import Path

# OpenAI Vision API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠ OpenAI library not installed. Install with: pip install openai")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model variable
model = None
device = None
class_to_gardiner = None  # Mapping from class index to Gardiner number
gardiner_data = None  # Full hieroglyph data

def load_class_mapping():
    """Load class index to Gardiner number mapping - must match training exactly"""
    global class_to_gardiner, gardiner_data
    
    if class_to_gardiner is not None:
        return class_to_gardiner, gardiner_data
    
    # Try to load model_vals.json first (preferred), then fallback to training_class_list.json
    model_vals_path = Path("/Users/aayankhare/Desktop/D-en-ominators/model_vals.json")
    class_list_path = Path("/Users/aayankhare/Desktop/D-en-ominators/training_class_list.json")
    
    training_classes = None
    
    # Try model_vals.json first (this is the exact order the model needs)
    if model_vals_path.exists():
        print("✓ Found model_vals.json, using it for class mapping")
        with open(model_vals_path, 'r') as f:
            model_vals = json.load(f)
        
        if isinstance(model_vals, list):
            # Direct array format - this is what the model expects
            training_classes = model_vals
            print(f"✓ Loaded {len(training_classes)} classes from model_vals.json")
        elif 'classes' in model_vals:
            training_classes = model_vals['classes']
            print(f"✓ Loaded {len(training_classes)} classes from model_vals.json (dict format)")
    
    # Fallback to training_class_list.json
    if training_classes is None and class_list_path.exists():
        print("✓ Found training_class_list.json, using it for class mapping")
        with open(class_list_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                training_classes = data
            elif isinstance(data, dict) and 'classes' in data:
                training_classes = data['classes']
            else:
                training_classes = data
    
    if training_classes:
        # Ensure classes are in lexicographic order (as model expects)
        training_classes = sorted(training_classes)
        
        # Create mapping from the training class list (even if not exactly 782)
        # This ensures we use the exact order from training
        class_to_gardiner = {idx: gardiner for idx, gardiner in enumerate(training_classes)}
        
        if len(training_classes) == 782:
            print(f"✓ Loaded exact 782-class mapping (model expects 782)")
        else:
            print(f"✓ Loaded {len(training_classes)}-class mapping (model expects 782)")
            print(f"  Missing {782 - len(training_classes)} classes - predictions beyond index {len(training_classes)-1} will show 'unknown'")
            print(f"  But predictions within range [0-{len(training_classes)-1}] should be accurate!")
        
        # Load gardiner data for metadata
        json_path = Path("/Users/aayankhare/Desktop/D-en-ominators/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json")
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                gardiner_data = json.load(f)
        else:
            gardiner_data = []
        return class_to_gardiner, gardiner_data
    
    # Load hieroglyph data for metadata
    json_path = Path("/Users/aayankhare/Desktop/D-en-ominators/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json")
    
    gardiner_data = []
    gardiner_to_info = {}
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            gardiner_data = json.load(f)
        
        # Create mapping: Gardiner number -> full glyph info
        for glyph in gardiner_data:
            gardiner_num = glyph.get('gardiner_num')
            if gardiner_num:
                gardiner_to_info[gardiner_num] = glyph
    
    # IMPORTANT: Get classes from PNG files (same as training)
    # Training used: unique_classes = sorted(all_df['label'].unique())
    # This means lexicographic (string) sorting: "A1", "A10", "A11", "A2"
    png_dir = Path("/Users/aayankhare/Desktop/D-en-ominators/archaeohack-starterpack/data/utf-pngs")
    if png_dir.exists():
        png_files = list(png_dir.glob("*.png"))
        # Extract Gardiner numbers from PNG filenames
        # Use sorted() for lexicographic sort (same as training)
        unique_gardiners_from_pngs = sorted(set([f.stem for f in png_files]))
        
        print(f"✓ Found {len(unique_gardiners_from_pngs)} classes from PNG files")
        
        # CRITICAL ISSUE: Training had 782 classes from multiple sources (local, HF, GitHub)
        # We only have PNG files for 769 classes locally
        # The missing 13 classes cause systematic mapping errors!
        # 
        # Example: If training had classes [A1, A10, MISSING_CLASS, A11, A2...]
        #          But we map:           [A1, A10, A11, A2...]
        #          Then class index 2 maps to wrong Gardiner number!
        #
        # SOLUTION: We need the exact training class list. For now, we'll use
        # what we have, but predictions will be systematically offset.
        
        # Create reverse mapping: class index -> Gardiner number
        # WARNING: This mapping is incomplete - missing 13 classes causes offset errors
        class_to_gardiner = {}
        for idx, gardiner in enumerate(unique_gardiners_from_pngs):
            class_to_gardiner[idx] = gardiner
        
        # For indices beyond our PNG list (782 - 769 = 13 classes), use placeholder
        # But this causes systematic errors - all predictions after the missing classes are offset
        
        if len(class_to_gardiner) != 782:
            print(f"⚠ CRITICAL: Model outputs 782 classes, but only {len(class_to_gardiner)} have PNG files")
            print(f"  Missing {782 - len(class_to_gardiner)} classes - predictions will be systematically offset!")
            print(f"  This causes 'precise but inaccurate' results (consistent wrong mapping)")
            print(f"  SOLUTION: Save training class list to training_class_list.json")
            print(f"  To fix: Run training notebook and save: json.dump(unique_classes, open('training_class_list.json', 'w'))")
        
        print(f"✓ Loaded {len(class_to_gardiner)} class mappings from PNG files (INCOMPLETE - may cause offset errors)")
        return class_to_gardiner, gardiner_data
    else:
        print(f"⚠ Warning: Could not find PNG directory at {png_dir}")
        # Fallback to JSON if PNG directory doesn't exist
        if gardiner_to_info:
            unique_gardiners = sorted(gardiner_to_info.keys())
            class_to_gardiner = {idx: gardiner for idx, gardiner in enumerate(unique_gardiners)}
            print(f"✓ Loaded {len(class_to_gardiner)} class mappings from JSON (fallback)")
            return class_to_gardiner, gardiner_data
        else:
            return None, None

def load_model():
    """Load the ResNeXT model"""
    global model, device
    
    if model is not None:
        return model, device
    
    # Load class mapping first
    load_class_mapping()
    
    # IMPORTANT: Model was trained with 782 classes - must match exactly
    num_classes = 782
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model wrapper class
    class ResNeXTClassifier(nn.Module):
        def __init__(self, model_name="resnext50_32x4d", num_classes=782, pretrained=False):
            super().__init__()
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        def forward(self, x):
            return self.model(x)
    
    model = ResNeXTClassifier("resnext50_32x4d", num_classes=num_classes, pretrained=False)
    
    # Load weights
    model_path = "/Users/aayankhare/Desktop/D-en-ominators/ResNeXT-0.8133.pth"
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle state dict keys that might have "model." prefix
    if any(k.startswith("model.") for k in state_dict.keys()):
        # Strip "model." prefix if present
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        # Load into the inner model
        model.model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device} with {num_classes} classes")
    if class_to_gardiner and len(class_to_gardiner) < num_classes:
        print(f"⚠ Note: Only {len(class_to_gardiner)} classes have PNG files, but model outputs {num_classes} classes")
        print(f"  Predictions for missing classes will show 'unknown' instead of Gardiner numbers")
    return model, device

def preprocess_image(image_bytes, img_size=224):
    """Preprocess image for inference"""
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = np.array(image)
    
    # Apply transforms
    transform = albu.Compose([
        albu.Resize(img_size, img_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)

def format_predictions(top_indices, top_probs):
    """Format predictions with Gardiner numbers and image paths"""
    predictions = []
    # top_indices shape is [batch_size, k], so we iterate over k (number of top predictions)
    num_predictions = top_indices.shape[1]  # Should be 3
    for i in range(num_predictions):
        class_idx = int(top_indices[0][i].item())
        prob = float(top_probs[0][i].item())
        
        # Get Gardiner number from class index
        # Use "unknown" for classes not in our mapping
        if class_to_gardiner and class_idx in class_to_gardiner:
            gardiner_num = class_to_gardiner[class_idx]
        else:
            gardiner_num = "unknown"
            # Debug: Log when we hit unknown classes
            if class_idx >= 768:
                print(f"⚠ Prediction {i+1}: Class index {class_idx} is beyond our 768-class mapping (max: 767)")
            else:
                print(f"⚠ Prediction {i+1}: Class index {class_idx} not found in mapping (unexpected)")
        
        # Get glyph info
        glyph_info = None
        if gardiner_data and gardiner_num != "unknown":
            glyph_info = next((g for g in gardiner_data if g.get('gardiner_num') == gardiner_num), None)
        
        # Image path (only if not unknown)
        if gardiner_num != "unknown":
            image_path = f"archaeohack-starterpack/data/utf-pngs/{gardiner_num}.png"
        else:
            image_path = None
        
        prediction = {
            'class_idx': class_idx,
            'gardiner_num': gardiner_num,
            'probability': prob,
            'confidence': f"{prob*100:.2f}%",
            'image_path': image_path,
            'description': glyph_info.get('description', '') if glyph_info else '',
            'hieroglyph': glyph_info.get('hieroglyph', '') if glyph_info else ''
        }
        predictions.append(prediction)
    
    return predictions

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Model server is running"})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict hieroglyph class from uploaded image"""
    try:
        # Load model if not loaded
        if model is None:
            load_model()
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Get top-3 predictions
            top_probs, top_indices = torch.topk(probs, k=3, dim=1)
            
            # Format predictions with Gardiner numbers and images
            predictions = format_predictions(top_indices, top_probs)
        
        return jsonify({
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Predict from base64 encoded image (for canvas/drawing)"""
    try:
        # Load model if not loaded
        if model is None:
            load_model()
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Get top-3 predictions
            top_probs, top_indices = torch.topk(probs, k=3, dim=1)
            
            # Format predictions with Gardiner numbers and images
            predictions = format_predictions(top_indices, top_probs)
        
        return jsonify({
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_openai_vision', methods=['POST'])
def predict_openai_vision():
    """Use OpenAI Vision API to identify hieroglyph from drawing"""
    try:
        if not OPENAI_AVAILABLE:
            return jsonify({"error": "OpenAI library not installed. Install with: pip install openai"}), 500
        
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({"error": "OPENAI_API_KEY environment variable not set"}), 500
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Get base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Keep data URL format for OpenAI
            image_url = image_data
        else:
            # Add data URL prefix if missing
            image_url = f"data:image/png;base64,{image_data}"
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Load hieroglyph data for reference
        load_class_mapping()
        
        # Create a comprehensive prompt
        prompt = """You are an expert Egyptologist analyzing an Egyptian hieroglyph drawing. 

Identify this hieroglyph and provide:
1. The Gardiner number (e.g., A1, D21, N36) - this is the standard classification system
2. The name/description of the hieroglyph
3. The phonetic sound if applicable
4. What the hieroglyph represents

If you cannot identify it clearly, respond with "Unknown" for the Gardiner number.

Respond in JSON format:
{
  "gardiner_num": "A1" or "Unknown",
  "name": "description of the hieroglyph",
  "phonetic": "phonetic sound if applicable",
  "meaning": "what it represents",
  "confidence": "high/medium/low",
  "reasoning": "brief explanation of your identification"
}"""
        
        # Call OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-vision-preview" for older models
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Extract response
        response_text = response.choices[0].message.content
        
        # Try to parse JSON from response
        try:
            # Extract JSON from response (might be wrapped in markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_str = response_text.strip()
            
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response from text
            result = {
                "gardiner_num": "Unknown",
                "name": response_text,
                "phonetic": "",
                "meaning": "",
                "confidence": "medium",
                "reasoning": "Could not parse structured response"
            }
        
        # Try to find matching glyph info
        gardiner_num = result.get('gardiner_num', 'Unknown')
        glyph_info = None
        image_path = None
        
        if gardiner_num != 'Unknown' and gardiner_data:
            glyph_info = next((g for g in gardiner_data if g.get('gardiner_num') == gardiner_num), None)
            if glyph_info:
                image_path = f"archaeohack-starterpack/data/utf-pngs/{gardiner_num}.png"
        
        # Format response similar to ResNeXT predictions
        prediction = {
            'gardiner_num': gardiner_num,
            'description': result.get('name', ''),
            'phonetic': result.get('phonetic', ''),
            'meaning': result.get('meaning', ''),
            'confidence': result.get('confidence', 'medium'),
            'reasoning': result.get('reasoning', ''),
            'image_path': image_path,
            'hieroglyph': glyph_info.get('hieroglyph', '') if glyph_info else '',
            'source': 'openai_vision'
        }
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "raw_response": response_text
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"OpenAI Vision error: {error_details}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("Starting ResNeXT Model Server")
    print("="*60)
    
    # Load model on startup
    load_model()
    
    print("\nServer starting on http://localhost:5001")
    print("Endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict - Upload image file for prediction")
    print("  POST /predict_base64 - Send base64 image for prediction")
    print("  POST /predict_openai_vision - Use OpenAI Vision API for prediction")
    if OPENAI_AVAILABLE:
        if os.getenv('OPENAI_API_KEY'):
            print("  ✓ OpenAI Vision API: Available (API key set)")
        else:
            print("  ⚠ OpenAI Vision API: Library installed but OPENAI_API_KEY not set")
    else:
        print("  ⚠ OpenAI Vision API: Not available (install with: pip install openai)")
    print("\nPress Ctrl+C to stop the server")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5001, debug=False)

