# OpenAI Vision Integration

This app now supports using OpenAI's Vision API (GPT-4 Vision) to identify hieroglyphs from user drawings.

## Setup

1. **Install OpenAI library:**
   ```bash
   pip install openai
   ```
   Or install all requirements:
   ```bash
   pip install -r requirements_model.txt
   ```

2. **Set OpenAI API Key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or add to your shell profile (`~/.zshrc` or `~/.bashrc`):
   ```bash
   echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
   source ~/.zshrc
   ```

3. **Restart the model server:**
   ```bash
   pkill -f model_server.py
   python3 model_server.py
   ```

## Usage

1. Draw a hieroglyph on the canvas
2. Click **"𓊹 Analyze with OpenAI Vision"** button
3. OpenAI Vision will analyze your drawing and provide:
   - Gardiner number (if identifiable)
   - Name/description
   - Phonetic sound
   - Meaning
   - Confidence level
   - Reasoning for the identification

## How It Works

- The canvas drawing is sent as a base64-encoded image to `/predict_openai_vision` endpoint
- OpenAI Vision API analyzes the image with a specialized prompt for hieroglyph identification
- The response is parsed and displayed in a user-friendly format
- If a Gardiner number is identified, the corresponding glyph image is shown

## Comparison: ResNeXT vs OpenAI Vision

- **ResNeXT Model**: Fast, local, trained specifically on hieroglyphs, provides top 3 predictions with confidence scores
- **OpenAI Vision**: Uses general vision understanding, provides detailed reasoning and context, may be better at identifying unusual or poorly drawn glyphs

## API Endpoint

**POST** `/predict_openai_vision`

Request body:
```json
{
  "image": "data:image/png;base64,..."
}
```

Response:
```json
{
  "success": true,
  "prediction": {
    "gardiner_num": "A1",
    "description": "Seated man",
    "phonetic": "...",
    "meaning": "...",
    "confidence": "high",
    "reasoning": "...",
    "image_path": "archaeohack-starterpack/data/utf-pngs/A1.png",
    "hieroglyph": "𓀀",
    "source": "openai_vision"
  },
  "raw_response": "..."
}
```

## Troubleshooting

- **"OPENAI_API_KEY environment variable not set"**: Set the API key as shown above
- **"OpenAI library not installed"**: Run `pip install openai`
- **API errors**: Check your OpenAI API key is valid and has credits
- **Slow responses**: OpenAI Vision API may take a few seconds to respond

