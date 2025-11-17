# Model Values Configuration

## Current Status

- **model_vals.json**: Contains 769 classes in lexicographic order
- **Expected**: 782 classes
- **Missing**: 13 classes (likely from HuggingFace dataset)

## To Get Exact 782 Classes

The exact 782-class list can only be extracted from the training notebook where `unique_classes` was created.

### Method 1: Run in Jupyter Notebook

1. Open `non-vit (1).ipynb` in Jupyter
2. Run all cells up to where `unique_classes = sorted(all_df['label'].unique())` is created
3. In a new cell, run:

```python
import json
from pathlib import Path

output_path = Path("/Users/aayankhare/Desktop/D-en-ominators/model_vals.json")

# Save unique_classes as a simple JSON array
with open(output_path, 'w') as f:
    json.dump(unique_classes, f, indent=2)

print(f"✓ Saved {len(unique_classes)} classes to model_vals.json")
print(f"  First 10: {unique_classes[:10]}")
print(f"  Last 10: {unique_classes[-10:]}")
```

4. Restart the model server:
   ```bash
   pkill -f model_server.py
   python3 model_server.py
   ```

### Method 2: Use Helper Script

Copy the code from `extract_782_classes_from_notebook.py` into a notebook cell and run it.

## File Format

`model_vals.json` should be a simple JSON array of class names (Gardiner numbers) in lexicographic order:

```json
["A1", "A10", "A11", ..., "Z9"]
```

This matches exactly what the model expects: `sorted(all_df['label'].unique())`

## Model Server Usage

The model server (`model_server.py`) automatically:
1. Loads `model_vals.json` first (if available)
2. Falls back to `training_class_list.json` if not found
3. Creates class-to-index mapping: `{0: "A1", 1: "A10", ...}`

## Verification

After updating `model_vals.json`:
- Check server logs for: `✓ Loaded 782 classes from model_vals.json`
- Test predictions in `App.html`
- Predictions should now be accurate!

