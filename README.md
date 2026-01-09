# FASHN Human Parser

A SegFormer-B4 model fine-tuned for human parsing with 18 semantic classes, optimized for fashion and virtual try-on applications.

This package provides the exact preprocessing used during model training for maximum accuracy. For quick experimentation, you can also use the model directly via [HuggingFace Hub](https://huggingface.co/fashn-ai/fashn-human-parser).

## Installation

```bash
pip install fashn-human-parser
```

## Usage

```python
from fashn_human_parser import FashnHumanParser

# Initialize parser (automatically uses GPU if available)
parser = FashnHumanParser()

# Run prediction on an image
segmentation = parser.predict("path/to/image.jpg")

# segmentation is a numpy array of shape (H, W) with values 0-17
# representing the 18 semantic classes
```

### Input Formats

The `predict` method accepts:
- File path (string)
- PIL Image
- NumPy array (RGB, uint8 or float32)

**Note:** NumPy arrays must be in RGB format. If using `cv2.imread()` (which returns BGR), convert first:
```python
img = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)
```

### Batch Processing

```python
# Process multiple images in a single forward pass
results = parser.predict([image1, image2, image3])
# results is a list of numpy arrays, one per image
```

### Get Raw Logits

```python
logits = parser.predict(image, return_logits=True)
# logits shape: (1, 18, H, W)
```

## Label Definitions

| ID | Label |
|----|-------|
| 0 | background |
| 1 | face |
| 2 | hair |
| 3 | top |
| 4 | dress |
| 5 | skirt |
| 6 | pants |
| 7 | belt |
| 8 | bag |
| 9 | hat |
| 10 | scarf |
| 11 | glasses |
| 12 | arms |
| 13 | hands |
| 14 | legs |
| 15 | feet |
| 16 | torso |
| 17 | jewelry |

### Accessing Labels

```python
from fashn_human_parser import IDS_TO_LABELS, LABELS_TO_IDS

# Get label name from ID
print(IDS_TO_LABELS[3])  # "top"

# Get ID from label name
print(LABELS_TO_IDS["top"])  # 3
```

## Model Details

- **Architecture**: SegFormer-B4 (MIT-B4 encoder + MLP decoder)
- **Input Size**: 384 x 576 (width x height)
- **Output**: 18-class semantic segmentation mask
- **Base Model**: [nvidia/mit-b4](https://huggingface.co/nvidia/mit-b4)

## Why Use This Package?

**Ease of use:**
- Simple one-line prediction: `parser.predict(image)`
- Output matches input dimensions - no manual resizing needed
- Returns parsed label IDs directly (not raw logits)
- Accepts multiple input formats (file path, PIL, numpy)
- Batch processing with single forward pass: `parser.predict([img1, img2, img3])`

**Utility exports:**
- `IDS_TO_LABELS` / `LABELS_TO_IDS` for label mapping
- `IDENTITY_LABELS` for labels preserved in virtual try-on
- `CATEGORY_TO_BODY_COVERAGE` for clothing category mappings

**Exact preprocessing:**
- Uses `cv2.INTER_AREA` resize (optimal for downsampling)
- Matches the exact preprocessing from model training

The HuggingFace Hub version uses PIL LANCZOS resampling for broader compatibility, which may result in slightly different outputs.

## License

This model inherits the [NVIDIA Source Code License for SegFormer](https://github.com/NVlabs/SegFormer/blob/master/LICENSE). Please review the license terms before use.

## Links

- [HuggingFace Model](https://huggingface.co/fashn-ai/fashn-human-parser)
- [GitHub Repository](https://github.com/fashn-AI/fashn-human-parser)
