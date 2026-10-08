```markdown
# Quick Start Guide

## Installation

```bash
pip install segmentor[torch,server]
```

## First Steps

### 1. Verify Installation

```bash
python -m segmentor.quickstart
```

This will check your installation and run a basic test.

### 2. Basic Usage

```python
from perceptra_seg import Segmentor
from PIL import Image

# Initialize
seg = Segmentor(backend="torch", model="sam_v1", device="cuda")

# Load image
image = Image.open("photo.jpg")

# Segment an object by drawing a box around it
result = seg.segment_from_box(
    image,
    box=(100, 150, 400, 500),  # x1, y1, x2, y2
    output_formats=["numpy", "png"]
)

# Save result
Image.fromarray(result.mask * 255).save("mask.png")

print(f"Confidence: {result.score:.2f}")
print(f"Area: {result.area} pixels")

seg.close()
```

### 3. Click on Object

```python
# Segment by clicking points on the object
result = seg.segment_from_points(
    image,
    points=[
        (250, 300, 1),  # positive point (x, y, 1)
        (260, 310, 1),  # another positive point
        (100, 100, 0),  # negative point (background)
    ],
    output_formats=["polygons", "rle"]
)

# Get polygon contours
for polygon in result.polygons:
    print(f"Polygon with {len(polygon)} points")
```

### 4. Start REST API

```bash
segmentor-cli serve
```

Or with custom config:

```bash
segmentor-cli serve --config my_config.yaml
```

Test it:

```bash
curl http://localhost:8080/v1/healthz
```

## Examples

### Example 1: Batch Processing

```python
from perceptra_seg import Segmentor
from pathlib import Path

seg = Segmentor(backend="torch", device="cuda")

image_dir = Path("images/")
boxes = {
    "cat.jpg": (50, 50, 300, 400),
    "dog.jpg": (100, 80, 450, 520),
    "bird.jpg": (200, 150, 350, 300),
}

for img_name, box in boxes.items():
    img_path = image_dir / img_name
    result = seg.segment_from_box(img_path, box, output_formats=["png"])
    
    output_path = image_dir / f"mask_{img_name}"
    Path(output_path).write_bytes(result.png_bytes)
    print(f"Processed {img_name}: score={result.score:.3f}")

seg.close()
```

### Example 2: REST API Client

```python
import requests
import base64
from pathlib import Path

# Prepare image
with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8080/v1/segment/box",
    json={
        "image": img_b64,
        "box": [100, 100, 400, 400],
        "output_formats": ["rle", "png"]
    }
)

result = response.json()
print(f"Score: {result['score']}, Area: {result['area']}")

# Decode mask
mask_bytes = base64.b64decode(result['png_base64'])
Path("output_mask.png").write_bytes(mask_bytes)
```

### Example 3: Multiple Objects

```python
from perceptra_seg import Segmentor

seg = Segmentor(backend="torch", device="cuda")

# Segment multiple objects in same image
results = seg.segment(
    image="scene.jpg",
    boxes=[
        (50, 50, 200, 200),    # object 1
        (300, 100, 500, 400),  # object 2
        (100, 300, 350, 550),  # object 3
    ],
    strategy="all",  # Return all masks separately
    output_formats=["numpy"]
)

print(f"Found {len(results)} objects")
for i, result in enumerate(results):
    print(f"Object {i+1}: area={result.area}, score={result.score:.3f}")

seg.close()
```

### Example 4: Docker Deployment

```bash
# Build image
docker build -t segmentor:latest .

# Run with GPU (weights persist in the perceptra-models volume)
docker run --gpus all -p 8080:8080 \
  -e SEGMENTOR_RUNTIME_DEVICE=cuda \
  -e SEGMENTOR_MODEL_NAME=sam_v2 \
  -v perceptra-models:/models \
  segmentor:latest

# Test
curl -X POST http://localhost:8080/v1/healthz
```

## Next Steps

- Read the [API Documentation](api.md)
- Explore [Configuration Options](config.md)
- Check out [Advanced Examples](examples.md)
```
