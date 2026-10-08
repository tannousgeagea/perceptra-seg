```markdown
# Configuration Reference

Complete reference for `config.yaml`.

## Model Configuration

```yaml
model:
  name: "sam_v1"  # Model version
  encoder_variant: "vit_h"  # Encoder architecture
  checkpoint_path: null  # Path to weights file
```

### `model.name`
- **Type**: `string`
- **Options**: `"sam_v1"`, `"sam_v2"`
- **Default**: `"sam_v1"`
- **Description**: Which SAM model version to use

### `model.encoder_variant`
- **Type**: `string`
- **Options**: `"vit_h"`, `"vit_l"`, `"vit_b"`
- **Default**: `"vit_h"`
- **Description**: ViT encoder size (huge > large > base)

### `model.checkpoint_path`
- **Type**: `string | null`
- **Default**: `null`
- **Description**: Path to model weights. If `null`, weights are auto-downloaded to `~/.cache/segmentor/`

## Runtime Configuration

```yaml
runtime:
  backend: "torch"
  device: "cuda"
  precision: "fp32"
  batch_size: 1
  deterministic: true
  seed: 42
```

### `runtime.backend`
- **Type**: `string`
- **Options**: `"torch"`, `"onnx"`
- **Default**: `"torch"`
- **Description**: Inference backend

### `runtime.device`
- **Type**: `string`
- **Options**: `"cuda"`, `"cpu"`, `"cuda:0"`, `"cuda:1"`, etc.
- **Default**: `"cuda"`
- **Description**: Device for inference

### `runtime.precision`
- **Type**: `string`
- **Options**: `"fp32"`, `"fp16"`, `"bf16"`
- **Default**: `"fp32"`
- **Description**: Floating point precision. FP16 can provide 2x speedup on GPU.

### `runtime.batch_size`
- **Type**: `integer`
- **Default**: `1`
- **Description**: Batch size for processing multiple prompts

### `runtime.deterministic`
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Enable deterministic operations for reproducibility

### `runtime.seed`
- **Type**: `integer`
- **Default**: `42`
- **Description**: Random seed for deterministic behavior

## Tiling Configuration

```yaml
tiling:
  enabled: false
  tile_size: 1024
  stride: 256
  blend_mode: "linear"
```

### `tiling.enabled`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: Enable tiling for large images

### `tiling.tile_size`
- **Type**: `integer`
- **Default**: `1024`
- **Description**: Size of each tile in pixels

### `tiling.stride`
- **Type**: `integer`
- **Default**: `256`
- **Description**: Stride between tiles (overlap = tile_size - stride)

### `tiling.blend_mode`
- **Type**: `string`
- **Options**: `"linear"`, `"average"`
- **Default**: `"linear"`
- **Description**: How to blend overlapping tile regions

## Output Configuration

```yaml
outputs:
  default_formats: ["rle"]
  include_overlay: false
  min_area_ratio: 0.001
```

### `outputs.default_formats`
- **Type**: `list[string]`
- **Options**: `"rle"`, `"png"`, `"polygons"`, `"numpy"`
- **Default**: `["rle"]`
- **Description**: Default output formats

### `outputs.include_overlay`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: Include visualization overlay with mask

### `outputs.min_area_ratio`
- **Type**: `float`
- **Default**: `0.001`
- **Description**: Minimum mask area as ratio of image size

## Thresholds

```yaml
thresholds:
  mask_threshold: 0.5
  iou_threshold: 0.88
```

### `thresholds.mask_threshold`
- **Type**: `float`
- **Range**: `0.0` to `1.0`
- **Default**: `0.5`
- **Description**: Threshold for binarizing mask predictions

### `thresholds.iou_threshold`
- **Type**: `float`
- **Range**: `0.0` to `1.0`
- **Default**: `0.88`
- **Description**: IoU threshold for mask selection

## Postprocessing

```yaml
postprocess:
  remove_small_components: true
  morphological_closing: false
  closing_kernel_size: 3
```

### `postprocess.remove_small_components`
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Remove small disconnected mask regions

### `postprocess.morphological_closing`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: Apply morphological closing to fill holes

### `postprocess.closing_kernel_size`
- **Type**: `integer`
- **Default**: `3`
- **Description**: Kernel size for morphological operations

## Cache Configuration

```yaml
cache:
  enabled: true
  max_items: 100
  ttl_seconds: 3600
```

### `cache.enabled`
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Enable LRU cache for embeddings

### `cache.max_items`
- **Type**: `integer`
- **Default**: `100`
- **Description**: Maximum cached items

### `cache.ttl_seconds`
- **Type**: `integer`
- **Default**: `3600`
- **Description**: Cache entry lifetime in seconds

## Server Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 1
  cors_origins: ["*"]
  api_keys: []
  max_image_size_mb: 20
  max_image_dimension: 8000
  request_timeout: 30
```

### `server.host`
- **Type**: `string`
- **Default**: `"0.0.0.0"`
- **Description**: Server bind address

### `server.port`
- **Type**: `integer`
- **Default**: `8080`
- **Description**: Server port

### `server.workers`
- **Type**: `integer`
- **Default**: `1`
- **Description**: Number of worker processes

### `server.cors_origins`
- **Type**: `list[string]`
- **Default**: `["*"]`
- **Description**: Allowed CORS origins

### `server.api_keys`
- **Type**: `list[string]`
- **Default**: `[]`
- **Description**: API keys for authentication. Empty = no auth.

### `server.max_image_size_mb`
- **Type**: `integer`
- **Default**: `20`
- **Description**: Maximum image upload size in MB

### `server.max_image_dimension`
- **Type**: `integer`
- **Default**: `8000`
- **Description**: Maximum image dimension in pixels

### `server.request_timeout`
- **Type**: `integer`
- **Default**: `30`
- **Description**: Request timeout in seconds

## Logging Configuration

```yaml
logging:
  level: "INFO"
  format: "json"
  log_file: null
```

### `logging.level`
- **Type**: `string`
- **Options**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`
- **Default**: `"INFO"`
- **Description**: Logging level

### `logging.format`
- **Type**: `string`
- **Options**: `"json"`, `"text"`
- **Default**: `"json"`
- **Description**: Log output format

### `logging.log_file`
- **Type**: `string | null`
- **Default**: `null`
- **Description**: Path to log file. `null` = stdout only

## Environment Variables

Override any config value using environment variables:

```bash
export SEGMENTOR_RUNTIME_DEVICE=cpu
export SEGMENTOR_MODEL_NAME=sam_v2
export SEGMENTOR_SERVER_PORT=9000
export SEGMENTOR_CACHE_ENABLED=false
```

Format: `SEGMENTOR_<SECTION>_<FIELD>=<value>`
```

---

## Usage Instructions

### Running Locally

```bash
# 1. Install dependencies
pip install -e .[torch,server,dev]

# 2. Run quickstart check
python -m segmentor.quickstart

# 3. Test SDK
python -c "
from perceptra_seg import Segmentor
import numpy as np

seg = Segmentor(backend='torch', device='cpu')
img = np.zeros((480, 640, 3), dtype=np.uint8)
result = seg.segment_from_box(img, (100, 100, 300, 300))
print(f'Score: {result.score}, Area: {result.area}')
seg.close()
"

# 4. Start server
segmentor-cli serve --config config.yaml

# 5. Test API
curl http://localhost:8080/v1/healthz

# 6. Run tests
pytest tests/ -v
```

### Running with Docker

```bash
# Recommended: docker compose (see README "Deploying the service")
cp .env.example .env && docker compose up -d --build

# Or plain docker
docker build -t perceptra-seg .
docker run --gpus all -p 8080:8080 \
  -e SEGMENTOR_MODEL_NAMES=sam_v3 -e HF_TOKEN=$HF_TOKEN \
  -v perceptra-models:/models \
  perceptra-seg

# With custom config
docker run -p 8080:8080 \
  -v $(pwd)/my_config.yaml:/app/config.yaml \
  segmentor:cpu

# Test
curl http://localhost:8080/v1/healthz
```

### Example cURL Requests

```bash
# Segment from box
curl -X POST http://localhost:8080/v1/segment/box \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'"$(base64 -w 0 image.jpg)"'",
    "box": [100, 100, 400, 400],
    "output_formats": ["rle", "png"]
  }' | jq .

# Segment from points
curl -X POST http://localhost:8080/v1/segment/points \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'"$(base64 -w 0 image.jpg)"'",
    "points": [
      {"x": 250, "y": 200, "label": 1},
      {"x": 300, "y": 250, "label": 1}
    ],
    "output_formats": ["polygons"]
  }' | jq .

# With authentication
curl -X POST http://localhost:8080/v1/segment/box \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '...'
```

### Python Client Example

```python
import requests
import base64
from pathlib import Path
from PIL import Image
import io

class SegmentorClient:
    def __init__(self, base_url="http://localhost:8080", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def segment_box(self, image_path, box, output_formats=["rle"]):
        # Load and encode image
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        # Make request
        response = requests.post(
            f"{self.base_url}/v1/segment/box",
            headers=self.headers,
            json={
                "image": img_b64,
                "box": list(box),
                "output_formats": output_formats
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_mask_image(self, result):
        """Decode PNG mask from result."""
        if "png_base64" not in result:
            return None
        mask_bytes = base64.b64decode(result["png_base64"])
        return Image.open(io.BytesIO(mask_bytes))

# Usage
client = SegmentorClient()
result = client.segment_box("photo.jpg", (100, 100, 400, 400), ["png", "rle"])
mask_img = client.get_mask_image(result)
mask_img.save("output_mask.png")
print(f"Score: {result['score']:.3f}, Area: {result['area']}")
```