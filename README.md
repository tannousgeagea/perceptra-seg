# Perceptra Seg

Production-grade segmentation tool powered by Segment Anything Models (SAM v1, v2 and v3).

Use it three ways:

- **Python SDK** (`perceptra_seg.Segmentor`): run models in-process.
- **REST service** (`docker compose up`): a standalone segmentation server for your infrastructure.
- **REST client** (`perceptra_seg.client.SegmentorClient`): call a deployed server from any Python app (no torch needed).

## Features

- 🚀 **Easy to use**: Simple Python SDK, REST API and API client
- 🔌 **Pluggable backends**: PyTorch and ONNX Runtime support
- 📦 **Multiple models**: SAM v1, SAM v2 and SAM v3 (latest upstream `facebookresearch/sam3`), several served at once
- 🎯 **Flexible prompts**: Boxes, points, text (concepts), visual exemplars
- 📤 **Multiple outputs**: RLE, PNG, polygons, numpy arrays
- ⚡ **Performance**: GPU acceleration, embedding reuse across prompts on the same image
- 🐳 **Ready for production**: Docker Compose deployment, API keys, metrics, structured logging

## Installation

```bash
# SDK with PyTorch backend
pip install "perceptra-seg[torch]"

# + FastAPI server
pip install "perceptra-seg[server,torch]"

# REST client only (talks to a deployed server; no torch required)
pip install perceptra-seg
```

The SAM model code is not on PyPI; install the models you need from their official repos:

```bash
pip install "git+https://github.com/facebookresearch/segment-anything.git"   # sam_v1
pip install "git+https://github.com/facebookresearch/sam2.git"               # sam_v2
pip install "perceptra-seg[sam3]" "git+https://github.com/facebookresearch/sam3.git"  # sam_v3 (latest)
```

SAM3 notes:

- Requires a CUDA GPU (upstream has no CPU path) and PyTorch ≥ 2.7.
- Weights are gated on HuggingFace: request access to [facebook/sam3](https://huggingface.co/facebook/sam3)
  and set `HF_TOKEN`, or point `checkpoint_path` / `SEGMENTOR_SAM3_CHECKPOINT` at a local `sam3.pt`.
- To pick up upstream changes later: `pip install -U --force-reinstall --no-deps "git+https://github.com/facebookresearch/sam3.git"`.

## Quick Start

### Python SDK

```python
from perceptra_seg import Segmentor
import numpy as np

# Initialize
segmentor = Segmentor(
    backend="torch",
    model="sam_v1",
    device="cuda"
)

# Load your image
image = np.array(...)  # or PIL.Image, path, URL

# Segment from bounding box
result = segmentor.segment_from_box(
    image,
    box=(100, 100, 400, 400),
    output_formats=["rle", "png", "polygons"]
)

print(f"Score: {result.score}, Area: {result.area} pixels")
print(f"Mask shape: {result.mask.shape}")

# Segment from points
result = segmentor.segment_from_points(
    image,
    points=[(250, 200, 1), (300, 250, 1)],  # (x, y, label)
    output_formats=["numpy"]
)

segmentor.close()
```

SAM3 concept prompts:

```python
seg = Segmentor(model="sam_v3", device="cuda")
trucks = seg.segment_from_text(image, "truck")                        # every truck
similar = seg.segment_from_exemplar_box(image, (120, 80, 300, 260))  # everything like this object
by_label = seg.segment_from_text_batch(image, ["truck", "wheel"])    # one image encoding, many concepts
```

### REST API

Start the server:

```bash
SEGMENTOR_MODEL_NAMES=sam_v3,sam_v2 uvicorn service.main:app --host 0.0.0.0 --port 8080
```

Call it with the bundled client:

```python
from perceptra_seg.client import SegmentorClient

client = SegmentorClient("http://localhost:8080", api_key="...")
client.health()
client.segment_box("truck.jpg", (100, 100, 400, 400), output_formats=["rle", "polygons"])
client.segment_text("truck.jpg", "wheel", model="sam_v3")
client.segment_points(image_array, [(250, 200, 1)], model="sam_v2")
```

Or with plain HTTP (`image` is base64 or an `http(s)://` URL; `?model=` selects a loaded model):

```bash
curl -X POST "http://localhost:8080/v1/segment/text?model=sam_v3" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -w0 truck.jpg)\", \"text\": \"wheel\"}"
```

Interactive API docs are served at `/docs`.

## Deploying the service (Docker Compose)

Requires Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

```bash
cp .env.example .env      # set HF_TOKEN, SEG_API_KEYS, SEG_MODELS, ...
docker compose up -d --build
curl http://localhost:29086/v1/healthz
```

- **Latest SAM3**: the image installs `facebookresearch/sam3` from git at build time (`SAM3_REF`, default `main`).
  Every `docker compose build` checks upstream and rebuilds that layer when a new commit exists.
  For reproducible production deploys, pin a commit: `SAM3_REF=<sha> docker compose build`.
  `/v1/healthz` reports the deployed commit under `build.sam3_commit`.
- **Weights** are not baked into the image. They download on first start into the `models` volume and
  persist across restarts and rebuilds. To warm the cache before a deploy:
  `docker compose run --rm perceptra-seg python scripts/download_sam3.py`.
- **Auth**: set `SEG_API_KEYS=key1,key2`; clients send `Authorization: Bearer <key>`. `/v1/healthz` and `/metrics` stay public.
- **Models**: `SEG_MODELS=sam_v3,sam_v2` loads both; the first is the default.
- **Concurrency**: each model runs one inference at a time (models keep per-image state); different models
  run concurrently. Image decoding/fetching, validation and post-processing (RLE, polygons, PNG) run outside
  that lock, so concurrent requests overlap everywhere except the GPU call.
  - Each model admits at most `SEGMENTOR_SERVER_MAX_QUEUE_PER_MODEL` requests (running + waiting, default 16);
    beyond that the service answers **429** with `Retry-After`.
  - A request that waits longer than `SEGMENTOR_SERVER_REQUEST_TIMEOUT` seconds (default 30) for its model
    gets **503** with `Retry-After`.
  - `SegmentorClient` retries 429/503 (`max_retries=2` by default, `0` disables it).
  - `/metrics`: `perceptra_queue_depth`, `perceptra_gpu_wait_seconds`, `perceptra_inference_seconds`,
    `perceptra_rejected_total`, `perceptra_image_repeat_total`; `/v1/healthz` shows `pending`/`max_pending` per model.
- **Limits**: `SEGMENTOR_SERVER_MAX_IMAGE_SIZE_MB` / `SEGMENTOR_SERVER_MAX_IMAGE_DIMENSION` (413 when exceeded).
- Any config value can be overridden with `SEGMENTOR_<SECTION>_<FIELD>`, or mount a YAML file and set `SEGMENTOR_CONFIG=/path/config.yaml`.

| Endpoint | Models | Purpose |
|---|---|---|
| `POST /v1/segment/box` | all | One mask from a box |
| `POST /v1/segment/points` | all | One mask from positive/negative points |
| `POST /v1/segment` | all | Several boxes and/or points (`strategy`: largest, merge, all) |
| `POST /v1/segment/text` | sam_v3 | All instances of a text concept (optional `box` exemplar) |
| `POST /v1/segment/text/batch` | sam_v3 | Several concepts, one image encoding |
| `POST /v1/segment/exemplar` | sam_v3 | All objects similar to an exemplar box |
| `POST /v1/segment/auto` | sam_v1, sam_v2 | Everything in the image, no prompt |
| `GET /v1/healthz` | | Status, loaded models, build info |
| `GET /metrics` | | Prometheus metrics |

## Configuration

Edit `config.yaml` or use environment variables:

```yaml
model:
  name: "sam_v1"  # sam_v1 | sam_v2
  encoder_variant: "vit_h"  # vit_h | vit_l | vit_b
  checkpoint_path: null  # Auto-download if null

runtime:
  backend: "torch"  # torch | onnx
  device: "cuda"  # cuda | cpu
  precision: "fp32"  # fp16 | bf16 | fp32

server:
  host: "0.0.0.0"
  port: 8080
  api_keys: []  # Add keys for authentication
```

Environment overrides:

```bash
export SEGMENTOR_RUNTIME_DEVICE=cpu
export SEGMENTOR_MODEL_NAME=sam_v2
```

## CLI Usage

```bash
# Segment from bounding box
segmentor-cli segment-box \
  --image path/to/image.jpg \
  --box 10 20 200 240 \
  --out mask.png \
  --backend torch \
  --model sam_v1

# Start server
segmentor-cli serve --config config.yaml
```

## Model Weights & Licenses

This tool uses Meta's Segment Anything Models. Model weights are licensed under Apache 2.0.

**SAM v1 checkpoints** (auto-downloaded):
- vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
- vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

**SAM v2 checkpoints** (auto-downloaded):
- hiera_large: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
- hiera_base_plus: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

Weights are downloaded to `~/.cache/segmentor/` on first use.

**Important**: Review Meta's license terms before commercial use.

## Development

```bash
# Clone repository
git clone https://github.com/tannousgeagea/perceptra-seg.git
cd segmentor

# Install in development mode
pip install -e .[dev,all]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=segmentor

# Run linters
black segmentor/ service/
isort segmentor/ service/
ruff check segmentor/ service/
mypy segmentor/ service/

# Build documentation
cd docs && mkdocs serve
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Segmentor SDK                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │     segment_from_box / segment_from_points      │   │
│  └───────────────────┬─────────────────────────────┘   │
│                      │                                   │
│  ┌───────────────────▼─────────────────────────────┐   │
│  │        Backend Abstraction Layer                 │   │
│  │  ┌──────────┬──────────┬──────────┬──────────┐  │   │
│  │  │ Torch    │ Torch    │  ONNX    │  ONNX    │  │   │
│  │  │ SAM v1   │ SAM v2   │  SAM v1  │  SAM v2  │  │   │
│  │  └──────────┴──────────┴──────────┴──────────┘  │   │
│  └───────────────────┬─────────────────────────────┘   │
│                      │                                   │
│  ┌───────────────────▼─────────────────────────────┐   │
│  │      Utilities: Image I/O, Mask Utils,          │   │
│  │      Tiling, Caching, Postprocessing            │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         │
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Service                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  /v1/segment/box  │  /v1/segment/points         │   │
│  │  /v1/segment      │  /v1/healthz  │  /metrics   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  Auth • CORS • Logging • Metrics • Rate Limiting        │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Backend Protocol Pattern**: Uses Python's `Protocol` for type-safe backend abstraction, allowing new backends to be added without modifying core logic.

2. **Configuration-Driven**: Single YAML config controls all aspects (model, runtime, outputs), with environment variable overrides for deployment flexibility.

3. **Separation of Concerns**: 
   - `core.py`: High-level API and orchestration
   - `backends/`: Model-specific inference logic
   - `utils/`: Reusable image/mask operations
   - `service/`: HTTP layer completely separate from SDK

4. **Output Flexibility**: Supports multiple output formats (RLE, PNG, polygons, numpy) generated on-demand to minimize memory usage.

5. **Caching Strategy**: LRU cache for image embeddings (expensive to compute), keyed by image hash for exact-match speedups.

6. **Error Handling**: Custom exception hierarchy maps to appropriate HTTP status codes in the service layer.

7. **ONNX Placeholder**: ONNX backends are stubs requiring pre-exported models, as SAM's official ONNX export is complex and model-specific.

## API Reference

### Python SDK

#### `Segmentor`

Main class for segmentation operations.

**Constructor**:
```python
Segmentor(
    config: SegmentorConfig | None = None,
    **kwargs
)
```

**Methods**:
- `segment_from_box(image, box, *, output_formats, return_overlay)` → `SegmentationResult`
- `segment_from_points(image, points, *, output_formats, return_overlay)` → `SegmentationResult`
- `segment(image, boxes, points, *, strategy, output_formats, return_overlay)` → `list[SegmentationResult]`
- `warmup(image_size)` → `None`
- `set_backend(backend_name)` → `None`
- `close()` → `None`

#### `SegmentationResult`

Result object containing:
- `mask`: numpy array (HxW) if 'numpy' in output_formats
- `rle`: COCO RLE dict if 'rle' in output_formats
- `polygons`: List of polygon contours if 'polygons' in output_formats
- `png_bytes`: PNG-encoded mask if 'png' in output_formats
- `score`: Confidence score (0-1)
- `area`: Number of pixels in mask
- `bbox`: Bounding box (x1, y1, x2, y2)
- `latency_ms`: Processing time
- `model_info`: Dict with model metadata
- `request_id`: Unique request identifier

### REST API

#### `POST /v1/segment/box`

Segment from bounding box.

**Request**:
```json
{
  "image": "base64_string_or_url",
  "box": [x1, y1, x2, y2],
  "output_formats": ["rle", "png", "polygons"],
  "strategy": "largest"
}
```

**Response**:
```json
{
  "rle": {"size": [H, W], "counts": [...]},
  "png_base64": "...",
  "polygons": [[[x1, y1], [x2, y2], ...]],
  "score": 0.95,
  "area": 12345,
  "bbox": [x1, y1, x2, y2],
  "latency_ms": 123.4,
  "model_info": {"name": "sam_v1", "backend": "torch"},
  "request_id": "uuid"
}
```

#### `POST /v1/segment/points`

Segment from point prompts.

**Request**:
```json
{
  "image": "base64_string_or_url",
  "points": [
    {"x": 100, "y": 200, "label": 1},
    {"x": 150, "y": 220, "label": 1}
  ],
  "output_formats": ["rle"]
}
```

#### `POST /v1/segment`

General segmentation supporting boxes and/or points.

**Request**:
```json
{
  "image": "base64_string_or_url",
  "boxes": [[x1, y1, x2, y2], ...],
  "points": [{"x": 100, "y": 200, "label": 1}, ...],
  "strategy": "merge",
  "output_formats": ["rle"]
}
```

**Strategies**:
- `"largest"`: Return only the largest mask
- `"merge"`: Union all masks into one
- `"all"`: Return all masks as separate results

#### `GET /v1/healthz`

Health check endpoint.

**Response**: `{"status": "ok"}`

#### `GET /metrics`

Prometheus metrics endpoint.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=segmentor --cov-report=html

# Run specific test file
pytest tests/test_core.py -v

# Run with markers
pytest -m "not slow"
```

Test coverage includes:
- ✅ Core segmentation logic
- ✅ Backend switching
- ✅ Input validation
- ✅ Output format conversion
- ✅ REST API endpoints
- ✅ Error handling
- ✅ Configuration loading

## Performance Tips

1. **Use GPU**: Set `device: "cuda"` for 10-50x speedup
2. **Enable caching**: Keep `cache.enabled: true` for repeated images
3. **Batch processing**: Use `segment()` with multiple boxes instead of separate calls
4. **FP16 precision**: Set `precision: "fp16"` on GPU for 2x speedup with minimal quality loss
5. **Warm up**: Call `warmup()` before processing to avoid first-call overhead
6. **Tiling**: Enable for very large images (>4K) to avoid OOM

## Troubleshooting

### CUDA out of memory
- Reduce `runtime.batch_size`
- Enable `tiling.enabled: true`
- Use smaller model variant (`vit_b` instead of `vit_h`)
- Use `precision: "fp16"`

### Slow inference
- Ensure GPU is being used: check `torch.cuda.is_available()`
- Warm up the model first
- Enable caching for repeated images
- Use FP16 precision

### Import errors
- Ensure correct extras installed: `pip install perceptra-seg[torch]`
- For SAM v1: `pip install git+https://github.com/facebookresearch/segment-anything.git`
- For SAM v2: `pip install git+https://github.com/facebookresearch/sam2.git`
- For SAM v3: `pip install "perceptra-seg[sam3]" git+https://github.com/facebookresearch/sam3.git`

### Model download fails
- Check internet connection
- Manually download from URLs in README and set `checkpoint_path` in config
- Verify disk space in `~/.cache/segmentor/`

## Roadmap

- [ ] HQ-SAM and MobileSAM backend support
- [ ] Complete ONNX backend implementation
- [ ] Video segmentation support (SAM 2 temporal)
- [ ] Automatic mask quality filtering
- [ ] Batch API endpoint
- [ ] WebSocket streaming API
- [ ] Triton Inference Server backend
- [ ] Model quantization (INT8)
- [ ] Multi-GPU support

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and coverage >80%
5. Run pre-commit hooks
6. Submit a pull request

## License

Apache License 2.0 - see LICENSE file.

This project uses SAM models from Meta, which are also licensed under Apache 2.0.

## Citation

If you use this tool in research, please cite the original SAM papers:

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv:2408.00714},
  year={2024}
}
```

## Contact

- Issues: https://github.com/tannousgeagea/perceptra-seg/issues
- Discussions: https://github.com/tannousgeagea/perceptra-seg/discussions
- Email: team@example.com

---

Built with ❤️ by the Segmentor team# Segmentor: Production-Grade Segmentation Tool

A modular, high-performance segmentation library and microservice powered by Segment Anything Models (SAM v1 & v2).

## Project Structure

```
perceptra-seg/
├── pyproject.toml
├── README.md
├── config.yaml
├── Dockerfile
├── docker-compose.yml
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml
├── perceptra_seg/
│   ├── __init__.py
│   ├── core.py
│   ├── config.py
│   ├── models.py
│   ├── exceptions.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── torch_sam_v1.py
│   │   ├── torch_sam_v2.py
│   │   ├── onnx_sam_v1.py
│   │   └── onnx_sam_v2.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_io.py
│   │   ├── mask_utils.py
│   │   ├── tiling.py
│   │   └── cache.py
│   ├── cli.py
│   └── quickstart.py
├── service/
│   ├── __init__.py
│   ├── main.py
│   ├── routes.py
│   └── middleware.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core.py
│   ├── test_backends.py
│   ├── test_utils.py
│   └── test_service.py
└── docs/
    ├── index.md
    ├── quickstart.md
    ├── api.md
    └── config.md
```

```markdown
## For Package Developers

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/tannousgeagea/perceptra-seg.git
cd perceptra-seg

# Install in editable mode with all dependencies
pip install -e .[all]

# Install pre-commit hooks
pre-commit install
```

### Using perceptra-seg in Your Project

**Install from PyPI (when published)**:
```bash
pip install perceptra-seg[torch]
```

**Install from GitHub**:
```bash
pip install git+https://github.com/tannousgeagea/perceptra-seg.git
```

**Install specific version**:
```bash
pip install perceptra-seg[torch]==0.1.0
```

**Add to requirements.txt**:
```
perceptra-seg[torch]>=0.1.0
```

**Add to pyproject.toml**:
```toml
dependencies = [
    "perceptra-seg[torch]>=0.1.0",
]
```

### Quick Integration Example

```python
# Add to your project
from perceptra_seg import Segmentor

class MyImageProcessor:
    def __init__(self):
        self.segmentor = Segmentor(backend="torch", device="cuda")
    
    def process(self, image, box):
        result = self.segmentor.segment_from_box(image, box)
        return result.mask
```

### API Stability

- **Stable**: Core API (`Segmentor`, `SegmentationResult`, `SegmentorConfig`)
- **Beta**: Service endpoints may change in minor versions
- **Experimental**: ONNX backends, tiling features

### Version Compatibility

| Segmentor Version | Python | PyTorch | NumPy |
|-------------------|--------|---------|-------|
| 0.1.x             | 3.10+  | 2.0+    | 1.24+ |
