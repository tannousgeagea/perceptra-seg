"""Command-line interface for Segmentor."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from perceptra_seg import Segmentor, SegmentorConfig
from perceptra_seg.utils.mask_utils import mask_to_png_bytes


def segment_box_command(args: argparse.Namespace) -> None:
    """Execute segment-box command."""
    config = SegmentorConfig()
    config.runtime.backend = args.backend
    config.model.name = args.model
    config.runtime.device = args.device

    segmentor = Segmentor(config=config)

    # Parse box
    box = tuple(map(int, args.box))

    # Segment
    result = segmentor.segment_from_box(
        image=args.image,
        box=box,
        output_formats=["numpy", "png"],
    )

    # Save output
    if args.out:
        png_bytes = mask_to_png_bytes(result.mask)
        Path(args.out).write_bytes(png_bytes)
        print(f"Saved mask to {args.out}")

    print(f"Score: {result.score:.3f}, Area: {result.area} pixels")
    print(f"Latency: {result.latency_ms:.1f}ms")

    segmentor.close()


def serve_command(args: argparse.Namespace) -> None:
    """Execute serve command."""
    import uvicorn

    from service.main import create_app

    config = SegmentorConfig.from_yaml(args.config) if args.config else SegmentorConfig()
    config.apply_env_overrides()

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Segmentor CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # segment-box command
    box_parser = subparsers.add_parser("segment-box", help="Segment from bounding box")
    box_parser.add_argument("--image", required=True, help="Input image path")
    box_parser.add_argument(
        "--box", required=True, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="Bounding box coordinates"
    )
    box_parser.add_argument("--out", help="Output mask path")
    box_parser.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    box_parser.add_argument("--model", default="sam_v1", choices=["sam_v1", "sam_v2"])
    box_parser.add_argument("--device", default="cuda")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    serve_parser.add_argument("--config", help="Path to config.yaml")

    args = parser.parse_args()

    if args.command == "segment-box":
        segment_box_command(args)
    elif args.command == "serve":
        serve_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()