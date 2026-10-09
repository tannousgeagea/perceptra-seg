"""
End-to-end smoke test for a running perceptra-seg service.

Runs every endpoint against every loaded model, checks that unsupported prompt types
are rejected cleanly (400, never 500), and saves an overlay PNG per call.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --url http://seg-server:29086 --api-key KEY \\
        --image assets/images/truck.jpg --text wheel --batch wheel window door

Exit code is non-zero if any check fails.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perceptra_seg.client import SegmentorAPIError, SegmentorClient  # noqa: E402

# Which models are expected to support each operation.
CONCEPT_MODELS = {"sam_v3"}
AUTO_MODELS = {"sam_v1", "sam_v2"}

COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128)]

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def overlay(image: Image.Image, results: list[dict[str, Any]], path: Path,
            prompt_box: tuple[int, int, int, int] | None = None) -> None:
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    for i, r in enumerate(results):
        color = COLORS[i % len(COLORS)]
        for poly in r.get("polygons") or []:
            if len(poly) >= 3:
                draw.polygon([tuple(p) for p in poly], fill=(*color, 90), outline=(*color, 255))
        if r.get("bbox"):
            draw.rectangle(r["bbox"], outline=(*color, 255), width=2)
            draw.text((r["bbox"][0] + 3, r["bbox"][1] + 2), f"{r['score']:.2f}", fill=(*color, 255))
    if prompt_box:
        draw.rectangle(prompt_box, outline=(255, 255, 0, 255), width=3)
    img.save(path)


class Runner:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.rows: list[tuple[str, str, str, str]] = []

    def check(
        self,
        model: str,
        name: str,
        supported: bool,
        call: Callable[[], Any],
        image: Image.Image,
        prompt_box: tuple[int, int, int, int] | None = None,
    ) -> Any:
        start = time.perf_counter()
        try:
            result = call()
        except SegmentorAPIError as e:
            ms = (time.perf_counter() - start) * 1000
            if not supported and e.status_code == 400:
                self.rows.append((model, name, "PASS", f"unsupported -> 400 as expected ({ms:.0f} ms)"))
            else:
                self.rows.append((model, name, "FAIL", f"HTTP {e.status_code}: {str(e.detail)[:120]}"))
            return None
        except Exception as e:  # connection errors, timeouts
            self.rows.append((model, name, "FAIL", f"{type(e).__name__}: {e}"))
            return None

        ms = (time.perf_counter() - start) * 1000
        if not supported:
            self.rows.append((model, name, "FAIL", "expected 400 (unsupported) but call succeeded"))
            return result

        # Single result (dict with "score"), list of results, or text batch ({prompt: [results]}).
        is_batch = isinstance(result, dict) and "score" not in result
        if is_batch:
            items = [r for v in result.values() for r in v]
        else:
            items = result if isinstance(result, list) else [result]
        detail = f"{len(items)} mask(s) in {ms:.0f} ms"
        empty = sum(1 for r in items if not r.get("area"))
        if empty:
            detail += f" ({empty} empty after post-processing)"
        if is_batch:
            detail += " | " + ", ".join(f"{k}: {len(v)}" for k, v in result.items())
        elif items:
            detail += " | scores " + ", ".join(f"{r['score']:.2f}" for r in items[:6])
        status = "PASS" if len(items) > empty else "WARN"  # WARN = worked, but nothing found
        self.rows.append((model, name, status, detail))
        overlay(image, items, self.out_dir / f"{model}_{name}.png", prompt_box)
        return result

    def report(self) -> int:
        width = max(len(f"{m} {n}") for m, n, _, _ in self.rows) + 2
        color = {"PASS": GREEN, "FAIL": RED, "WARN": YELLOW}
        print()
        for model, name, status, detail in self.rows:
            print(f"  {color[status]}{status}{RESET}  {f'{model} {name}':<{width}} {detail}")
        failed = sum(s == "FAIL" for _, _, s, _ in self.rows)
        warned = sum(s == "WARN" for _, _, s, _ in self.rows)
        print(f"\n{len(self.rows) - failed} passed, {failed} failed ({warned} returned no masks)."
              f"  Overlays: {self.out_dir}/")
        return 1 if failed else 0


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:29086")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--image", default=str(root / "assets/images/truck.jpg"))
    parser.add_argument("--text", default="wheel", help="text prompt")
    parser.add_argument("--batch", nargs="+", default=["wheel", "window", "door"], help="prompts for text batch")
    parser.add_argument("--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                        help="box prompt (default: centre of the image)")
    parser.add_argument("--models", nargs="+", help="models to test (default: all loaded)")
    parser.add_argument("--out", default="smoke_test_output")
    parser.add_argument("--skip-auto", action="store_true", help="skip auto-segmentation (slow)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    box = tuple(args.box) if args.box else (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

    client = SegmentorClient(args.url, api_key=args.api_key, timeout=300)
    health = client.health()
    print(f"Service {args.url}: status={health['status']} build={health.get('build')}")
    models = args.models or list(health["models"])
    if not models:
        print(f"{RED}No models loaded{RESET}")
        return 1
    print(f"Testing models {models} on {args.image} ({w}x{h}), box={box}")

    run = Runner(out_dir)
    fmt = ["polygons", "rle"]
    for model in models:
        concept = model in CONCEPT_MODELS

        # Geometric prompts — every model.
        run.check(model, "box", True, lambda: client.segment_box(image, box, output_formats=fmt, model=model),
                  image, box)
        cx, cy = w // 2, h // 2
        run.check(model, "points", True,
                  lambda: client.segment_points(image, [(cx, cy, 1)], output_formats=fmt, model=model), image)
        boxes = [box, (0, 0, w // 2, h // 2)]
        run.check(model, "segment_multi_box_all", True,
                  lambda: client.segment(image, boxes=boxes, strategy="all", output_formats=fmt, model=model),
                  image)
        run.check(model, "segment_multi_box_merge", True,
                  lambda: client.segment(image, boxes=boxes, strategy="merge",
                                         output_formats=["numpy", *fmt], model=model), image)

        # Concept prompts — sam_v3 only.
        text_results = run.check(model, "text", concept,
                                 lambda: client.segment_text(image, args.text, output_formats=fmt, model=model),
                                 image)
        # Exemplar: reuse the best text match as the example object, if there is one.
        found = [r for r in text_results or [] if r.get("bbox")]
        exemplar = tuple(found[0]["bbox"]) if found else box
        run.check(model, "text_with_box", concept,
                  lambda: client.segment_text(image, args.text, box=exemplar, output_formats=fmt, model=model),
                  image, exemplar)
        run.check(model, "text_batch", concept,
                  lambda: client.segment_text_batch(image, args.batch, output_formats=fmt, model=model), image)
        run.check(model, "exemplar", concept,
                  lambda: client.segment_exemplar(image, exemplar, output_formats=fmt, model=model),
                  image, exemplar)

        # Automatic — sam_v1 / sam_v2 only.
        if not args.skip_auto:
            run.check(model, "auto", model in AUTO_MODELS,
                      lambda: client.segment_auto(image, points_per_side=16, output_formats=fmt, model=model),
                      image)

    return run.report()


if __name__ == "__main__":
    sys.exit(main())
