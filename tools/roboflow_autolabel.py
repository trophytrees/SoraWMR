from __future__ import annotations

import argparse
from pathlib import Path

from sorawm.annotation import auto_label_with_roboflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate YOLO labels via Roboflow workflow.")
    parser.add_argument("--image-dir", type=Path, default=Path("datasets/missed_frames"))
    parser.add_argument("--api-url", default="https://serverless.roboflow.com")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--confidence", type=float, default=None, help="Optional min confidence.")
    parser.add_argument(
        "--no-overwrite", action="store_true", help="Do not overwrite existing .txt files."
    )
    args = parser.parse_args()

    summary = auto_label_with_roboflow(
        image_dir=args.image_dir,
        api_key=args.api_key,
        workspace=args.workspace,
        workflow=args.workflow,
        api_url=args.api_url,
        confidence=args.confidence,
        overwrite=not args.no_overwrite,
        map_class_id=0,
        progress_callback=lambda idx, total: print(
            f"[{idx}/{total}] processed", end="\r"
        ),
    )

    print(
        f"\nCompleted. Processed {summary['processed']} image(s), "
        f"wrote {summary['boxes']} box(es)."
    )


if __name__ == "__main__":
    main()
