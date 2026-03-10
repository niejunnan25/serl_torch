#!/usr/bin/env python3
"""Download HuggingFace ResNet pretrained weights for offline use.

Usage
-----
    # Download default models (resnet-18 + resnet-50):
    python tools/download_resnet.py

    # Download a specific model:
    python tools/download_resnet.py --models microsoft/resnet-18

    # Download multiple models:
    python tools/download_resnet.py --models microsoft/resnet-18 microsoft/resnet-50 microsoft/resnet-101

    # Specify output directory:
    python tools/download_resnet.py --output-dir /path/to/pretrained_models

    # Use HuggingFace mirror (for environments with limited Hub access):
    python tools/download_resnet.py --mirror https://hf-mirror.com

After downloading, set ``model_name`` in your YAML config to the local path:
    sac:
      resnet:
        model_name: pretrained_models/microsoft--resnet-18
"""

import argparse
import os
import sys


def _project_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def download_model(
    model_id: str,
    output_dir: str,
    mirror: str | None = None,
) -> str:
    """Download a single HuggingFace model and return its local path."""
    from huggingface_hub import snapshot_download

    local_dir = os.path.join(output_dir, model_id.replace("/", "--"))
    os.makedirs(local_dir, exist_ok=True)

    kwargs = dict(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=["*.h5", "flax_model*", "*.ot", "*.msgpack"],
        resume_download=True,
        max_workers=4,
    )
    if mirror:
        kwargs["endpoint"] = mirror

    print(f"[download] {model_id} -> {local_dir}")
    snapshot_download(**kwargs)
    return local_dir


def verify_model(local_dir: str) -> bool:
    """Verify that a downloaded model can be loaded by transformers."""
    from transformers import ResNetModel

    print(f"[verify]   Loading from {local_dir} ... ", end="", flush=True)
    try:
        model = ResNetModel.from_pretrained(local_dir)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"OK  ({n_params:,} params)")
        return True
    except Exception as e:
        print(f"FAILED ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace ResNet weights for offline use.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["microsoft/resnet-18", "microsoft/resnet-50"],
        help="HuggingFace model IDs to download (default: resnet-18 + resnet-50)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_project_root(), "pretrained_models"),
        help="Directory to save downloaded models",
    )
    parser.add_argument(
        "--mirror",
        default=None,
        help="HuggingFace mirror URL (e.g. https://hf-mirror.com)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-download verification",
    )
    args = parser.parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.mirror

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for model_id in args.models:
        try:
            local_dir = download_model(model_id, args.output_dir, mirror=args.mirror)
            if args.skip_verify:
                results[model_id] = True
            else:
                results[model_id] = verify_model(local_dir)
        except Exception as e:
            print(f"[error]    {model_id}: {e}")
            results[model_id] = False

    print("\n" + "=" * 50)
    print("Summary:")
    all_ok = True
    for model_id, ok in results.items():
        status = "OK" if ok else "FAILED"
        local_name = model_id.replace("/", "--")
        print(f"  {model_id:30s}  {status}  ({args.output_dir}/{local_name})")
        if not ok:
            all_ok = False
    print("=" * 50)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
