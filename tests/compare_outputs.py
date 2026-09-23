"""Opt-in pixel comparison of locally generated benchmark PNGs, not a quality score."""
# ruff: noqa: T201
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    reference = np.asarray(Image.open(args.reference).convert("RGB"), dtype=np.float32) / 255
    for image in args.images:
        pixels = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255
        mse = float(np.mean((pixels - reference) ** 2))
        print(json.dumps({"image": image.name, "mae": float(np.mean(np.abs(pixels - reference))),
                          "psnr_db": float(-10 * np.log10(mse)) if mse else None,
                          "ssim": float(structural_similarity(reference, pixels, channel_axis=2, data_range=1))}))
