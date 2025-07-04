import argparse
import pathlib
import os
import sys

from tqdm import tqdm

from PIL import Image
import numpy as np
import shapely

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
from libs import esd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=pathlib.Path, required=True, help="Output directory for binary track labels")
    parser.add_argument("svg_dir", type=pathlib.Path, help="Directory with SVG label files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    inputs = list(args.svg_dir.glob("*.svg"))
    for svg_path in tqdm(inputs):
        out_path = (args.out_dir / svg_path.stem).with_suffix(".png")

        label = esd.Label(svg_path)
        img = np.zeros((label.height, label.width), dtype=np.uint8)
        for track in label.tracks:
            esd._fill_poly(img, track, (255, 255, 255))
        Image.fromarray(img.astype(bool)).save(out_path, bits=1, optimize=True)


if __name__ == "__main__":
    main()
