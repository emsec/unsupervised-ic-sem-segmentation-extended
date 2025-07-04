import argparse
import pathlib
import os

from tqdm import tqdm

import kornia

import numpy as np
from PIL import Image

import torch
from torchvision.io import ImageReadMode, read_image

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-pw", "--patch_width", type=int, default=512)
    parser.add_argument("-ph", "--patch_height", type=int, default=512)
    parser.add_argument("-o", "--out_dir", type=pathlib.Path, help="Directory to store patches in image and label subdirs")
    parser.add_argument("sem_dir", type=pathlib.Path, help="Directory with SEM images to process")
    parser.add_argument("label_dir", type=pathlib.Path, help="Directory with associated labels to process")

    args = parser.parse_args()
    os.makedirs(args.out_dir / "image", exist_ok=True)
    os.makedirs(args.out_dir / "label", exist_ok=True)

    # Collect all input images
    ipaths: list[pathlib.Path] = []
    for ipath in args.sem_dir.glob(f"*.png"):
        ipaths.append(ipath)

    assert len(ipaths) > 0, "Found no SEM images"

    # Split and save patches
    patch_nr = 0
    for ipath in tqdm(ipaths):
        lpath = args.label_dir / ipath.name.replace("sem", "label")
        assert lpath.exists(), f"Label image {lpath} not found"

        sem = read_image(str(ipath), mode=ImageReadMode.GRAY)
        label = read_image(str(lpath), mode=ImageReadMode.GRAY)
        sem_patches = kornia.contrib.extract_tensor_patches(sem.unsqueeze(0), (args.patch_height, args.patch_width), (args.patch_height, args.patch_width)).squeeze(0)
        label_patches = kornia.contrib.extract_tensor_patches(label.unsqueeze(0), (args.patch_height, args.patch_width), (args.patch_height, args.patch_width)).squeeze(0)

        for patch, label in zip(sem_patches.split(1), label_patches.split(1)):
            Image.fromarray(patch.squeeze().to("cpu", torch.uint8).numpy()).save(args.out_dir / "image" / f"img_{patch_nr:03d}.jpg")
            Image.fromarray(label.squeeze().to("cpu", torch.bool).numpy()).save(args.out_dir / "label" / f"img_{patch_nr:03d}.png", bits=1, optimize=True)
            patch_nr += 1


if __name__ == "__main__":
    main()
