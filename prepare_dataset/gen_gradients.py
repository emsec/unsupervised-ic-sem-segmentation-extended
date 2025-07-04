import argparse
import pathlib
import os

from tqdm import tqdm

from PIL import Image

import numpy as np
import torch
from torchvision.transforms import functional as F
import kornia

def median_blur(sem, kernel_size: int) -> torch.Tensor:
    return kornia.filters.median_blur(sem.unsqueeze(0), kernel_size).squeeze(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--median_blur", type=int, default=5, help="Perform median filtering with the corresponding filter size before computing the gradient")
    parser.add_argument("-e", "--extension", default="jpg", choices=["jpg", "png"], help="File extension of the SEM images to search for")
    parser.add_argument("-l", "--lambda", type=float, default=5.0, help="Weight for the SEM image gradient")
    parser.add_argument("-k", "--kernel_size", type=int, default=5, help="Size of the kernel used to compute the morphological gradient")
    parser.add_argument("output_dir", type=pathlib.Path, help="Directory to store gradients")
    parser.add_argument("input_dirs", type=pathlib.Path, nargs='+', help="Directories with SEM images to process")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all input images
    ipaths: list[pathlib.Path] = []
    for idir in args.input_dirs:
        for ipath in idir.glob(f"*.{args.extension}"):
            ipaths.append(ipath)

    assert len(ipaths) > 0, "Found no SEM images"

    kernel = torch.ones(args.kernel_size, args.kernel_size, device=device)
    for ipath in tqdm(ipaths):
        sem = F.to_tensor(Image.open(str(ipath))).to(device)

        if args.median_blur > 0:
            sem = median_blur(sem, args.median_blur)

        grad = getattr(args, "lambda") * kornia.morphology.gradient(sem.unsqueeze(0), kernel, border_type="reflect", engine='convolution').squeeze(0)
        grad = grad.clamp(0, 1)

        F.to_pil_image(grad).save(args.output_dir / ipath.with_suffix(".png").name)


if __name__ == "__main__":
    main()
