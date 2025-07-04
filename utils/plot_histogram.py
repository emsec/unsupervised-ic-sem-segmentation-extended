import argparse
import pathlib
import os

from tqdm import tqdm

import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image

import kornia

import matplotlib.pyplot as plt


def median_blur(sem, kernel_size: int) -> torch.Tensor:
    return kornia.filters.median_blur(sem.unsqueeze(0), kernel_size).squeeze(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--median_blur", type=int, default=5, help="Perform median filtering with the corresponding filter size before computing the histogram")
    parser.add_argument("-e", "--extension", default="png", choices=["jpg", "png"], help="File extension of the SEM images to search for")
    parser.add_argument("input_dirs", type=pathlib.Path, nargs='+', help="Directories with SEM images to process")
    args = parser.parse_args()

    # Collect all input images
    ipaths: list[pathlib.Path] = []
    for idir in args.input_dirs:
        for ipath in idir.glob(f"*.{args.extension}"):
            ipaths.append(ipath)

    assert len(ipaths) > 0, "Found no SEM images"

    # Collect histogram data
    hist = np.zeros(256)
    for ipath in tqdm(ipaths):
        sem = read_image(str(ipath), mode=ImageReadMode.GRAY).to(device, torch.float)
        if args.median_blur > 0:
            sem = median_blur(sem, args.median_blur)

        sem_hist, bin_edges = np.histogram(sem.cpu().numpy(), bins=256, range=[0, 255])
        hist += sem_hist

    # Plot histogram
    plt.bar(bin_edges[:-1], hist, color="gray", width=1)
    plt.title("SEM Dataset Histogram")
    plt.show()


if __name__ == "__main__":
    main()
