import argparse
import pathlib
import os
import sys

from tqdm import tqdm

import kornia

import numpy as np
from PIL import Image

import torch
from torchvision.io import ImageReadMode, read_image

from skimage.filters import threshold_multiotsu
from skimage.exposure import histogram

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
from libs.morphsnakes import morphsnakes

def threshold(sem, threshold: int) -> torch.Tensor:
    thresh = torch.zeros_like(sem)
    thresh[sem > threshold] = 255
    return thresh


def median_blur(sem, kernel_size: int) -> torch.Tensor:
    return kornia.filters.median_blur(sem.unsqueeze(0), kernel_size).squeeze(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    # Thresholds and median_blur defaults as chosen for RUB dataset in ASHES Workshop paper
    parser.add_argument("-tt", "--track_threshold", type=int, default=67, help="Threshold to separate tracks from background")
    parser.add_argument("-vt", "--via_threshold", type=int, default=157, help="Threshold to separate vias from tracks")
    parser.add_argument("-rv", "--random_variance", type=int, default=25, help="Half interval around thresholds which will be used as random thresholds ")
    parser.add_argument("-m", "--median_blur", type=int, default=5, help="Perform median filtering with the corresponding filter size before thresholding")
    parser.add_argument("-c", "--channels", required=True, type=int, choices=[1, 3], help="Store either as 3 channel masks for background/track/via, or as single channel track labels without via information")
    parser.add_argument("-a", "--algorithm", required=True, choices=["global_threshold", "random_threshold", "otsu", "morphsnakes", "random_morphsnakes"], help="Algorithm to use for mask generation")
    parser.add_argument("-e", "--extension", default="jpg", choices=["jpg", "png"], help="File extension of the SEM images to search for")
    parser.add_argument("output_dir", type=pathlib.Path, help="Directory to store masks")
    parser.add_argument("input_dirs", type=pathlib.Path, nargs='+', help="Directories with SEM images to process")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all input images
    ipaths: list[pathlib.Path] = []
    for idir in args.input_dirs:
        for ipath in idir.glob(f"*.{args.extension}"):
            ipaths.append(ipath)

    assert len(ipaths) > 0, "Found no SEM images"

    morph = None
    hist = None

    if args.algorithm == "otsu":
        hist = np.zeros(256)
        for ipath in tqdm(ipaths, desc="Collecting histogram data"):
            sem = np.array(Image.open(ipath), dtype=np.uint8)
            sem_hist, _ = histogram(sem, source_range='dtype')
            hist += sem_hist

        track_thresh, via_thresh = threshold_multiotsu(hist=hist, classes=3)
        args.track_threshold = track_thresh
        args.via_threshold = via_thresh
        args.algorithm = "global_threshold"
        print(f"Otsu thresholds: {track_thresh} (tracks), {via_thresh} (vias)")

    # Perform thresholding and save
    for ipath in tqdm(ipaths):
        sem = read_image(str(ipath), mode=ImageReadMode.GRAY).to(device, torch.float)
        if args.median_blur > 0:
            sem = median_blur(sem, args.median_blur)

        if args.algorithm == "random_threshold":
            tthresh =  args.random_variance * torch.rand(1).item() + args.track_threshold
            vthresh =  args.random_variance * torch.rand(1).item() + args.via_threshold
            mtracks = threshold(sem, tthresh)
            mvias = threshold(sem, vthresh)

        elif args.algorithm == "global_threshold":
            mtracks = threshold(sem, args.track_threshold)
            mvias = threshold(sem, args.via_threshold)

        elif args.algorithm == "morphsnakes":
            if morph is None:
                morph = morphsnakes.MorphACWE(None, sem.shape[-2:])

            sem = sem.squeeze(0).cpu().numpy()
            mtracks = torch.from_numpy(255 * morph(sem, args.track_threshold, 50, smoothing=3, lambda1=1, lambda2=2)).unsqueeze(0)
            mvias = torch.from_numpy(255 * morph(sem, args.via_threshold, 50, smoothing=3, lambda1=2, lambda2=1)).unsqueeze(0)

        elif args.algorithm == "random_morphsnakes":
            if morph is None:
                morph = morphsnakes.MorphACWE(None, sem.shape[-2:])

            tthresh =  args.random_variance * torch.rand(1).item() + args.track_threshold
            vthresh =  args.random_variance * torch.rand(1).item() + args.via_threshold
            sem = sem.squeeze(0).cpu().numpy()
            mtracks = torch.from_numpy(255 * morph(sem, tthresh, 50, smoothing=3, lambda1=1, lambda2=2)).unsqueeze(0)
            mvias = torch.from_numpy(255 * morph(sem, vthresh, 50, smoothing=3, lambda1=2, lambda2=1)).unsqueeze(0)

        else:
            raise ValueError(f"Unknown algorithm {args.algorithm}")

        if args.channels ==1:
            mask = mtracks
            Image.fromarray(mask.clamp(0, 255).permute(1, 2, 0).squeeze().to("cpu", torch.bool).numpy()).save(args.output_dir / ipath.with_suffix(".png").name, bits=1, optimize=True)

        elif args.channels == 3:


            mbackground = 255 - mtracks
            mtracks = mtracks - mvias
            mask = torch.cat([mbackground, mtracks, mvias], dim=0)
            Image.fromarray(mask.clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()).save(args.output_dir / ipath.with_suffix(".png").name, optimize=True)
        else:
            raise ValueError("Unsupported number of channels")



if __name__ == "__main__":
    main()
