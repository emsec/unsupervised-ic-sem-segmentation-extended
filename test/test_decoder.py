import argparse
import pathlib
import os
import sys

from tqdm import tqdm

import kornia

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
from libs.unet_resize_conv import UNet


class DecoderDataset:
    def __init__(self, split_file_path, grad_dir, mask_dir, norm="no"):
        self.grad_dir = pathlib.Path(grad_dir)
        self.mask_dir = pathlib.Path(mask_dir)
        self.norm = norm

        self.samples = []
        with open(split_file_path) as dir:
            for name in dir.readlines():
                self.samples.append(name.strip())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        grad = F.to_tensor(Image.open(str((self.grad_dir / name).with_suffix(".png"))))
        mask = F.to_tensor(Image.open(str((self.mask_dir / name).with_suffix(".png"))))
        if self.norm in ["grad", "both"]:
            grad = F.normalize(grad, 0.5, 0.5)
            # norm = "sem" is only applicable to encoder training
        return grad, mask, name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--decoder", type=pathlib.Path, required=True, help="Trained decoder model parameters")
    parser.add_argument("-o", "--out_dir", type=pathlib.Path, help="Folder to save the results to")
    parser.add_argument("-n", "--normalize", action="store_true", help="Shorthand for --norm both")
    parser.add_argument("--norm", default="no", choices=["no", "sem", "grad", "both"], help="Whether to normalize the SEM and gradient patches to [-1, 1]")
    parser.add_argument("test_split", type=pathlib.Path, help="File listing all images in test set")
    parser.add_argument("mask_dir", type=pathlib.Path, help="Directory with pseudo mask patches to train on")
    parser.add_argument("grad_dir", type=pathlib.Path, help="Directory with gradient patches as labels")
    args = parser.parse_args()

    if args.normalize:
        args.norm = "both"

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    loader = DataLoader(DecoderDataset(args.test_split, args.grad_dir, args.mask_dir, norm=args.norm), shuffle=False)

    decoder = UNet(3, 1, activation=nn.Tanh()).to(device)
    decoder.load_state_dict(torch.load(args.decoder))

    criterion = torch.nn.MSELoss()

    decoder.eval()
    with torch.no_grad():
        loss = 0.0
        for grad, mask, name in tqdm(loader):
            grad, mask = grad.to(device), mask.to(device)
            pred = decoder(mask)
            loss += criterion(pred, grad).item()

            if args.out_dir:
                name = name[0]
                F.to_pil_image(pred.squeeze(0)).save((args.out_dir / name).with_suffix(".png"))

        loss /= len(loader)
        print(f"Decoder test loss (mean): {loss}")


if __name__ == "__main__":
    main()
