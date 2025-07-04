import argparse
import pathlib
import os
import sys
import time
import math

from tqdm import tqdm

import kornia

import numpy as np
from PIL import Image

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F

sys.path.append(str(pathlib.Path(__file__).absolute().parents[1]))
from libs.unet_resize_conv import UNet


class Timer:
    def __init__(self):
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *_):
        end = time.time()
        self.elapsed += end - self.start


class PreEncoderDataset:
    def __init__(self, split_file_path, sem_dir, mask_dir, norm="no", dataset_fraction=1.0):
        self.sem_dir = pathlib.Path(sem_dir)
        self.mask_dir = pathlib.Path(mask_dir)
        self.norm = norm
        dataset_fraction = min(1.0, max(0.0, dataset_fraction))

        self.samples = []
        with open(split_file_path) as dir:
            lines = dir.readlines()
            np.random.shuffle(lines)
            lines = lines[:math.ceil(dataset_fraction * len(lines))]

            for name in lines:
                self.samples.append(name.strip())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        sem = F.to_tensor(Image.open(str((self.sem_dir / name).with_suffix(".jpg"))))
        mask = F.to_tensor(Image.open(str((self.mask_dir / name).with_suffix(".png"))))
        if self.norm in ["sem", "both"]:
            sem = F.normalize(sem, 0.5, 0.5)
        return sem, mask


def regularization(masks, weight):
    """
    Force per-pixel exclusivity of b, t, v labels.
    """
    if weight != 0.0:
        b, t, v = masks.split(1, dim=1)
        return weight * (b*t + b*v + t*v).mean()
    else:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("-fr", "--fraction", type=float, default=1.0, help="Train only on this fraction of the train set")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-f", "--fast", action="store_true", help="Use torch AMP for faster training with reduced precision")
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help="Choose gpu devices, e.g., 0 1 2 3")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("-o", "--out", type=pathlib.Path, required=True, help="File to save the best performing encoder model to")
    parser.add_argument("-c", "--chan_excl_loss", type=float, default=0.1, help="Weight for the channel exclusivity loss regularization factor")
    parser.add_argument("-n", "--normalize", action="store_true", help="Shorthand for --norm both")
    parser.add_argument("--norm", default="no", choices=["no", "sem", "grad", "both"], help="Whether to normalize the SEM and gradient patches to [-1, 1]")
    parser.add_argument("sem_dir", type=pathlib.Path, help="Directory with input SEM image patches")
    parser.add_argument("mask_dir", type=pathlib.Path, help="Directory with thresholded pseudo mask patches")
    parser.add_argument("train_split", type=pathlib.Path, help="File listing all images in train set")
    parser.add_argument("validation_split", type=pathlib.Path, help="File listing all images in validation set")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu_ids)
    if hasattr(torch.cuda.device_count, 'cache_clear'):
        torch.cuda.device_count.cache_clear() # Fixes https://stackoverflow.com/questions/78486965/pytorch-runtimeerror-device-0-device-num-gpus-internal-assert-failed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.normalize:
        args.norm = "both"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("Training without deterministic random seed")

    train_loader = DataLoader(PreEncoderDataset(args.train_split, args.sem_dir, args.mask_dir, norm=args.norm, dataset_fraction=args.fraction), num_workers=8, pin_memory=True, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PreEncoderDataset(args.validation_split, args.sem_dir, args.mask_dir, norm=args.norm), num_workers=8, pin_memory=True, batch_size=args.batch_size, shuffle=False)

    # TODO Encoder is right now trained without overlap between patches
    # and outputs masks with the same size as the input patches (might yield worse performance)
    encoder = UNet(1, 3, activation=nn.Softmax(dim=1))
    encoder.init_weights()

    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder, device_ids=list(range(torch.cuda.device_count())))

    encoder = encoder.to(device)

    opt = torch.optim.Adam(list(encoder.parameters()))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    criterion = torch.nn.MSELoss()

    if args.fast:
        torch.set_float32_matmul_precision('high')
        scaler = amp.GradScaler()

    BEST_LOSS = float("inf")
    BEST_EPOCH = 0
    BEST_PARAMS = None

    TIMER = Timer()

    for epoch in range(args.epochs):
        encoder.train()
        with TIMER:
            for sem, mask in tqdm(train_loader, desc=f"Epoch {epoch+1} pretrain"):
                sem, mask = sem.to(device), mask.to(device)
                opt.zero_grad(set_to_none=True)

                if args.fast:
                    with amp.autocast():
                        pred_mask = encoder(sem)
                        loss = criterion(pred_mask, mask) + regularization(pred_mask, args.chan_excl_loss)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred_mask = encoder(sem)
                    loss = criterion(pred_mask, mask) + regularization(pred_mask, args.chan_excl_loss)
                    loss.backward()
                    opt.step()

        encoder.eval()
        with torch.no_grad():
            loss = 0.0
            with TIMER:
                for sem, mask in tqdm(val_loader, desc=f"Epoch {epoch+1} pretest"):
                    sem, mask = sem.to(device), mask.to(device)

                    if args.fast:
                        with amp.autocast():
                            pred_mask = encoder(sem)
                            loss += criterion(pred_mask, mask).item() # Run without channel exclusivity loss here?
                    else:
                        pred_mask = encoder(sem)
                        loss += criterion(pred_mask, mask).item() # Run without channel exclusivity loss here?

                loss /= len(val_loader)
                sched.step(loss)

            if loss < BEST_LOSS:
                BEST_LOSS = loss
                BEST_EPOCH = epoch+1
                #BEST_PARAMS = encoder.module.state_dict() if torch.cuda.device_count() > 1 else encoder.state_dict()
            print(f"Validation loss: {loss} (best {BEST_LOSS} in epoch {BEST_EPOCH})")

    print(f"Trained on {args.epochs * len(train_loader.dataset)} patches total")
    print(f"Encoder pretraining took {TIMER.elapsed:.2f} seconds")

    os.makedirs(args.out.parent, exist_ok=True)
    #torch.save(BEST_PARAMS, args.out.with_suffix(".pt"))
    torch.save(encoder.module.state_dict() if torch.cuda.device_count() > 1 else encoder.state_dict(), args.out.with_suffix(".pt"))
    # For pretraining, save the final params, not the best ones.

if __name__ == "__main__":
    main()
