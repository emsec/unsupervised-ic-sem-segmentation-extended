import argparse
import pathlib
import os
import sys
import time

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


class DecoderDataset:
    def __init__(self, split_file_path, grad_dir, mask_dir, norm="no", grad_ext="png"):
        self.grad_dir = pathlib.Path(grad_dir)
        self.mask_dir = pathlib.Path(mask_dir)
        self.norm = norm
        self.grad_ext = grad_ext

        self.samples = []
        with open(split_file_path) as dir:
            for name in dir.readlines():
                self.samples.append(name.strip())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        grad = F.to_tensor(Image.open(str((self.grad_dir / name).with_suffix(f".{self.grad_ext}"))))
        mask = F.to_tensor(Image.open(str((self.mask_dir / name).with_suffix(".png"))))
        if self.norm in ["grad", "both"]:
            grad = F.normalize(grad, 0.5, 0.5)
            # norm = "sem" is only applicable to encoder training
            # Mask corresponds to Encoder Softmax output in [0,1], so can't normalize hat
        return grad, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train for")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-f", "--fast", action="store_true", help="Use torch AMP for faster training with reduced precision")
    parser.add_argument("-n", "--normalize", action="store_true", help="Shorthand for --norm both")
    parser.add_argument("--norm", default="no", choices=["no", "sem", "grad", "both"], help="Whether to normalize the SEM and gradient patches to [-1, 1]")
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help="Choose gpu devices, e.g., 0 1 2 3")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--grad_ext", default="png", choices=["png", "jpg"], help="File extensions of image gradient files")
    parser.add_argument("-o", "--out", type=pathlib.Path, required=True, help="File to save the best performing model to")
    parser.add_argument("mask_dir", type=pathlib.Path, help="Directory with pseudo mask patches to train on")
    parser.add_argument("grad_dir", type=pathlib.Path, help="Directory with gradient patches as labels")
    parser.add_argument("train_split", type=pathlib.Path, help="File listing all images in train set")
    parser.add_argument("validation_split", type=pathlib.Path, help="File listing all images in validation set")
    args = parser.parse_args()

    if args.normalize:
        args.norm = "both"

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu_ids)
    if hasattr(torch.cuda.device_count, 'cache_clear'):
        torch.cuda.device_count.cache_clear() # Fixes https://stackoverflow.com/questions/78486965/pytorch-runtimeerror-device-0-device-num-gpus-internal-assert-failed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("Training without deterministic random seed")

    train_loader = DataLoader(DecoderDataset(args.train_split, args.grad_dir, args.mask_dir, norm=args.norm, grad_ext=args.grad_ext), num_workers=8, pin_memory=True, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DecoderDataset(args.validation_split, args.grad_dir, args.mask_dir, norm=args.norm, grad_ext=args.grad_ext), num_workers=8, pin_memory=True, batch_size=args.batch_size, shuffle=False)

    decoder = UNet(3, 1, activation=nn.Tanh()).to(device)
    decoder.init_weights()

    if torch.cuda.device_count() > 1:
        decoder = torch.nn.DataParallel(decoder, device_ids=list(range(torch.cuda.device_count())))

    opt = torch.optim.Adam(list(decoder.parameters()))
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
        decoder.train()
        with TIMER:
            for grad, mask in tqdm(train_loader, desc=f"Epoch {epoch+1} train"):
                grad, mask = grad.to(device), mask.to(device)
                opt.zero_grad(set_to_none=True)

                if args.fast:
                    with amp.autocast():
                        pred = decoder(mask)
                        loss = criterion(pred, grad)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred = decoder(mask)
                    loss = criterion(pred, grad)
                    loss.backward()
                    opt.step()

        decoder.eval()
        with torch.no_grad():
            loss = 0.0
            with TIMER:
                for grad, mask in tqdm(val_loader, desc=f"Epoch {epoch+1} test"):
                    grad, mask = grad.to(device), mask.to(device)

                    if args.fast:
                        with amp.autocast():
                            pred = decoder(mask)
                            loss += criterion(pred, grad).item()
                    else:
                        pred = decoder(mask)
                        loss += criterion(pred, grad).item()

                loss /= len(val_loader)
                sched.step(loss)

            if loss < BEST_LOSS:
                BEST_LOSS = loss
                BEST_EPOCH = epoch+1
                BEST_PARAMS = decoder.module.state_dict() if torch.cuda.device_count() > 1 else decoder.state_dict()
            print(f"Validation loss: {loss} (best {BEST_LOSS} in epoch {BEST_EPOCH})")

    print(f"Decoder training took {TIMER.elapsed:.2f} seconds")

    os.makedirs(args.out.parent, exist_ok=True)
    #torch.save(BEST_PARAMS, args.out.with_suffix(f".epoch{BEST_EPOCH}.pt"))
    torch.save(BEST_PARAMS, args.out.with_suffix(".pt"))


if __name__ == "__main__":
    main()
