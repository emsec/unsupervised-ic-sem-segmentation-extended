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


class EncoderDataset:
    def __init__(self, split_file_path, sem_dir, grad_dir, norm="no", grad_ext="png"):
        self.sem_dir = pathlib.Path(sem_dir)
        self.grad_dir = pathlib.Path(grad_dir)
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
        sem = F.to_tensor(Image.open(str((self.sem_dir / name).with_suffix(".jpg"))))
        grad = F.to_tensor(Image.open(str((self.grad_dir / name).with_suffix(f".{self.grad_ext}"))))
        if self.norm in ["sem", "both"]:
            sem = F.normalize(sem, 0.5, 0.5)
        if self.norm in ["grad", "both"]:
            grad = F.normalize(grad, 0.5, 0.5)
        return sem, grad, name


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-d", "--decoder", type=pathlib.Path, required=True, help="Trained decoder model parameters")
    parser.add_argument("-e", "--encoder", type=pathlib.Path, required=True, help="Trained encoder model parameters")
    parser.add_argument("-o", "--out_dir", type=pathlib.Path, help="Folder to save the results to")
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help="Choose gpu devices, e.g., 0 1 2 3")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("-g", "--save_grads", action="store_true", help="Save decoder SEM gradient outputs")
    parser.add_argument("-m", "--save_masks", action="store_true", help="Save encoder mask outputs")
    parser.add_argument("-n", "--normalize", action="store_true", help="Shorthand for --norm both")
    parser.add_argument("--norm", default="no", choices=["no", "sem", "grad", "both"], help="Whether to normalize the SEM and gradient patches to [-1, 1]")
    parser.add_argument("--grad_ext", default="png", choices=["png", "jpg"], help="File extensions of image gradient files")
    parser.add_argument("test_split", type=pathlib.Path, help="File listing all images in test set")
    parser.add_argument("sem_dir", type=pathlib.Path, help="Directory with input SEM images")
    parser.add_argument("grad_dir", type=pathlib.Path, help="Directory with gradient patches as labels")
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
        print("Testing without deterministic random seed")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    loader = DataLoader(EncoderDataset(args.test_split, args.sem_dir, args.grad_dir, norm=args.norm, grad_ext=args.grad_ext), num_workers=8, pin_memory=True, batch_size=args.batch_size, shuffle=False)

    decoder = UNet(3, 1, activation=nn.Tanh()).to(device)
    decoder.load_state_dict(torch.load(args.decoder))

    # TODO Encoder is right now trained without overlap between patches (might yield worse performance on border)
    encoder = UNet(1, 3, activation=nn.Softmax(dim=1)).to(device)
    encoder.load_state_dict(torch.load(args.encoder))

    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder, device_ids=list(range(torch.cuda.device_count())))
        decoder = torch.nn.DataParallel(decoder, device_ids=list(range(torch.cuda.device_count())))

    criterion = torch.nn.MSELoss()

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        loss = 0.0
        for sem, grad, name in tqdm(loader):
            sem, grad = sem.to(device), grad.to(device)
            pred_mask = encoder(sem)
            pred_grad = decoder(pred_mask)
            loss += criterion(pred_grad, grad).item() # Run without channel exclusivity loss here?

            if args.norm in ["grad", "both"]:
                pred_grad = pred_grad.mul(0.5).add(0.5)

            if args.out_dir:
                for i in range(sem.shape[0]):
                    nm = name[i]
                    if args.save_masks:
                        F.to_pil_image(pred_mask[i].squeeze(0)).save((args.out_dir / f"{nm}_mask").with_suffix(".png"))
                    if args.save_grads:
                        F.to_pil_image(pred_grad[i].squeeze(0)).save((args.out_dir / f"{nm}_grad").with_suffix(".png"))

    loss /= len(loader)
    print(f"Encoder test loss (mean): {loss}")


if __name__ == "__main__":
    main()
