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

def eval_px(cdir, masks, args):
    mean_iou = 0
    mean_acc = 0
    num_patches = 0
    for mpath in tqdm(masks, desc=cdir.name):
        lpath = (args.label_dir / mpath.name.replace('_mask', ''))
        assert lpath.exists(), f"No label found for {mpath.name}"

        label_img = np.array(Image.open(lpath), dtype=np.uint8)

        track_mask = np.array(Image.open(mpath), dtype=np.uint8)

        if track_mask.shape[-1] == 3:
            mask = np.zeros(track_mask.shape[:2], dtype=np.uint8)
            mask[np.argmax(track_mask, axis=2) > 0] = 255
            track_mask = mask

        h, w = track_mask.shape[:2]
        tp_px = np.logical_and(track_mask, label_img).sum()
        union_px = np.logical_or(track_mask, label_img).sum()
        total_px = h * w
        tn_px = total_px - union_px
        mean_iou += tp_px / union_px if union_px > 0 else 0
        mean_acc += (tp_px + tn_px) / total_px if total_px > 0 else 0
        num_patches += 1

    mean_iou /= num_patches
    mean_acc /= num_patches
    with open(args.data_dir / "px_metrics.txt", "a+") as f:
        f.write(f"{cdir.name}: mIoU={mean_iou:.6f}, mPxAcc={mean_acc:.6f}\n")
    print(f"{cdir.name}: mIoU={mean_iou:.6f}, mPxAcc={mean_acc:.6f}\n")


def eval_esd(cdir, masks, args):
    os.makedirs(cdir / "ESD", exist_ok=True)
    stats = esd.ESDResult.Stats()
    for mpath in tqdm(masks, desc=cdir.name):
        lpath = (args.label_dir / mpath.name.replace('_mask', ''))
        assert lpath.exists(), f"No label found for {mpath.name}"

        label = esd.Label(lpath, border=args.border)

        track_mask = np.array(Image.open(mpath), dtype=np.uint8)

        if track_mask.shape[-1] == 3:
            mask = np.zeros(track_mask.shape[:2], dtype=np.uint8)
            mask[np.argmax(track_mask, axis=2) > 0] = 255
            track_mask = mask

        tracks = esd.extract_polygons(track_mask)

        if args.filter > 0:
            tracks = [t for t in tracks if shapely.area(t) >= args.filter]

        result = esd.ESDResult(tracks, label)
        stats += result.stats

        if args.draw == "always" or (args.draw == "errors" and result.stats.count_errors() > 0):
            img = np.zeros((*track_mask.shape[:2], 3), dtype=np.uint8)
            result.draw(img)
            Image.fromarray(img).save(cdir / "ESD" / f"esd_{mpath.stem}.png")

    with open(cdir / "ESD" / "esd.txt", "a+") as f:
        f.write(str(stats) + "\n")
    print(stats)

def traverse_dir(cdir, test_split, args):
    masks = []

    for d in cdir.iterdir():
        if d.is_dir():
            traverse_dir(d, test_split, args)
        elif d.stem.replace('_mask', '') in test_split:
            masks.append(d)

    if len(masks) > 0:
        if args.esd:
            eval_esd(cdir, masks, args)
        if args.px_metrics:
            eval_px(cdir, masks, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filter", type=int, default=35, help="Filter polygons with an area below this threshold")
    parser.add_argument("-b", "--border", type=int, default=35, help="Ignore pixels within this range of the image borders")
    parser.add_argument("-p", "--px_metrics", action="store_true", help="Compute mean intersection over union and pixel accuracy")
    parser.add_argument("-e", "--esd", action="store_true", help="Compute ESD errors")
    parser.add_argument("-d", "--draw", default="errors", choices=["never", "errors", "always"], help="Save ESD visualizations always, on error or never")
    parser.add_argument("data_dir", type=pathlib.Path, help="Directory tree to search for results")
    parser.add_argument("label_dir", type=pathlib.Path, help="Directory with ground-truth labels")
    parser.add_argument("test_split_file", type=pathlib.Path, help="File listing all images in test split")

    args = parser.parse_args()

    if not args.px_metrics and not args.esd:
        print("One of --px_metric and --esd flags is required")
        exit(-1)

    args.border = max(args.border, 0)

    test_split = []
    with open(args.test_split_file) as dir:
        for name in dir.readlines():
            test_split.append(name.strip())

    traverse_dir(args.data_dir, test_split, args)


if __name__ == "__main__":
    main()
