import argparse
from pathlib import Path
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset-dir', type=str, required=True, help='the dataset directory')
    parser.add_argument('-split', type=float, nargs='+', required=True, help='ratio of samples for train/val/test split, e.g., 0.7 0.1 0.2')
    parser.add_argument('-seed', type=int, default=0, help='seed value for random split of train/val, seed for test split is default to 0')
    return parser.parse_args()

def splitter(args):
    dataset_root = Path(args.dataset_dir)
    image_root = dataset_root / 'image'
    image_list = image_root.glob('*')
    image_list = list(image_name.stem for image_name in image_list)

    # if there is no test split, do split for test set with random seed 0
    test_txt_filepath = Path(dataset_root / 'test_image_names_seed0.txt')
    if not test_txt_filepath.exists():
        random.seed(0)
        test_images = random.sample(image_list, int(args.split[2] * len(image_list)))
        with open(test_txt_filepath, 'w') as f:
            for img in test_images:
                f.write(f"{img}\n")
    else:
        with open(test_txt_filepath, 'r') as f:
            test_images = f.read().splitlines()

    # do split for train/val based on input rand seed
    remain_images = [element for _, element in enumerate(image_list) if element not in test_images]
    random.seed(args.seed)
    val_images = random.sample(remain_images, int(args.split[1] * len(image_list)))
    # remain as train images
    train_images = [element for _, element in enumerate(remain_images) if element not in val_images]
    # write splits to txt file
    for split in ['train', 'val']:
        txt_filepath = dataset_root / f'{split}_image_names_seed{args.seed}.txt'
        with open(txt_filepath, 'w') as f:
            images = locals()[f'{split}_images']
            for img in images:
                f.write(f"{img}\n")

if __name__ == '__main__':
    args = get_args()
    splitter(args)
