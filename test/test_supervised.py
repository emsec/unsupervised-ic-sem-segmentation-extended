import argparse
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import torchvision.models.segmentation as tv_seg
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).absolute().parents[1]))
from libs.supervised_dataset import segmentation_dataset, get_test_imgs
from libs.supervised_unet import unet

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset-dir', type=str, default=None, help='directory with imgs for inference')
    parser.add_argument('-out-dir', type=str, required=True, help='directory to save outputs')

    parser.add_argument('-log-dir', type=str, required=True, help='directory for all exps and models')

    parser.add_argument('-in-channel', type=int, default=3, help='the number of channels from input image')
    parser.add_argument('-nclass', type=int, default=2, help='the number of output classes')

    parser.add_argument('-seed', type=int, default=0, help='random seed')
    parser.add_argument('-gpu-ids', type=int, nargs='+', default=[0, 1, 2, 3], help="choose gpu device. e.g., 0 1 2 3")
    parser.add_argument('-batch', type=int, default=32, help='batch size')

    return parser.parse_args()


def predict(args):
    # computing environment
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get dataset for inference
    dataset_dir = Path(args.dataset_dir)
    imgs_list = get_test_imgs(dataset_dir, args.seed)
    imgs_ds = segmentation_dataset(folder_path=dataset_dir, folder_name=['image', 'label'], img_list=imgs_list, file_format=['.jpg', '.png'], transform_options=['normalize'], model_phase='test')
    imgs_loader = DataLoader(imgs_ds, batch_size=args.batch, shuffle=False)
    # get trained models (all .pth model files in model-dir will be loaded for inference)
    log_dir = Path(args.log_dir)        # directory with all experiment logs
    exp_dirs = [dir for dir in log_dir.iterdir() if dir.is_dir()]
    for exp_dir in exp_dirs:
        print(f'Fetching experiment: {exp_dir.name}')
        # get all trained model files
        model_paths = exp_dir.glob('*.pth')
        for model_path in model_paths:
            # define model and load model state dict
            print(f'\tEvaluating: {model_path.stem}')
            model_arch = model_path.stem.split('_')[2]
            if model_arch == 'unet':
                model = unet(img_channel=args.in_channel, mask_channel=args.nclass).to(device)
            elif model_arch == 'deeplabv3':
                model = tv_seg.deeplabv3_resnet50(num_classes=args.nclass).to(device)
            elif model_arch == 'fcn':
                model = tv_seg.fcn_resnet50(num_classes=args.nclass).to(device)
            else:
                ValueError('Specified model is not supported!')
            model_state = torch.load(model_path, map_location=device)
            model.load_state_dict(model_state)
            model = model.to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            # run inference
            output_dir = Path(args.out_dir) / exp_dir.name / model_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            model.eval()
            softmax_layer = nn.Softmax(dim=1)
            with torch.no_grad():
                for imgs, masks, img_names in tqdm(imgs_loader):
                    imgs = imgs.to(device)
                    score = model(imgs) if model_arch == 'unet' else model(imgs)['out']
                    score = softmax_layer(score).data.cpu().numpy()
                    score = np.argmax(score, axis=1)
                    for sample_idx in range(imgs.size(0)):
                        score_img = Image.fromarray((score[sample_idx, :, :] * 255.0 / (args.nclass - 1)).astype(np.uint8))
                        img_name = Path(img_names[sample_idx]).stem
                        score_img.save(output_dir / (img_name + '.png'))


if __name__ == '__main__':
    args = get_args()
    predict(args)
