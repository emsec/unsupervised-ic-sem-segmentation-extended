import argparse
import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.models.segmentation as tv_seg
import numpy as np
import time
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parents[1]))
from libs.supervised_dataset import get_train_val_imgs, segmentation_dataset
from libs.supervised_unet import unet
from libs.supervised_metric import compute_iou, compute_pixel_acc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset-dir', type=str, required=True, help='directory with the image dataset')
    parser.add_argument('-model', type=str, required=True, help='segmentation model to use')
    parser.add_argument('-expname', type=str, required=True, help='name for current experiment')

    parser.add_argument('-in-channel', type=int, default=3, help='the number of channels from input image')
    parser.add_argument('-nclass', type=int, default=2, help='the number of output classes')

    parser.add_argument('-epoch', type=int, default=100, help='maximum number of training epochs')
    parser.add_argument('-batch', type=int, default=32, help='batch size')
    parser.add_argument('-init-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-lr-patience', type=int, default=10, help='epoch patience for learning rate decay')
    parser.add_argument('-lr-decay', type=float, default=0.1, help='decay factor for learning rate')

    parser.add_argument('-seed', type=int, default=0, help='random seed')
    parser.add_argument('-gpu-ids', type=int, nargs='+', default=[0, 1, 2, 3], help="choose gpu device. e.g., 0 1 2 3")
    return parser.parse_args()


def train_val(args):
    # computing environment
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # dataset
    dataset_dir = Path(args.dataset_dir)
    train_imgs, val_imgs = get_train_val_imgs(dataset_dir, args.seed)
    train_ds = segmentation_dataset(folder_path=dataset_dir, folder_name=['image', 'label'], img_list=train_imgs, file_format=['.jpg', '.png'], transform_options=['hflip', 'vflip', 'normalize', 'rotate'], model_phase='train', seed=args.seed)
    val_ds = segmentation_dataset(folder_path=dataset_dir, folder_name=['image', 'label'], img_list=val_imgs, file_format=['.jpg', '.png'], transform_options=['normalize'], model_phase='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    # model
    if args.model == 'unet':
        model = unet(img_channel=args.in_channel, mask_channel=args.nclass).to(device)
    elif args.model == 'deeplabv3':
        model = tv_seg.deeplabv3_resnet50(num_classes=args.nclass).to(device)
    elif args.model == 'fcn':
        model = tv_seg.fcn_resnet50(num_classes=args.nclass).to(device)
    else:
        ValueError('Specified model is not supported!')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=args.lr_patience)
    # train and val with iter
    checkpoint_path = Path.cwd() / f'{dataset_dir.name}_explogs' / args.expname
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    train_total_time = 0
    val_total_time = 0
    val_best_iou = 0
    softmax_layer = torch.nn.Softmax(dim=1)
    with open(checkpoint_path / "train_val_log.txt", 'w') as log_file:
        for epoch_idx in range(args.epoch):
            # train
            model.train()
            train_start = time.time()
            for img, mask in tqdm(train_loader, desc=f"Epoch {epoch_idx+1} train"):
                img, mask = img.to(device), mask.to(device)
                optimizer.zero_grad()
                img_predict = model(img) if args.model == 'unet' else model(img)['out']
                img_loss = criterion(img_predict, mask)
                img_loss.backward()
                optimizer.step()
            train_epoch_time = time.time() - train_start
            # val
            model.eval()
            val_start = time.time()
            val_iou_list = []
            with torch.no_grad():
                for img, mask in tqdm(val_loader, desc=f"Epoch {epoch_idx+1} test"):
                    img, mask = img.to(device), mask.to(device)
                    img_predict = model(img) if args.model == 'unet' else model(img)['out']
                    img_predict = softmax_layer(img_predict)
                    val_iou_list = val_iou_list + compute_iou(img_predict, mask)
            val_mean_iou = np.mean(np.asarray(val_iou_list))
            scheduler.step(val_mean_iou)
            val_epoch_time = time.time() - val_start
            # checkpoint for best model state in terms of validation iou
            if val_mean_iou > val_best_iou:
                val_best_iou = val_mean_iou
                best_epoch = epoch_idx + 1
                best_state = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
            # log of per-epoch stats
            train_total_time += train_epoch_time
            val_total_time += val_epoch_time
            print('Epoch {:03d}/{:03d} with validation mean IoU: {:.6f}, total training time: {:.6f} seconds, total validation time: {:.6f} seconds'.format(epoch_idx + 1, args.epoch, val_mean_iou, train_total_time, val_total_time))
            log_file.write('Epoch {:03d}/{:03d} with validation mean IoU: {:.6f}, total training time: {:.6f} seconds, total validation time: {:.6f} seconds\n'.format(epoch_idx + 1, args.epoch, val_mean_iou, train_total_time, val_total_time))
        # final log
        final_state = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        print('Epoch {:03d}/{:03d} with best validation mean IoU: {:.6f}'.format(best_epoch, args.epoch, val_best_iou))
        log_file.write('Epoch {:03d}/{:03d} with best validation mean IoU: {:.6f}'.format(best_epoch, args.epoch, val_best_iou))
    torch.save(best_state, checkpoint_path / '{}_epoch{:03d}_{}.pth'.format(args.expname, best_epoch, args.model))
    torch.save(final_state, checkpoint_path / '{}_epoch{:03d}_{}.pth'.format(args.expname, args.epoch, args.model))
    print(f'Best model state and final model state have been saved to {checkpoint_path}')


if __name__ == '__main__':
    args = get_args()
    train_val(args)
