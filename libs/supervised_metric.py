import torch


def compute_iou(predict_score, mask):
    predict_label = predict_score > 0.5
    predict_label = predict_label[:, 1, :, :]
    intersection = torch.sum(predict_label * mask, (1, 2))
    union = torch.sum(torch.logical_or(predict_label, mask), (1, 2))
    iou = (intersection + 0.0001) / (union + 0.0001)
    iou = iou.tolist()
    return iou


def compute_pixel_acc(predict_score, mask):
    predict_label = predict_score > 0.5
    predict_label = predict_label[:, 1, :, :]
    mask = mask.squeeze(1)
    pixel_acc = (predict_label==mask).float().mean(dim=[1,2])
    pixel_acc = pixel_acc.tolist()
    return pixel_acc
