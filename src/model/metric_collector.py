import torch
import os
from einops import rearrange
import numpy as np


def compute_scale_and_shift(prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

def compute_depth_metrics(gt, pred, mask=None, median_align=False):
    """Computation of metrics between predicted and ground truth depths
    """

    if mask is None:
        mask = gt > 0

    gt = gt.squeeze(1)
    pred = pred.squeeze(1)
    mask = mask.squeeze(1)

    scale, shift = compute_scale_and_shift(pred, gt, mask)
    scale = scale.unsqueeze(1).unsqueeze(2)
    shift = shift.unsqueeze(1).unsqueeze(2)

    aligned_pred = pred * scale + shift
    pred = aligned_pred

    gt = gt[mask]
    pred = pred[mask]

    if median_align:
        pred *= torch.median(gt) / torch.median(pred)

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt) - torch.log10(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_ = torch.mean(torch.abs(gt - pred))

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    log10 = torch.mean(torch.abs(torch.log10(pred/gt)))

    return abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DepthMetricCollector:
    def __init__(self, median_align=False):
        self.median_align = median_align
        # 用于收集所有batch的结果
        self.all_gt_depths = []
        self.all_pred_depths = []
        self.all_masks = []
        
        # 初始化评估指标
        self.metrics = {}
        self.metrics["err/abs_"] = AverageMeter()
        self.metrics["err/abs_rel"] = AverageMeter()
        self.metrics["err/sq_rel"] = AverageMeter()
        self.metrics["err/rms"] = AverageMeter()
        self.metrics["err/log_rms"] = AverageMeter()
        self.metrics["err/log10"] = AverageMeter()
        self.metrics["acc/a1"] = AverageMeter()
        self.metrics["acc/a2"] = AverageMeter()
        self.metrics["acc/a3"] = AverageMeter()

    def reset(self):
        self.all_gt_depths = []
        self.all_pred_depths = []
        self.all_masks = []
        for metric in self.metrics.values():
            metric.reset()

    def update(self, gt_depth, pred_depth, mask):
        # 收集每个batch的结果
        self.all_gt_depths.append(gt_depth.detach().cpu())
        self.all_pred_depths.append(pred_depth.detach().cpu())
        self.all_masks.append(mask.detach().cpu())

    def compute_metrics(self):
        # 合并所有batch的结果
        gt_depth = torch.cat(self.all_gt_depths, dim=0)
        pred_depth = torch.cat(self.all_pred_depths, dim=0)
        mask = torch.cat(self.all_masks, dim=0)
        
        N = gt_depth.shape[0]
        
        # 计算指标
        abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
            compute_depth_metrics(gt_depth, pred_depth, mask, self.median_align)

        # 更新指标
        self.metrics["err/abs_"].update(abs_, N)
        self.metrics["err/abs_rel"].update(abs_rel, N)
        self.metrics["err/sq_rel"].update(sq_rel, N)
        self.metrics["err/rms"].update(rms, N)
        self.metrics["err/log_rms"].update(rms_log, N)
        self.metrics["err/log10"].update(log10, N)
        self.metrics["acc/a1"].update(a1, N)
        self.metrics["acc/a2"].update(a2, N)
        self.metrics["acc/a3"].update(a3, N)

    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["err/abs_"].avg)
        avg_metrics.append(self.metrics["err/abs_rel"].avg)
        avg_metrics.append(self.metrics["err/sq_rel"].avg)
        avg_metrics.append(self.metrics["err/rms"].avg)
        avg_metrics.append(self.metrics["err/log_rms"].avg)
        avg_metrics.append(self.metrics["err/log10"].avg)
        avg_metrics.append(self.metrics["acc/a1"].avg)
        avg_metrics.append(self.metrics["acc/a2"].avg)
        avg_metrics.append(self.metrics["acc/a3"].avg)

        print("\n  "+ ("{:>9} | " * 9).format("abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
        print(("&  {: 8.5f} " * 9).format(*avg_metrics))

        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'w') as f:
                print("\n  " + ("{:>9} | " * 9).format("abs_", "abs_rel", "sq_rel", "rms", "rms_log",
                                                      "log10", "a1", "a2", "a3"), file=f)
                print(("&  {: 8.5f} " * 9).format(*avg_metrics), file=f)