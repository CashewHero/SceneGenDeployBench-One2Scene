from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(10)
torch.cuda.manual_seed(10)

class L1NormLoss(nn.Module):
    def __init__(self):
        super(L1NormLoss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = (target > 0).detach()

        if mask is not None:
            valid_mask *= mask.detach()

        _min, _max = torch.quantile(target[mask].cpu().detach(), torch.tensor([0.02, 1 - 0.02]),)
        gt_depth_norm = (target - _min) / (_max - _min)
        gt_depth_norm = torch.clip(gt_depth_norm, 0.01, 1.0)

        diff = gt_depth_norm - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss
    
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = (target > 0).detach()

        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss * self.loss_weight

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = (diff**2).mean()
        return loss


class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss
    
class Silog_Loss(nn.Module):
    def __init__(self, variance_focus=0.85, loss_weight=1.0):
        super(Silog_Loss, self).__init__()
        self.variance_focus = variance_focus
        self.loss_weight = loss_weight

    def forward(self, target, pred, mask=None):
        d = torch.log(pred[mask]) - torch.log(target[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0 * self.loss_weight
    
class RMSELog(nn.Module):
    def __init__(self):
        super(RMSELog, self).__init__()

    def forward(self, target, pred, mask=None):
        #assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()
        target = target[valid_mask]
        pred = pred[valid_mask]
        log_error = torch.abs(torch.log(target / (pred + 1e-12)))
        loss = torch.sqrt(torch.mean(log_error**2))
        return loss

class SSILoss(nn.Module):
    """
    Scale shift invariant MAE loss.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1, data_type=['sfm', 'stereo', 'lidar'], **kwargs):
        super(SSILoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    
    def ssi_mae(self, target, prediction, mask):
        valid_pixes = torch.sum(mask) + self.eps

        gt_median = torch.median(target) if target.numel() else 0
        gt_s = torch.abs(target - gt_median).sum() / valid_pixes
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = torch.median(prediction) if prediction.numel() else 0
        pred_s = torch.abs(prediction - pred_median).sum() / valid_pixes
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)
        
        ssi_mae_sum = torch.sum(torch.abs(gt_trans - pred_trans))
        return ssi_mae_sum, valid_pixes

    def forward(self, target, prediction, mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = prediction.shape
        loss = 0
        valid_pix = 0
        for i in range(B):
            mask_i = mask[i, ...]
            gt_depth_i = target[i, ...][mask_i]
            pred_depth_i = prediction[i, ...][mask_i]
            ssi_sum, valid_pix_i = self.ssi_mae(pred_depth_i, gt_depth_i, mask_i) 
            loss += ssi_sum
            valid_pix += valid_pix_i
        loss /= (valid_pix + self.eps)
        return loss * self.loss_weight
    
def gradient_log_loss(log_prediction_d, log_gt, mask):
    log_d_diff = log_prediction_d - log_gt

    v_gradient = torch.abs(log_d_diff[:, :, :-2, :] - log_d_diff[:, :, 2:, :])
    v_mask = torch.mul(mask[:, :, :-2, :], mask[:, :, 2:, :])
    v_gradient = torch.mul(v_gradient, v_mask)

    h_gradient = torch.abs(log_d_diff[:, :, :, :-2] - log_d_diff[:, :, :, 2:])
    h_mask = torch.mul(mask[:, :, :, :-2], mask[:, :, :, 2:])
    h_gradient = torch.mul(h_gradient, h_mask)

    EPSILON = 1e-6
    N = torch.sum(h_mask) + torch.sum(v_mask) + EPSILON

    gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
    gradient_loss = gradient_loss / N

    return gradient_loss
    
class GradientLoss_Li(nn.Module):
    def __init__(self, scale_num=4, loss_weight=1, data_type = ['lidar', 'stereo'], **kwargs):
        super(GradientLoss_Li, self).__init__()
        self.__scales = scale_num
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction, target, mask, **kwargs):
        total = 0
        target_trans = target + (~mask) * 100
        pred_log = torch.log(prediction)
        gt_log = torch.log(target_trans)
        for scale in range(self.__scales):
            step = pow(2, scale)
            
            total += gradient_log_loss(pred_log[:, ::step, ::step], gt_log[:, ::step, ::step], mask[:, ::step, ::step])
        loss = total / self.__scales
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            return 0 * torch.sum(prediction)
            # raise RuntimeError(f'VNL error, {loss}')
        return loss * self.loss_weight
    
class EPNLoss(nn.Module):
    """
    Hieratical depth spatial normalization loss for Gaussian sampling.
    Replace the original grid masks with the random created masks.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1.0, random_num=32, batch_limit=8, lower_bound=0.125, upper_bound=0.5, **kwargs):
        super(EPNLoss, self).__init__()
        self.loss_weight = loss_weight
        self.random_num = random_num
        self.batch_limit = batch_limit
        self.eps = 1e-6
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_random_masks_for_batch(self, image_size: list)-> torch.Tensor:
        height, width = image_size
        crop_h_min = int(self.lower_bound * height)
        crop_h_max = int(self.upper_bound * height)
        crop_w_min = int(self.lower_bound * height)
        crop_w_max = int(self.upper_bound * height)
        h_max = height - crop_h_min
        w_max = width - crop_w_min
        crop_height = np.random.choice(np.arange(crop_h_min, crop_h_max), self.random_num, replace=False)
        crop_width = np.random.choice(np.arange(crop_w_min, crop_w_max), self.random_num, replace=False)
        crop_y = np.clip(np.random.normal(h_max / 2, h_max / 6, self.random_num).astype(int), 0, h_max - 1)
        crop_x = np.random.choice(w_max, self.random_num, replace=False)
        crop_y_end = crop_height + crop_y
        crop_y_end[crop_y_end>=height] = height
        crop_x_end = crop_width + crop_x

        mask_new = torch.zeros((self.random_num, height, width), dtype=torch.bool, device="cuda") #.cuda() #[N, H, W]
        for i in range(self.random_num):
            if crop_x_end[i] <= width:
                mask_new[i, crop_y[i]:crop_y_end[i], crop_x[i]:crop_x_end[i]] = True
            else:
                mask_new[i, crop_y[i]:crop_y_end[i], crop_x[i]:width] = True
                mask_new[i, crop_y[i]:crop_y_end[i], 0:(crop_x_end[i] - width)] = True

        return mask_new
  
    def ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        prediction_nan = prediction.clone().detach()
        target_nan = target.clone()
        prediction_nan[~mask_valid] = float('nan')
        target_nan[~mask_valid] = float('nan')

        valid_pixs = mask_valid.reshape((B, C,-1)).sum(dim=2, keepdims=True) + 1e-10
        valid_pixs = valid_pixs[:, :, :, None]

        gt_median = target_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        gt_median[torch.isnan(gt_median)] = 0
        gt_diff = (torch.abs(target - gt_median) ).reshape((B, C, -1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = prediction_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        pred_median[torch.isnan(pred_median)] = 0
        pred_diff = (torch.abs(prediction - pred_median)).reshape((B, C, -1))
        pred_s = pred_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)

        loss_sum = torch.sum(torch.abs(gt_trans - pred_trans)*mask_valid)
        return loss_sum

    def forward(self, prediction, target, mask=None, sem_mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = target.shape
        
        loss = 0.0
        valid_pix = 0.0

        device = target.device
        
        self.batch_valid = torch.tensor([1], device=device)[:,None,None,None]

        batch_limit = self.batch_limit
        
        random_sample_masks = self.get_random_masks_for_batch([H, W]) # [N, H, W]
        for i in range(B):
            # each batch
            mask_i = mask[i, ...] #[1, H, W]
            pred_i = prediction[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            target_i = target[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            random_sem_masks = random_sample_masks

            sampled_masks_num = random_sem_masks.shape[0]
            loops = int(np.ceil(sampled_masks_num / batch_limit))

            for j in range(loops):
                mask_random_sem_loopi = random_sem_masks[j*batch_limit:(j+1)*batch_limit, ...]
                mask_sample = (mask_i & mask_random_sem_loopi).unsqueeze(1) # [N, 1, H, W]
                loss += self.ssi_mae(
                    prediction=pred_i[:mask_sample.shape[0], ...], 
                    target=target_i[:mask_sample.shape[0], ...], 
                    mask_valid=mask_sample)
                valid_pix += torch.sum(mask_sample)
        
        # the whole image
        mask = mask * self.batch_valid.bool()
        loss += self.ssi_mae(
                    prediction=prediction, 
                    target=target, 
                    mask_valid=mask)
        valid_pix += torch.sum(mask)
        loss = loss / (valid_pix + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f'HDSNL NAN error, {loss}, valid pix: {valid_pix}')
        return loss * self.loss_weight