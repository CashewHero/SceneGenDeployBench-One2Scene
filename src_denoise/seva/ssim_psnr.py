import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt



def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)



def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))



def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()



def calculate_ssim_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity) (PyTorch version).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)
    return ssim


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).

    It is called by func:`calculate_ssim_pt`.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])

if __name__ == '__main__':
    # Define paths
    gt_path = '/data/liyi_chen/code/mvsplat360/256x448crop256x256_gt'
    pred1_path = '/data/liyi_chen/code/mvsplat360/256x448crop256x256_mvsplat'
    pred2_path = '/data/liyi_chen/code/mvsplat360/256x448crop256x256_mvsplat360'
    pred3_path = '/data/liyi_chen/code/mvsplat360/256x448crop256x256_mvsplat360'
    visual_comp_path = 'visual_comp'
    os.makedirs(visual_comp_path, exist_ok=True)

    # Get all image files from GT folder
    gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Transform to convert PIL image to tensor
    transform = transforms.ToTensor()

    # Initialize list to store differences
    differences = []

    for gt_file in tqdm(gt_files):
        # Check if corresponding files exist in pred folders
        pred1_file = os.path.join(pred1_path, gt_file)
        pred2_file = os.path.join(pred2_path, gt_file)
        pred3_file = os.path.join(pred3_path, gt_file)
        if not all(os.path.exists(f) for f in [pred1_file, pred2_file, pred3_file]):
            print(f"Warning: {gt_file} not found in one or more prediction folders")
            continue

        # Load images
        gt_img = Image.open(os.path.join(gt_path, gt_file)).convert('RGB')
        pred1_img = Image.open(pred1_file).convert('RGB')
        pred2_img = Image.open(pred2_file).convert('RGB')
        pred3_img = Image.open(pred3_file).convert('RGB')

        # Convert to tensors
        gt_tensor = transform(gt_img).unsqueeze(0)  # Add batch dimension
        pred1_tensor = transform(pred1_img).unsqueeze(0)
        pred2_tensor = transform(pred2_img).unsqueeze(0)
        pred3_tensor = transform(pred3_img).unsqueeze(0)

        # Calculate metrics for all predictions
        ssim1 = calculate_ssim_pt(gt_tensor, pred1_tensor, 0)
        psnr1 = calculate_psnr_pt(gt_tensor, pred1_tensor, 0)
        ssim2 = calculate_ssim_pt(gt_tensor, pred2_tensor, 0)
        psnr2 = calculate_psnr_pt(gt_tensor, pred2_tensor, 0)
        ssim3 = calculate_ssim_pt(gt_tensor, pred3_tensor, 0)
        psnr3 = calculate_psnr_pt(gt_tensor, pred3_tensor, 0)

        # Get the values
        ssim1_val = ssim1.item()
        psnr1_val = psnr1.item()
        ssim2_val = ssim2.item()
        psnr2_val = psnr2.item()
        ssim3_val = ssim3.item()
        psnr3_val = psnr3.item()

        # Calculate combined scores (SSIM + PSNR) for each prediction
        score1 = ssim1_val + psnr1_val
        score2 = ssim2_val + psnr2_val
        score3 = ssim3_val + psnr3_val

        # Find the gap between first and second place
        scores = [(score1, 'pred1'), (score2, 'pred2'), (score3, 'pred3')]
        scores.sort(reverse=True)
        gap = scores[0][0] - scores[1][0]
        
        # Store the differences and file info
        differences.append({
            'filename': gt_file,
            'gap': gap,
            'winner': scores[0][1],
            'second': scores[1][1],
            'ssim1': ssim1_val,
            'psnr1': psnr1_val,
            'ssim2': ssim2_val,
            'psnr2': psnr2_val,
            'ssim3': ssim3_val,
            'psnr3': psnr3_val,
            'score1': score1,
            'score2': score2,
            'score3': score3
        })

    # Sort by the gap between first and second place
    differences.sort(key=lambda x: x['gap'], reverse=True)

    # Save top 100 images
    print("\nSaving top 100 images with largest gap between first and second place...")
    for i, diff in enumerate(differences[:100]):
        filename = diff['filename']
        save_dir = os.path.join(visual_comp_path, f"{i+1:03d}_gap{diff['gap']:.4f}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each image separately
        Image.open(os.path.join(gt_path, filename)).convert('RGB').save(os.path.join(save_dir, 'gt.png'))
        Image.open(os.path.join(pred1_path, filename)).convert('RGB').save(os.path.join(save_dir, 'pred1.png'))
        Image.open(os.path.join(pred2_path, filename)).convert('RGB').save(os.path.join(save_dir, 'pred2.png'))
        Image.open(os.path.join(pred3_path, filename)).convert('RGB').save(os.path.join(save_dir, 'pred3.png'))
        
        # Save metrics to a text file
        with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Filename: {filename}\n")
            f.write(f"Winner: {diff['winner']}\n")
            f.write(f"Second: {diff['second']}\n")
            f.write(f"Gap: {diff['gap']:.4f}\n\n")
            f.write(f"Pred1 - SSIM: {diff['ssim1']:.4f}, PSNR: {diff['psnr1']:.4f}, Score: {diff['score1']:.4f}\n")
            f.write(f"Pred2 - SSIM: {diff['ssim2']:.4f}, PSNR: {diff['psnr2']:.4f}, Score: {diff['score2']:.4f}\n")
            f.write(f"Pred3 - SSIM: {diff['ssim3']:.4f}, PSNR: {diff['psnr3']:.4f}, Score: {diff['score3']:.4f}\n")
        
        print(f"Saved {i+1}/100: {save_dir}")
        print(f"Gap: {diff['gap']:.4f}, Winner: {diff['winner']}, Second: {diff['second']}")

    # Print summary
    print("\nSummary:")
    print(f"Total images processed: {len(differences)}")
    print(f"Top 100 images saved to {visual_comp_path}")
    print("\nTop 5 largest gaps:")
    # for i, diff in enumerate(differences[:5]):
    #     print(f"\n{i+1}. {diff['filename']}")
    #     print(f"   Gap: {diff['gap']:.4f}")
    #     print(f"Winner: {diff['winner']} (Score: {diff[f'score{diff['winner'][-1]}']:.4f})")
    #     print(f"Second: {diff['second']} (Score: {diff[f'score{diff['second'][-1]}']:.4f})")