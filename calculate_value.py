import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import cv2
import lpips
import warnings


warnings.simplefilter("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_psnr(res_tensor, gt_tensor):

    res_tensor = torch.clamp(res_tensor, 0, 1)
    gt_tensor = torch.clamp(gt_tensor, 0, 1)

    mse = F.mse_loss(res_tensor, gt_tensor, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    if (mse == 0).any():
        return 100

    max_pixel = 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))

    return psnr

def ssim(img1_tensor, img2_tensor):
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2

    window = cv2.getGaussianKernel(11, 1.5)
    window = torch.tensor(np.outer(window, window.transpose()), dtype=torch.float32, device=img1_tensor.device)

    mu1 = F.conv2d(img1_tensor[:, None, :, :], weight=window.view(1, 1, 11, 11), padding=5)
    mu2 = F.conv2d(img2_tensor[:, None, :, :], weight=window.view(1, 1, 11, 11), padding=5)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1_tensor[:, None, :, :]**2, weight=window.view(1, 1, 11, 11), padding=5) - mu1_sq
    sigma2_sq = F.conv2d(img2_tensor[:, None, :, :]**2, weight=window.view(1, 1, 11, 11), padding=5) - mu2_sq
    sigma12 = F.conv2d(img1_tensor[:, None, :, :] * img2_tensor[:, None, :, :], weight=window.view(1, 1, 11, 11), padding=5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def calculate_ssim(img1_tensor, img2_tensor, border=0):
    '''
    calculate SSIM between two tensors
    img1_tensor, img2_tensor: [batch_size, channel, height, width] (values in [0, 1])
    '''
    if not img1_tensor.shape == img2_tensor.shape:
        raise ValueError('Input tensors must have the same dimensions.')

    batch_size, channels, height, width = img1_tensor.shape

    img1_tensor = img1_tensor[:, :, border:height-border, border:width-border]
    img2_tensor = img2_tensor[:, :, border:height-border, border:width-border]

    if channels == 1:
        return ssim(img1_tensor.squeeze(), img2_tensor.squeeze())
    elif channels == 3:
        ssims = []
        for i in range(channels):
            ssims.append(ssim(img1_tensor[:, i, :, :], img2_tensor[:, i, :, :]))
        return torch.tensor(ssims).mean().item()
    else:
        raise ValueError('Wrong number of channels.')


def calculate_lpips(image1_tensor, image2_tensor):

    lpips_model = lpips.LPIPS(net="alex").to(device)

    distance = lpips_model(image1_tensor, image2_tensor)

    return distance
