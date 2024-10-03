import numpy as np
import cv2
import random
import torch
from scipy.linalg import orth
from skimage.util import random_noise
import matplotlib.pyplot as plt
from torch import nn
import numpy.linalg as la
import numpy.random as npr
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def add_gaussian_noise_new(img_tensor, noise_level1=5, noise_level2=20):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    img_tensor = img_tensor.to(device)

    if rnum > 0.6:
        noise_shape = img_tensor.shape[2:]
        noise = torch.from_numpy(np.random.normal(0, noise_level/255.0, noise_shape)).float().to(device)
        img_tensor += noise
    elif rnum < 0.4:
        noise_shape = img_tensor.shape[2:]
        noise = torch.from_numpy(np.random.normal(0, noise_level / 255.0, noise_shape)).float().to(device)
        img_tensor += noise.unsqueeze(0)
    else:
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(U, D), np.transpose(U))
        noise_shape = img_tensor.shape[2:][::-1]
        noise = torch.from_numpy(
            np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), noise_shape)).float().permute(2, 0, 1).to(device)
        img_tensor += noise

    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    return img_tensor


def add_Poisson_noise(img):


    vals = 100 ** (2 * random.random() + 2.0)

    if random.random() < 0.5:
        noisy_img = np.random.poisson(img * vals) / vals
    else:
        img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        noise_gray = np.random.poisson(img_gray * vals) / vals - img_gray
        noisy_img = img + noise_gray[:, :, np.newaxis]

    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return noisy_img



def add_rayleigh_noise_new(input_tensor, scale=0.02):
    """

    Args:
        input_tensor (torch.Tensor): 输入张量，格式为 [batch_size, channels, height, width]。
        sigma (float): Rayleigh 分布的参数

    Returns:
        torch.Tensor: 添加噪声后的张量，格式不变。
    """

    noise = torch.randn_like(input_tensor) * scale

    noisy_tensor = input_tensor + noise
    noisy_tensor = noisy_tensor.to(device)

    return noisy_tensor


def motion_blur_tensor(input_tensor, degree=8, angle=35):

    image = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree

    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    blurred = cv2.normalize(blurred, None, 0, 1, cv2.NORM_MINMAX)

    blurred_tensor = torch.from_numpy(np.array(blurred, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    blurred_tensor = F.interpolate(blurred_tensor, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)

    return blurred_tensor


def add_salt_pepper(image):


    noise_salt = random_noise(image.cpu().numpy(), mode="salt", amount=0.008, clip=True)

    noise_salt_tensor = torch.from_numpy(noise_salt).to(device)
    return noise_salt_tensor



def MeanFilter(img_tensor, kernel_size=(5, 5)):

    # weight = torch.ones(img_tensor.shape[1], 1, kernel_size[0], kernel_size[1]) / (kernel_size[0] * kernel_size[1])
    # weight = weight.to(img_tensor.device)
    weight = torch.ones(img_tensor.shape[1], 1, kernel_size[0], kernel_size[1]) / (kernel_size[0] * kernel_size[1])
    weight = weight.to(img_tensor.device)

    blurred_tensor = F.conv2d(img_tensor, weight, padding=(kernel_size[0] // 2, kernel_size[1] // 2), groups=img_tensor.shape[1])

    return blurred_tensor

def apply_random_degradation(image):

    random_number = random.randint(0, 4)

    if random_number == 0:
        image = add_gaussian_noise_new(image, noise_level1=5, noise_level2=20).to(device)
    elif random_number == 1:
        image = add_rayleigh_noise_new(image, scale=0.02)
    elif random_number == 2:
        image = motion_blur_tensor(image, degree=8, angle=25)
    elif random_number == 3:
        image = add_salt_pepper(image).to(device)
    elif random_number == 4:
        image = MeanFilter(image, kernel_size=(5, 5))
    else:
        print("Error！")
    return image


import os
input_folder = ''
output_folder = ''

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_names = os.listdir(input_folder)

for file_name in file_names:
    input_image_path = os.path.join(input_folder, file_name)

    input_image = Image.open(input_image_path).convert("RGB")

    transform = transforms.ToTensor()
    input_tensor = transform(input_image).unsqueeze(0)

    noisy_image_tensor = apply_random_degradation(input_tensor)

    transform = transforms.ToPILImage()
    noisy_image = transform(noisy_image_tensor.squeeze(0).cpu())

    file_name_without_extension, extension = os.path.splitext(file_name)

    output_image_path = os.path.join(output_folder, file_name_without_extension + extension)

    noisy_image.save(output_image_path)

print("Complete！")

