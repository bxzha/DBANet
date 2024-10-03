import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class TrainDataset(Dataset):
    def __init__(self, hr_folder, lr_folder, transform=None):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.hr_images = os.listdir(hr_folder)
        self.lr_images = os.listdir(lr_folder)
        self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_folder, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_folder, self.lr_images[idx])

        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')

        hr_image = self.transform(hr_image)
        lr_image = self.transform(lr_image)

        return lr_image, hr_image


class ValidationDataset(Dataset):
    def __init__(self, hr_folder, lr_folder, transform=None):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.hr_images = os.listdir(hr_folder)
        self.lr_images = os.listdir(lr_folder)
        self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_folder, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_folder, self.lr_images[idx])

        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')

        hr_image = self.transform(hr_image)
        lr_image = self.transform(lr_image)

        return lr_image, hr_image

class TestDataset(Dataset):
    def __init__(self, hr_folder, lr_folder, transform=None):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.hr_images = os.listdir(hr_folder)
        self.lr_images = os.listdir(lr_folder)
        self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_folder, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_folder, self.lr_images[idx])

        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')

        hr_image = self.transform(hr_image)
        lr_image = self.transform(lr_image)

        return lr_image, hr_image
