import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from datetime import datetime
from tqdm import tqdm
from model import DBANet
from dataset import TrainDataset, ValidationDataset
import matplotlib.pyplot as plt
from calculate_value import calculate_psnr, calculate_ssim, calculate_lpips

batch_size = 2
learning_rate = 1e-4
num_epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_hr_folder = ''
train_lr_folder = ''
val_hr_folder = ''
val_lr_folder = ''


transform = transforms.Compose([
    transforms.ToTensor(),
])

if __name__ == '__main__':

    torch.manual_seed(42)

    train_dataset = TrainDataset(hr_folder=train_hr_folder, lr_folder=train_lr_folder, transform=transform)
    val_dataset = ValidationDataset(hr_folder=val_hr_folder, lr_folder=val_lr_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = DBANet(in_channels=3, img_size=128)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = [float('inf')]

    best_psnr = 0.0
    epochs_without_improvement = 0
    max_epochs_without_improvement = 20

    total_start_time = datetime.now()

    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()

        total_loss = 0.0

        model.train()

        for lr_images, hr_images in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            output_images = model(lr_images)

            loss = criterion(output_images, hr_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        total_psnr = 0.0
        num_samples = 0
        with torch.no_grad():
            for val_lr_images, val_hr_images in val_loader:
                val_lr_images, val_hr_images = val_lr_images.to(device), val_hr_images.to(device)

                val_output_images = model(val_lr_images)

                psnr = calculate_psnr(val_output_images, val_hr_images)

                total_psnr += psnr.mean().item()

                num_samples += val_lr_images.size(0)

        avg_psnr = total_psnr / num_samples

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_losses[-1]:.4f}, PSNR: {avg_psnr:.4f}')

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_epochs_without_improvement:
            print(f'Early stopping at epoch {epoch + 1} with PSNR {best_psnr:.4f}')
            break

        train_losses.append(total_loss / len(train_loader))

        val_loss = 0.0
        model.eval()

        with torch.no_grad():
            for val_lr_images, val_hr_images in val_loader:
                val_lr_images, val_hr_images = val_lr_images.to(device), val_hr_images.to(device)

                val_output_images = model(val_lr_images)
                val_loss += criterion(val_output_images, val_hr_images).item()

        val_losses.append(val_loss / len(val_loader))

        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Duration: {epoch_duration}')

    total_end_time = datetime.now()
    total_duration = total_end_time - total_start_time
    print(f'Total training duration: {total_duration}')

    torch.save(model.state_dict(), 'DBANet.pth')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig('DBANet.png')
    plt.show()
