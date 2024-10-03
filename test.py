import torch
from torch.utils.data import DataLoader
from dataset import TestDataset
from model import DBANet
from calculate_value import calculate_psnr, calculate_ssim, calculate_lpips
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# file path
test_lr_folder = ''
test_hr_folder = ''

transform = transforms.Compose([
    transforms.ToTensor(),
])

lpips_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def visualize_and_save(lr_images, hr_images, output_images, index):
    lr_image = transforms.ToPILImage()(lr_images[0].cpu())
    hr_image = transforms.ToPILImage()(hr_images[0].cpu())
    output_image = transforms.ToPILImage()(output_images[0].cpu())

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(lr_image)
    axs[0].set_title('LR Image')

    axs[1].imshow(hr_image)
    axs[1].set_title('HR Image')

    axs[2].imshow(output_image)
    axs[2].set_title('Output Image')

    plt.savefig(f'result_visualization_{index}.png')
    plt.close()

def save_results_to_file(psnr_results, ssim_results, lpips_results):

    with open('evaluation_results.txt', 'w') as f:
        f.write('PSNR Results:\n')
        for psnr in psnr_results:
            f.write(f'{psnr:.4f}\n')

        f.write('\nSSIM Results:\n')
        for ssim in ssim_results:
            f.write(f'{ssim:.4f}\n')

        f.write('\nLPIPS Results:\n')
        for lpips in lpips_results:
            f.write(f'{lpips:.4f}\n')

if __name__ == '__main__':
    test_dataset_psnr_ssim = TestDataset(lr_folder=test_lr_folder, hr_folder=test_hr_folder, transform=transform)
    test_loader_psnr_ssim = DataLoader(test_dataset_psnr_ssim, batch_size=batch_size, shuffle=False, num_workers=1)

    test_dataset_lpips = TestDataset(lr_folder=test_lr_folder, hr_folder=test_hr_folder, transform=lpips_transform)
    test_loader_lpips = DataLoader(test_dataset_lpips, batch_size=batch_size, shuffle=False, num_workers=1)

    model = DBANet(in_channels=3, img_size=128).to(device)
    model.load_state_dict(torch.load('DBANet.pth'))
    model.eval()

    psnr_results = []
    ssim_results = []
    lpips_results = []

    with torch.no_grad():
        for i, (lr_images, hr_images) in enumerate(test_loader_psnr_ssim):

            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            output_images = model(lr_images).to(device)

            psnr_value = calculate_psnr(output_images, hr_images)
            ssim_value = calculate_ssim(output_images, hr_images)

            psnr_results.append(psnr_value.mean().item())
            ssim_results.append(ssim_value)

            if i < 10:
                visualize_and_save(lr_images, hr_images, output_images, i)

        for i, (lr_images, hr_images) in enumerate(test_loader_lpips):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            output_images = model(lr_images).to(device)

            lpips_value = calculate_lpips(output_images, hr_images)

            lpips_results.append(lpips_value.mean().item())

            if i < 10:
                visualize_and_save(lr_images, hr_images, output_images, i)

    avg_psnr = sum(psnr_results) / len(psnr_results)
    avg_ssim = sum(ssim_results) / len(ssim_results)
    avg_lpips = sum(lpips_results) / len(lpips_results)

    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average LPIPS: {avg_lpips:.4f}')

    save_results_to_file(psnr_results, ssim_results, lpips_results)
