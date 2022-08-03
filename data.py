import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch


class HRdataset(data.Dataset):
    def __init__(self, image_root, gt_root, low_size, high_size):
        super(HRdataset, self).__init__()
        self.low_size = low_size
        self.high_size = high_size
        self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                      or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.image = sorted(self.image)
        self.gts = sorted(self.gts)
        self.size = len(self.image)
        self.high_transform = transforms.Compose([
            transforms.Resize((self.high_size, self.high_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.low_transform = transforms.Compose([
            transforms.Resize((self.low_size, self.low_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.high_size, self.high_size)),
            transforms.ToTensor()])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = self.rgb_loader(self.image[index])
        gt = self.binary_loader(self.gts[index])
        low = self.low_transform(image)
        high = self.high_transform(image)
        gt = self.gt_transform(gt)
        return low, high, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
