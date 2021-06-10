import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class Custom_Dataset(Dataset):
    def __init__(self, images, segmented_images):
        self.images = images
        self.segmented_images = segmented_images
        self.lst_images = [i for i in sorted(os.listdir(f"{self.images}/")) if i.endswith(".png") or i.endswith(".jpg")]
        self.lst_segmented_images = [i for i in sorted(os.listdir(f"{self.segmented_images}/")) if i.endswith(".png") and not i.startswith(".") or i.endswith(".jpg") and not i.startswith(".")][:len(self.lst_images)]

    def __len__(self):
        return len(self.lst_images)

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor()])
        images = [f"{os.getcwd()}/{self.images}/{i}" for i in self.lst_images]
        segmented_images = [f"{os.getcwd()}/{self.segmented_images}/{i}" for i in self.lst_segmented_images]
        if idx < 0 or idx > len(images):
            return

        image = np.array(Image.open(images[idx]).resize((512, 512)).convert("RGB"))
        segmented_image = np.array(Image.open(segmented_images[idx]).resize((512, 512)).convert("L"))

        image = transform(image)
        segmented_image = transform(segmented_image)

        return image, segmented_image


