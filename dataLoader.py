import os
import pandas as pd
import torch
from torchvision.io import read_image

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):

        #this is for the COVID-19 Radiography Dataset
        #read the four label folders for the images and the labels
        labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        self.img_labels = pd.DataFrame(columns=['image', 'label'])
        for label in labels:
            path = os.path.join(img_dir, label)
            path = os.path.join(path, 'images')
            for img in os.listdir(path):
                self.img_labels = self.img_labels.append({'image': os.path.join(label+"/images", img), 'label': label}, ignore_index=True)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        print(img_path)
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
