import os
import pandas as pd
import torch
from PIL import Image

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
                if self.img_labels.empty:
                    self.img_labels = pd.DataFrame([[os.path.join(label+"/images", img), label]],columns=['image', 'label'])
                else:
                    imageLabel = pd.DataFrame([[os.path.join(label+"/images", img), label]],columns=['image', 'label'])
                    self.img_labels = pd.concat([self.img_labels,imageLabel], ignore_index=True)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def getImageLabelAsInteger(self, label):
        if label == 'COVID':
            return 0
        elif label == 'Lung_Opacity':
            return 1
        elif label == 'Normal':
            return 2
        elif label == 'Viral Pneumonia':
            return 3
        else:
            return -1

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, self.getImageLabelAsInteger(label)
