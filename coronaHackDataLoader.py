import os
import pandas as pd
import torch
from PIL import Image

class CoronaHackImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, metadata_file, phase='train', transform=None, target_transform=None):

        #this is for the COVID-19 Radiography Dataset
        #read the four label folders for the images and the labels
        labels = ['Normal' 'Virus' 'Other' 'COVID-19']
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.phase = phase
        self.meta_data = pd.read_csv(metadata_file)
        self.meta_data['Label_2_Virus_category'].fillna(self.meta_data['Label_1_Virus_category'], inplace=True)

        #replace the empty values in label_2_Virus_category with 'Normal'
        self.meta_data['Label_2_Virus_category'].fillna('Normal', inplace=True)
        xrayLabels = ['COVID-19', 'Normal', 'Virus']
        #change the values in label_2_Virus_category which are not in labels to 'Other'
        #self.meta_data['Label_2_Virus_category'] = self.meta_data['Label_2_Virus_category'].apply(lambda x: x if x in xrayLabels else 'Other')

        if self.phase == 'train':
            self.meta_data = self.meta_data[self.meta_data['Dataset_type'] == 'TRAIN']
        elif self.phase == 'test':
            self.meta_data = self.meta_data[self.meta_data['Dataset_type'] == 'TEST']
            #only keep the images with label_2_Virus_category as in xraylabels
            self.meta_data = self.meta_data[self.meta_data['Label_2_Virus_category'].isin(xrayLabels)]
        else:
            print("Invalid phase. Please enter 'train', 'val', or 'test'")
            return
        
        
        self.img_labels = pd.DataFrame(columns=['image','phase' ,'label'])

        #create a dataframe with the image path and the label
        for row in self.meta_data.itertuples():
            if self.img_labels.empty:
                self.img_labels = pd.DataFrame([[row.X_ray_image_name, self.phase,row.Label_2_Virus_category]],columns=['image','phase' ,'label'])
            else:
                imageLabel = pd.DataFrame([[row.X_ray_image_name,self.phase ,row.Label_2_Virus_category]],columns=['image','phase' ,'label'])
                self.img_labels = pd.concat([self.img_labels,imageLabel], ignore_index=True)


    def __len__(self):
        return len(self.img_labels)
    
    def getImageLabelAsInteger(self, label):
        labels = ['Normal', 'ARDS', 'Virus', 'COVID-19','SARS','bacteria','Streptococcus','Lung_Opacity' ,'Other']
        return labels.index(label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,1],self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, self.getImageLabelAsInteger(label)
