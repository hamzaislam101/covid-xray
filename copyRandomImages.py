#read the CoronaHack-ChestX-Ray-Dataset metadata file and read the bacterial images and copy them to a new folder
import pandas as pd
import os
import shutil

#read the metadata file and create a dataframe
metadata = pd.read_csv('CoronaHack-ChestX-Ray-Dataset/Chest_xray_Corona_Metadata.csv')

#get the unique values for the labels

#if label is 'Normal' then label_2_Virus_category is 'Normal'
#if label is 'Pnemonia' and label_2_Virus_category is empty then label_2_Virus_category is values from label_1_Virus_category
#if label is 'Pnemonia' and label_2_Virus_category is not empty then label_2_Virus_category is values from label_2_Virus_category

#replace the empty values in label_2_Virus_category with values from label_1_Virus_category
metadata['Label_2_Virus_category'].fillna(metadata['Label_1_Virus_category'], inplace=True)

#replace the empty values in label_2_Virus_category with 'Normal'
metadata['Label_2_Virus_category'].fillna('Normal', inplace=True)
labels = ['COVID-19', 'Normal', 'Virus']
#change the values in label_2_Virus_category which are not in labels to 'Other'

metadata['Label_2_Virus_category'] = metadata['Label_2_Virus_category'].apply(lambda x: x if x in labels else 'Other')

print(metadata['Label_2_Virus_category'].unique())

#read the Other images from the metadata file
otherImages = metadata[metadata['Label_2_Virus_category'] == 'Other']

#save the metadata for the Other images to a csv file
#otherImages.to_csv('CoronaHack-ChestX-Ray-Dataset/OtherImages.csv', index=False)

#copy the Other Train images to a new folder
for index, row in otherImages.iterrows():
    if row['Dataset_type'] == 'TRAIN':
        src = os.path.join('CoronaHack-ChestX-Ray-Dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train', row['X_ray_image_name'])
        dst = os.path.join('OtherImages', row['X_ray_image_name'])
        shutil.copyfile(src, dst)
        print("Copied Train image: ", index)