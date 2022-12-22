# import pandas as pd

# #read the metadata file and create a dataframe
# metadata = pd.read_csv('CoronaHack-ChestX-Ray-Dataset/Chest_xray_Corona_Metadata.csv')

# #get the unique values for the labels

# #if label is 'Normal' then label_2_Virus_category is 'Normal'
# #if label is 'Pnemonia' and label_2_Virus_category is empty then label_2_Virus_category is values from label_1_Virus_category
# #if label is 'Pnemonia' and label_2_Virus_category is not empty then label_2_Virus_category is values from label_2_Virus_category

# #replace the empty values in label_2_Virus_category with values from label_1_Virus_category
# metadata['Label_2_Virus_category'].fillna(metadata['Label_1_Virus_category'], inplace=True)

# #replace the empty values in label_2_Virus_category with 'Normal'
# metadata['Label_2_Virus_category'].fillna('Normal', inplace=True)
# labels = ['COVID-19', 'Normal', 'Virus']
# #change the values in label_2_Virus_category which are not in labels to 'Other'
# metadata['Label_2_Virus_category'] = metadata['Label_2_Virus_category'].apply(lambda x: x if x in labels else 'Other')

# print(metadata['Label_2_Virus_category'].unique())


# read the images from COVID-19 Radiography Dataset folders and print the number of images in each folder
import os
import random
import shutil
import pandas as pd

# read the folders in the COVID-19 Radiography Dataset
folders = os.listdir('COVID-19_Radiography_Dataset')

# read the images in each folder and print the number of images in each folder
for folder in folders:
    # if folder is a file then skip
    if not os.path.isdir('COVID-19_Radiography_Dataset/' + folder):
        continue
    images = os.listdir('COVID-19_Radiography_Dataset/' + folder + '/images')
    print(folder, len(images))

    x = 0
    # copy 25 % images randomly to Other/images folder
    # if folder is COVID-19 then copy 25 % images randomly to Other/images folder
    if folder != 'Other':
        for image in images:
            if image.endswith('.png') or image.endswith('.jpg'):
                #if random number between 1 and 4 is 2 then copy the image to Other/images folder
                if random.randint(1, 4) == 2:
                    x+=1
                    src = os.path.join('COVID-19_Radiography_Dataset/' + folder + '/images', image)
                    dst = os.path.join('COVID-19_Radiography_Dataset/Other/images', image)
                    shutil.copyfile(src, dst)
                    print("Copied image: ", x)    





