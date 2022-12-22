#read all the images in shuffledImages folder and copy them to COVID-19_Radiography_Dataset/Other/images 
import os
import shutil
x=1

for filename in os.listdir("COVID-19_Radiography_Dataset/Other/images"):
        os.remove("COVID-19_Radiography_Dataset/Other/images/"+filename)

for img in os.listdir('shuffledImages'):
    if img.endswith(".png") or img.endswith(".jpg"):
        src = os.path.join('shuffledImages', img)
        dst = os.path.join('COVID-19_Radiography_Dataset/Other/images', img)
        shutil.copyfile(src, dst)
        print("Copied image: ", x)
        x+=1
