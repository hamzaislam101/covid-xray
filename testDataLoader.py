from dataLoader import CustomImageDataset
from coronaHackDataLoader import CoronaHackImageDataset

#ds = CustomImageDataset("COVID-19_Radiography_Dataset")

ds = CoronaHackImageDataset("CoronaHack-ChestX-Ray-Dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset", "CoronaHack-ChestX-Ray-Dataset/Chest_xray_Corona_Metadata.csv",'test')
print(ds[0])

print("Data Loader Test Complete")

#read image as PIL image
#im = Image.open("COVID-19_Radiography_Dataset/COVID/images/COVID-11.png", "r")
#arr = im.load() #pixel data stored in this 2D array
