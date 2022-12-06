from dataLoader import CustomImageDataset

ds = CustomImageDataset("COVID-19_Radiography_Dataset")

print(ds[0])

print("Data Loader Test Complete")