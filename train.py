#import pytorch and other libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

batch_size = 16

#read the images from COVID-19 Radiography Dataset folder with images in the labeled folders
train_dataset = datasets.ImageFolder(root="COVID-19_Radiography_Dataset", transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#read the images from COVID-19 Radiography Dataset folder with images in the labeled folders
test_dataset = datasets.ImageFolder(root="COVID-19_Radiography_Dataset", transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
     transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#check the number of classes
classes = train_dataset.classes
print(classes)

#check the number of classes
classes = test_dataset.classes
print(classes)

#use the resnet18 model from pytorch models
model = models.resnet18(pretrained=True)

#add a new fully connected layer with 4 outputs
model.fc = nn.Linear(512, 4)

#provide input to the model
x = torch.randn(16, 3, 224, 224)
print(model(x).shape)

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(train_loader, epochs=10)

#evaluate the model
loss_0, acc_0 = model.evaluate(test_loader)

print("loss "+str(loss_0)+" acc "+str(acc_0))
