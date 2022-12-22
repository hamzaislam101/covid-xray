import torch
from torchvision import datasets, transforms
from torchvision import models
from torch import nn
from torch import optim

from coronaHackDataLoader import CoronaHackImageDataset

# Define the transforms for the images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#adding test data from CoronaHack-ChestX-Ray-Dataset
test_data2 = CoronaHackImageDataset("CoronaHack-ChestX-Ray-Dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset", "CoronaHack-ChestX-Ray-Dataset/Chest_xray_Corona_Metadata.csv",'test',data_transforms)

test_loader = torch.utils.data.DataLoader(
    test_data2,
    batch_size=64,
    shuffle=False
)


#load the model from model.pth and test it

device = torch.device("mps")
state_dict = torch.load('model.pth', map_location=device)
model = models.resnet50(pretrained=True)
model.load_state_dict(state_dict)
model.eval()
test_loss = 0.0
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(inputs)
        # Compute the loss
        loss = criterion(outputs, labels)
        # Update the test loss
        test_loss += loss.item() * inputs.size(0)
        # Get the predicted labels
        _, predicted = torch.max(outputs, 1)
        # Update the total number of correct predictions
        correct += (predicted == labels).sum().item()
        # Update the total number of images
        total += labels.size(0)

print('Test Loss:', test_loss/len(test_loader.dataset))
print('Test Accuracy:', correct/total)
