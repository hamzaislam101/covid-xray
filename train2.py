from coronaHackDataLoader import CoronaHackImageDataset
import torch
from torchvision import datasets, transforms
from torchvision import models
from torch import nn
from torch import optim
from dataLoader import CustomImageDataset
from torch.utils.tensorboard import SummaryWriter



# Define the transforms for the images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data = CustomImageDataset("COVID-19_Radiography_Dataset",data_transforms)

#adding test data from CoronaHack-ChestX-Ray-Dataset
test_data2 = CoronaHackImageDataset("CoronaHack-ChestX-Ray-Dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset", "CoronaHack-ChestX-Ray-Dataset/Chest_xray_Corona_Metadata.csv",'test',data_transforms)
train_data2 = CoronaHackImageDataset("CoronaHack-ChestX-Ray-Dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset", "CoronaHack-ChestX-Ray-Dataset/Chest_xray_Corona_Metadata.csv",'train',data_transforms)

# Split the dataset into train, train, and validation sets
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))
test_size = len(data) - train_size - val_size

train_data, val_data, test_data = torch.utils.data.random_split(
    data,
    [train_size, val_size, test_size]
)


# Split the dataset into train, train, and validation sets
train_size2 = int(0.85 * len(train_data2))
val_size2 = len(train_data2) - train_size2

train_data2, val_data2 = torch.utils.data.random_split(
    train_data2,
    [train_size2, val_size2]
)


# Define the data loaders
train_loader = torch.utils.data.DataLoader(
    train_data2,
    batch_size=64,
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_data2,
    batch_size=64,
    shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_data2,
    batch_size=64,
    shuffle=False
)
test_loader2 = torch.utils.data.DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

# Define the classes
classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia', 'Other']

writer = SummaryWriter()

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Freeze the weights of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Define the additional layers
model.fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=1024, out_features=512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=512, out_features=9)
)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#move the model to the GPU
device = torch.device("mps")
model.to(device)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=8):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            # Update the training loss
            train_loss += loss.item() * inputs.size(0)
            print(f'train_loss: {train_loss}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(inputs)
                # Compute the loss
                loss = criterion(outputs, labels)
                # Update the validation loss
                val_loss += loss.item() * inputs.size(0)
                # Get the predicted labels
                _, predicted = torch.max(outputs, 1)
                # Update the total number of correct predictions
                correct += (predicted == labels).sum().item()
                # Update the total number of images
                total += labels.size(0)

        # Log the epoch statistics
        writer.add_scalar('Validation loss', val_loss/len(val_loader.dataset), epoch)
        writer.add_scalar('Validation accuracy', correct/total, epoch)
        writer.add_scalar('Training loss', train_loss/len(train_loader.dataset), epoch)

        # Print the epoch statistics
        print(f'Epoch: {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss/len(train_loader.dataset):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader.dataset):.4f}')
        print(f'Validation Accuracy: {correct/total:.4f}')

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader)

# Test the trained model
model.eval()
test_loss = 0.0
correct = 0
total = 0
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

# Print the test statistics
print('Test Loss:', test_loss/len(test_loader.dataset))
print('Test Accuracy:', correct/total)

print('Testing on first test set')

# Test the trained model
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader2:
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

#save the model
torch.save(model.state_dict(), 'model2.pth')

# Print the test statistics
print('Test Loss:', test_loss/len(test_loader.dataset))
print('Test Accuracy:', correct/total)


#pytorch tensor board for graphs

#data distribution name of labels of model data set train val test
#how many classes, how many samples in each class for train val test
#train val loss graph for each epoch
#train val accuracy graph for each epoch
#model parameters
#results with suffling and without shuffling


