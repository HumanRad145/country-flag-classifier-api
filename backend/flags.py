import torch
import torch.nn as nn                               # for CNN stuff
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np


#from flag_mapping import flags


BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 25

INPUT_HEIGHT = 96
INPUT_WIDTH = 78

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#resize all images to the specified height and width

def load_prepare_data(train_data_path, test_data_path):
    #resize all images to the specified height and width
    transform = v2.Compose([
        v2.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
        v2.CenterCrop((INPUT_HEIGHT, INPUT_WIDTH)), 
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder(
        root=train_data_path,
        transform=transform
    )
    test_dataset = ImageFolder(
        root=test_data_path,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False
    )

    return train_loader, test_loader


class FLAG_CNN(nn.Module):
    def __init__(self):
        super(FLAG_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        #self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        #self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)

        #58368 for 3 conv 
        #116736 for 4 conv
        self.fc1 = nn.Linear(116736, 256)
        self.fc2 = nn.Linear(256, 224)

        #All layers
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

def train_model(model, train_loader, loss_func, optimzer, device, num_epoch):
    model.train()

    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):
            print(f"Epoch: {epoch}, Batch: {i}")
            images, labels = images.to(device), labels.to(device)
            optimzer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimzer.step()


def evaluation_model(model, testloader, device):
    model.eval()
    all_targets = []
    all_predictions, all_probabilities = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities  = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    #indices = np.arange(1, 225)
    #names = [flags[i] for i in indices]

    cm = confusion_matrix(all_targets, all_predictions)
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_predictions))
    #plt.figure(figsize=(26, 20))
    #plt.xticks(indices, names, rotation='vertical', fontsize=6)
    #sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    #plt.xlabel("Predicted")
    #plt.ylabel("True")
    #plt.title("Confusion Matrix")
    #plt.show()



    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions))

    accuracy = 100 * correct/total

    return accuracy, all_probabilities

def save_model(model, path, test_accuracy):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'test_accuracy': test_accuracy,
        'model_config': {
            'conv1': '32 filters, 3x3 kernels, ReLU',
            'conv2': '64 filters, 3x3 kernels, ReLU',
            'conv3': '128 filters, 3x3 kernels, ReLU',
            'conv4': '256 filters, 3x3 kernels, ReLU',
            'pool': 'MaxPooling 2x2',
            'dropout1': '0.1',
            'flatten': 'Yes',
            'fc1': '256 neurons, ReLU',
            'fc2': '222 neurons, Softmax'
        }
    }, path)
    print(f"\n Model saved to {path}")
        

if __name__ == "__main__":
    train_data_path = r'C:\Users\user\OneDrive\Programming\Projects\Flags CNN\data\flags_train'  #folder path to the subfolder classes
    test_data_path = r'C:\Users\user\OneDrive\Programming\Projects\Flags CNN\data\flags_test'

    train_loader, test_loader = load_prepare_data(train_data_path, test_data_path)

    model = FLAG_CNN().to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, loss_func, optimzer, DEVICE, NUM_EPOCHS)

    test_accuracy, test_probabilities = evaluation_model(model, test_loader, DEVICE)

    print(f"Test accuracy: {test_accuracy}")

    save_model(model, r'C:\Users\user\OneDrive\Programming\Projects\Flags CNN\flag_model_4conv.pth', test_accuracy)

    





            










