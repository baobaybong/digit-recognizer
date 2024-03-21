import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import torchvision.models
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import joblib

def format(img, size=(28, 28)):
    img = Image.fromarray(img).convert('L').resize(size)

    transform = v2.Compose([v2.ToImageTensor(), v2.ConvertDtype(), v2.functional.invert])
    img_tensor = transform(img)  
    return img_tensor

def predict(img, model_selected):
    model_path = f"models/{model_selected}"
    img = format(img)

    if model_path.endswith('.pth'):
        model = torch.load(model_path)
        model.eval()

        # Repeat to 3 channels for pre-trained models
        if "finetune" in model_path:
            img = img.repeat(3, 1, 1)

        return int(torch.argmax(model(img.unsqueeze(0))))
    else:
        model = joblib.load(model_path)
        flattened_data = img.reshape(1, -1)
        df = pd.DataFrame(flattened_data, columns=[f'pixel_{i}' for i in range(flattened_data.shape[1])])

        return model.predict(df)[0]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(1568, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
    
    def forward(self, x):
        return self.model(x)