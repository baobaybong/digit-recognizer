import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from datasets import load_dataset

def format(img, size=(28, 28)):
    img = Image.fromarray(img).convert('L').resize(size)

    transform = v2.Compose([v2.ToImageTensor(), v2.ConvertDtype(), v2.functional.invert])
    img_tensor = transform(img)  
    return img_tensor

def predict(img, model_selected):
    model_path = f"models/{model_selected}"
    print("Using", model_path)
    model = torch.load(model_path)
    model.eval()

    img = format(img)

    return int(torch.argmax(model(img.unsqueeze(0))))

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