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

    transform = v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype(), v2.functional.invert])
    img_tensor = transform(img)  
    return img_tensor

def predict(img, model_path="models/mlp.pth"):
    model = torch.load(model_path)
    model.eval()

    img = format(img)

    # return int(torch.argmax(model(img)))

    return int(torch.argmax(model(img)))
    # return img.numpy()