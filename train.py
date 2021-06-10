import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataset import Custom_Dataset
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from model import UNET
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Custom_Dataset(images="images", segmented_images="trimaps")
train_set, test_set = random_split(dataset, [round(len(dataset)*0.8), round(len(dataset)*0.2)])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
net = UNET(3, 1).to(device)


lr = 0.0001
batch_size = 32
epochs = 5
optimizer = optim.Adam(net.parameters(), lr=lr)
loss_function = nn.BCELoss()


for epoch in range(epochs):
    for x, y in train_loader:
        output = net(x)
        net.zero_grad()
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

