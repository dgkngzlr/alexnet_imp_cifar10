import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):

    def __init__(self, output_size = 10):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=(11,11), stride=4, padding=0)
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5,5), stride=1, padding=2)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3,3), stride=1, padding=1)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3,3), stride=1, padding=1)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.01)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3,3), stride=1, padding=1)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.01)
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)

        self.fc1 = nn.Linear(6*6*256, 4096)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        self.fc2 = nn.Linear(4096,4096)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        self.fc3 = nn.Linear(4096,output_size)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.01)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):

        y = self.conv1(x)
        y = F.relu(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = F.relu(y)
        y = self.conv4(y)
        y = F.relu(y)
        y = self.conv5(y)
        y = F.relu(y)
        y = self.pool3(y)

        y = torch.flatten(y, start_dim=1)

        y = self.fc1(y)
        y = F.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.dropout(y)
        y = self.fc3(y)

        return y
