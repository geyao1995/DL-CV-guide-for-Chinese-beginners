import torch.nn as nn
import torch.nn.functional as F


class CarBrandClsNet(nn.Module):
    def __init__(self):
        super(CarBrandClsNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 7)
        self.conv1_bn = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 26, 5)
        self.conv2_bn = nn.BatchNorm2d(26)
        self.conv3 = nn.Conv2d(26, 54, 3)
        self.conv3_bn = nn.BatchNorm2d(54)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(54 * 29 * 29, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = self.pool(F.relu(self.conv3(x)))  # torch.Size([8, 54, 29, 29])
        x = self.conv3_bn(x)
        # print(x.size())
        x = x.view(-1, 54 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
