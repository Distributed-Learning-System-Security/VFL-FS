import torch
import torch.nn as nn
import torch.nn.functional as F


class ClientNet(nn.Module):
    def __init__(self, n_dim):
        super(ClientNet, self).__init__()
        self.fc1 = nn.Linear(n_dim, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class ServerNet(nn.Module):
    def __init__(self, n_input):
        super(ServerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.AvgPool2d(3, 2)
        self.fc1 = nn.Linear(n_input * 128, 3 * 32 * 32)
        self.fc2 = nn.Linear(448, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(x.shape[0], 3, 16, -1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 3, 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 3, 2)
        x = F.avg_pool2d(F.relu(self.conv3(x)), 3, 2)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
