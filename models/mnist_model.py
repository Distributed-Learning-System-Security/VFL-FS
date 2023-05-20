import torch
import torch.nn as nn
import torch.nn.functional as F


class ClientNet(nn.Module):
    def __init__(self, n_dim):
        super(ClientNet, self).__init__()
        self.fc = nn.Linear(n_dim, 128)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        return x


class ServerNet(nn.Module):
    def __init__(self, n_input):
        super(ServerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.fc1 = nn.Linear(n_input * 128, 1 * 2 * 28)
        # self.fc2 = nn.Linear(8448, 128)
        self.fc1 = nn.Linear(n_input * 128, 1 * 2 * 48)
        self.fc2 = nn.Linear(384, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(x.shape[0], 1, 16, -1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x
