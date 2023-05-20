import torch.nn as nn
import torch.nn.functional as F


class ClientNet(nn.Module):
    def __init__(self, n_dim):
        super(ClientNet, self).__init__()
        self.fc = nn.Linear(n_dim, 128)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return x


class ServerNet(nn.Module):
    def __init__(self, n_input):
        super(ServerNet, self).__init__()
        self.fc1 = nn.Linear(128 * n_input, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x
