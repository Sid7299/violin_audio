import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_model(nn.Module):
    def __init__(self, input_length = 24000, input_channels=1, num_classes=5, stride=8, hidden_channels=32, kernel_size=80):
        super().__init__()

        # layer 1
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # layer 2 # check kernel size
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size//10, padding=(kernel_size//10 - 1)//2, stride=stride//2)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # layer 3
        self.conv3 = nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size=kernel_size//10, padding=(kernel_size//10 - 3)//2, stride=stride//2)
        self.bn3 = nn.BatchNorm1d(2 * hidden_channels)

        # layer 4
        self.conv4 = nn.Conv1d(2 * hidden_channels, 2 * hidden_channels, kernel_size=kernel_size//20, padding=(kernel_size//20 - 1)//2, stride=stride//4)
        self.bn4 = nn.BatchNorm1d(2 * hidden_channels)

        # fully connected layers
        self.fc1 = nn.Linear(2 * hidden_channels * (input_length//256), 1024)

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, input):

        batch_size = input.size()[0]

        x = self.conv1(input)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = F.relu(self.fc1(x.view(batch_size, -1)))

        x = self.fc2(x)
        return x

