import torch
import torch.nn as nn
import torch.nn.functional as F

class InstrinsicDecoder(nn.Module):
    def __init__(self):
        super(InstrinsicDecoder, self).__init__()
        self.fc1 = nn.Linear(10000, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 9)
    def forward(self,x):
        x=torch.concat(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = x.float()
        return output