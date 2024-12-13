import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_receptive_field import receptive_field, receptive_field_for_unit, receptive_field_visualization_2d


class Net1D(nn.Module):
    def __init__(self):
        super(Net1D, self).__init__()
        self.conv = nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.avgpool(y)
        return y
    
class Net2D(nn.Module):
    def __init__(self):
        super(Net2D, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.avgpool(y)
        return y

class Net3D(nn.Module):
    def __init__(self):
        super(Net3D, self).__init__()
        self.conv = nn.Conv3d(3, 6, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(6)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.maxpool(y)
        return y

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"--> Using device: {device}")

model = Net1D().to(device)
receptive_field_dict = receptive_field(model, (3, 256), device=device)
receptive_field_for_unit(receptive_field_dict, "2", (1,))

model = Net2D().to(device)
receptive_field_dict = receptive_field(model, (3, 256, 256), device=device)
receptive_field_for_unit(receptive_field_dict, "2", (1,1))
receptive_field_visualization_2d(receptive_field_dict, "./examples/example.jpg", "example_receptive_field_2d")

model = Net3D().to(device)
receptive_field_dict = receptive_field(model, (3, 16, 16, 16), device=device)
receptive_field_for_unit(receptive_field_dict, "2", (1,1,1))
