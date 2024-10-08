import torch
import torch.nn as nn

class OCRModel(nn.Module):
    def __init__(self, res, label):
        super(OCRModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * res[0] * res[1], label)
     

    def forward(self, x):
        # Apply convolution and ReLU
        x = nn.ReLU()(self.conv1(x))  
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

