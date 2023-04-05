from torch import nn

class AlexNet(nn.Module):
    def __init__(self, channel, dim1, dim2, num_classes=1):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(channel, 32, (3, 3))
        self.maxp1 = nn.MaxPool2d(3, 1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.maxp2 = nn.MaxPool2d(3, 1)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear((dim1 - 8) * (dim2 - 8) * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
