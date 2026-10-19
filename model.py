import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

class ResNet18(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        # ImageNet's 7x7 stride=2 conv + maxpool is not suitable for 32x32 inputs
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()

    def features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)  # 512 dim embedding

    def forward(self, x):
        return self.fc(self.features(x))

print(ResNet18())
