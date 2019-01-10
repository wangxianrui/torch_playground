import torch
import torch.nn.functional


class MobileNetV1(torch.nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.model = torch.nn.Sequential(
            self.conv_bn(3, 32, 2),
            self.conv_dw(32, 64, 1),
            self.conv_dw(64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024, 1024, 1)
        )

    def conv_bn(self, inp, oup, stride):
        return torch.nn.Sequential(
            torch.nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            torch.nn.BatchNorm2d(oup),
            torch.nn.ReLU(inplace=True)
        )

    def conv_dw(self, inp, oup, stride):
        return torch.nn.Sequential(
            torch.nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            torch.nn.BatchNorm2d(inp),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.model(inputs)


class MobileNetV1Classify(torch.nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV1Classify, self).__init__()
        self.feature = MobileNetV1()
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(1024, num_classes)

    def forward(self, inputs):
        feature = self.feature(inputs)
        pool = self.pooling(feature)
        fc = self.fc(pool.view(-1, 1024))
        return torch.nn.functional.softmax(fc, dim=-1)
