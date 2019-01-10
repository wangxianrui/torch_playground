import torch
import torch.nn.functional


class LinearBottleneck(torch.nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=torch.nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(inplanes * t)

        self.conv2 = torch.nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
                                     groups=inplanes * t)
        self.bn2 = torch.nn.BatchNorm2d(inplanes * t)

        self.conv3 = torch.nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(outplanes)

        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


class MobileNetV2(torch.nn.Module):

    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.activation = torch.nn.ReLU6(inplace=True)

        self.t = [0, 1, 6, 6, 6, 6, 6, 6]
        self.c = [32, 16, 24, 32, 64, 96, 160, 320]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]

        self.conv1 = torch.nn.Conv2d(3, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = torch.nn.BatchNorm2d(self.c[0])
        self.bottlenecks = self._make_bottlenecks()

        # Last convolution has 1280 output chatorch.nn.ls for scale <= 1
        self.conv_last = torch.nn.Conv2d(self.c[-1], 1280, kernel_size=1, bias=False)
        self.bn_last = torch.nn.BatchNorm2d(1280)

    def _make_repeat(self, inchannel, outchannel, t, n, s):
        repeat = []

        # first bottle
        repeat.append(LinearBottleneck(inplanes=inchannel, outplanes=outchannel, stride=s, t=t))

        if n > 1:
            for i in range(n - 1):
                repeat.append(LinearBottleneck(inplanes=outchannel, outplanes=outchannel, stride=1, t=t))
        return torch.nn.Sequential(*repeat)

    def _make_bottlenecks(self):
        bottlenecks = []

        for i in range(len(self.c) - 1):
            bottlenecks.append(self._make_repeat(self.c[i], self.c[i + 1], self.t[i + 1], self.n[i + 1], self.s[i + 1]))

        return torch.nn.Sequential(*bottlenecks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.bottlenecks(x)

        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        return x


class MobileNetV2Classify(torch.nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2Classify, self).__init__()
        self.feature = MobileNetV2()
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.dropout = torch.nn.Dropout(p=0.2, inplace=True)
        self.fc = torch.nn.Linear(1280, num_classes)

    def forward(self, inputs):
        feature = self.feature(inputs)
        pool = self.pooling(feature)
        pool = self.dropout(pool)
        fc = self.fc(pool.view(-1, 1280))
        return torch.nn.functional.softmax(fc, dim=-1)
