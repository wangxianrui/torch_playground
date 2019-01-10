import torch

from model.mobilenetv2 import MobileNetV2Classify

inputs = torch.rand([1, 3, 224, 224])
model = MobileNetV2Classify(10)
print(model(inputs))