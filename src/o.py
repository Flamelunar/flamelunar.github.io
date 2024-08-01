import torch
import torchvision.models as models
from torchsummary import summary
# 创建模型对象
model = models.resnet50()
# 计算模型计算量
summary(model, input_size=(3, 224, 224))