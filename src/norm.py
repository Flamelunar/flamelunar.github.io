import torch
from torch.nn import LayerNorm

class myLayerNorm(LayerNorm):
    def __init__(self, name, dim, **kwargs):
        super().__init__(dim, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name
# 使用自定义的LayerNorm类
# norm_layer = myLayerNorm(name='my_layer_norm', dim=10)
# print(norm_layer)
# print(norm_layer.name)  # 输出自定义的属性name