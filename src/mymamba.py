import torch
from torch import nn
from mamba_simple import Mamba, Block

class MambaBlock(Block):
    def __init__(self, dim, **kwargs):
        # 移除不在 Mamba 类中定义的参数
        super().__init__(dim, mixer_cls=Mamba, norm_cls=nn.LayerNorm, fused_add_norm=True, residual_in_fp32=True, **kwargs)

# 定义一个包含三层 MambaBlock 的模型
class MambaModel(nn.Module):
    def __init__(self, name, dim, layer_num, **kwargs):
        super().__init__()
        # 添加三个 MambaBlock 层
        self._name = name
        self.blocks = nn.ModuleList([
            MambaBlock(dim, **kwargs) for _ in range(layer_num)
        ])
        
    def forward(self, x):
        #初始化残差，因为一开始没有
        residual = x
        #print(residual)
        for block in self.blocks:
            y_mamba,residual = block(x, residual)
            #print(residual)
        return y_mamba
    
    @property
    def name(self):
        return self._name

# mymambablock = MambaModel(dim=16).to("cuda")
# print(mymambablock)
# batch, length, dim = 1, 64, 16
# x = torch.randn(batch, length, dim).to("cuda")

# y = mymambablock(x)


# print(y.shape)
# assert y.shape == x.shape