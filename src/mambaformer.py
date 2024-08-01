from asyncio import FastChildWatcher
import sys
sys.path.append("/home/ljj/3-biaffine-taketurn/src+mambaformer")
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_Family import Mamba_Layer, AM_Layer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
import numpy as np
from mamba_simple import Mamba, Block
from torchsummaryX import summary  # 需要安装 torchsummary 包
from common import *

class Mambaformer(nn.Module):
    """
    MambaFormers
    """
    def __init__(self, name, configs):
        super(Mambaformer, self).__init__()
        self._name = name
        self.configs = configs
        # self.pred_len = configs.pred_len
        # self.output_attention = configs.output_attention
     
        #self.mamba_preprocess = Mamba_Layer(Mamba(configs.d_model), configs.d_model)
        self.AM_layers = nn.ModuleList(
            [
                AM_Layer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads, ),
                    Mamba(configs.d_model),
                    configs.d_model,
                    configs.dropout
                ) 
                for i in range(configs.d_layers)
            ]
        )
        self.out_proj=nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x, dec_self_mask=None, is_training=False):
        
        #x = self.mamba_preprocess(x)

        for i in range(self.configs.d_layers):
            x = self.AM_layers[i](x, dec_self_mask, is_training)
             
        out = self.out_proj(x)

        return out
    
    @property
    def name(self):
        return self._name
    
class Configs:
    def __init__(self, d_model, n_heads, d_layers, dropout, output_attention, c_out,factor):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_layers = d_layers
        self.dropout = dropout
        self.output_attention = output_attention
        self.c_out = c_out
        self.factor = factor

if __name__ == "__main__": 
    # 定义配置参数
    
    # 创建Configs类的实例
    configs = Configs(
        d_model=512,
        n_heads=8,
        d_layers=1,
        dropout=0.33,
        output_attention=False,
        c_out=512,
        factor=5
    )    
   
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型
    
    model = Mambaformer("mymambaformer",configs).cuda(mydevice)
    
    print(model)
    
    
    # 创建一个假定的输入张量 (batch_size, sequence_length, d_model)
    # 假设我们有一批包含10个序列的输入，每个序列长度为20，模型维度为512
    batch_size = 10
    sequence_length = 20
    d_model = configs.d_model

    # 随机初始化输入张量
    x = torch.randn(batch_size, sequence_length, d_model).cuda(mydevice)   
    
    # 调用模型
    output = model(x)

    # 输出结果
    print(output.shape)  # 应该输出 (batch_size, sequence_length, c_out) 的形状
    
  
    # 输出模型的参数量
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params/(1024*1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')
    
