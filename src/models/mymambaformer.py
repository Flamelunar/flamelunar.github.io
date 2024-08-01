import sys
sys.path.append("/home/ljj/3-biaffine-taketurn/src+kan+mambaformer-share+istraining+drop-best")
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_Family import Mamba_Layer, AM_Layer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
import numpy as np
from mamba_simple import Mamba, Block
from kan import KAN

class Model(nn.Module):
    """
    MambaFormer
    """
    def __init__(self, name, configs):
        super(Model, self).__init__()
        self._name = "mambaformer"
        self.configs = configs
        # self.pred_len = configs.pred_len
        # self.output_attention = configs.output_attention
     
        self.mamba_preprocess = Mamba_Layer(Mamba(configs.d_model), configs.d_model)
        #self.mamba_preprocess = Block(configs.d_model, mixer_cls=Mamba, norm_cls=nn.LayerNorm, fused_add_norm=True, residual_in_fp32=True)
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
        #self.out_proj=nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.out_proj = KAN(
            name="mambaformer_out_kan",
            layers_hidden=[configs.d_model, configs.c_out],   
        )

    def forward(self, x, dec_self_mask=None):
        
        x = self.mamba_preprocess(x)

        for i in range(self.configs.d_layers):
            x = self.AM_layers[i](x, dec_self_mask)
        
        kan_in = x.contiguous().view(-1, x.size()[2])  
        kan_out = self.out_proj(kan_in)             
        out = kan_out.view(x.size()[0], x.size()[1], -1)

        return out
    
    @property
    def name(self):
        return self._name
    
class Configs:
    def __init__(self, d_model, n_heads, d_layers, dropout, pred_len, output_attention, c_out,factor):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_layers = d_layers
        self.dropout = dropout
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.c_out = c_out
        self.factor = factor

if __name__ == "__main__": 
    # 定义配置参数
    
    # 创建Configs类的实例
    configs = Configs(
        d_model=512,
        n_heads=8,
        d_layers=6,
        dropout=0.1,
        pred_len=10,
        output_attention=False,
        c_out=512,
        factor=5
    )    
   
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型
    model = Model("mymambaformer",configs).cuda(mydevice)
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
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.---{total_params/(1024*1024):.2f}M')

    # 输出结果
    print(output.shape)  # 应该输出 (batch_size, sequence_length, c_out) 的形状