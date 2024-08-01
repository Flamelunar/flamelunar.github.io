import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values

# 定义自注意力分类器模型
# class SelfAttentionClassifier(nn.Module):
#     def __init__(self, embed_dim, hidden_dim, num_classes):
#         super(SelfAttentionClassifier, self).__init__()
#         self.attention = SelfAttention(embed_dim)
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         attended_values = self.attention(x)
#         x = attended_values.mean(dim=1)  # 对每个位置的向量求平均
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         return x


# 定义多头自注意力模块
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 经过线性变换和残差连接
        x = self.fc(attended_values) + x

        return x

# # 定义多头自注意力分类器模型
# class MultiHeadSelfAttentionClassifier(nn.Module):
#     def __init__(self, embed_dim, num_heads, hidden_dim, num_classes):
#         super(MultiHeadSelfAttentionClassifier, self).__init__()
#         self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         x = self.attention(x)
#         x = x.mean(dim=1)  # 对每个位置的向量求平均
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         return x

    
myself = MultiHeadSelfAttention(embed_dim=128,num_heads=4) #要能被除净
print(myself)

x = torch.randn(2, 30, 128)
y = myself(x)
print(y)
print(y.shape)