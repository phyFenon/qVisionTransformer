import torch
import math
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# S1: Define the PatchEmbedding class
class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_size: int):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        patches = self.patch_embed(x)
        batch_size, _, h, w = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        return patches

#S2: Define the PositionalEmbedding class
class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.pos_encoding = self.positional_encoding()
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, input_matrix):
        batch_size, num_patches, d_model = input_matrix.size()
        # Add class token to the input matrix at position 0
        input_matrix = torch.cat([self.class_token.expand(batch_size, 1, d_model), input_matrix], dim=1)# 使用cat将cls token插入在patches信息的前面，沿着第二个维度进行连接
        input_matrix += self.pos_encoding[:, :num_patches + 1]  # Add positional encoding to the input matrix
        return input_matrix

    def positional_encoding(self):
        pos_encoding = torch.zeros(1, self.num_patches + 1, self.d_model) #生成一个dim为（1,num_patches+1,d_model）的tensor
        position = torch.arange(0, self.num_patches + 1).unsqueeze(1) #生成一个position的vector (num_patches+1,1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))#生成一个关于 d_model里的feature的（1，d_model//2)的tensor
        #print(f'pos_shape:{position.shape},div_term shape:{div_term.shape}')
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term) #i为偶数
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term) #i为奇数

        return pos_encoding
 
 #S3: Transformer Encoder contains: One Transformer Unit Block, Num of Layer
 # One unit Transformer block includes functions: 
 # LayerNorm(according to the dimension of num_patches)
 # Multi-head-attention
 # ResidualAddition
 # Multi-layer-perceptron(两层全连接层，可以包含dropout_rate)
 
 #S3.1 根据输入矩阵的（batch_size, num_seq, d_model)中的num_seq维度进行归一化处理，此步骤包含两个伸缩变换参数 \alpha bias 
class LayerNormalization(nn.Module):
    def __init__(self, seq_len, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(seq_len))
        self.bias = nn.Parameter(torch.zeros(seq_len))
        self.eps = eps

    def forward(self, x):
        #输入矩阵x的维度为（batch_size, num_seq, d_model）
        mean = x.mean(dim=1, keepdim=True)  # 按照 num_seq 计算均值
        std = x.std(dim=1, keepdim=True)    # 按照 num_seq 计算标准差
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
 
 #S3.2 根据处理后的输入矩阵(batch_size, num_seq, d_model) 中每个 （;, num_seq,d_model）进行多头注意力机制操作，参数num_heads, d_k = d_model/ num_heads
 #首先需要制备 Q,K,V三个矩阵，使用线性连接层可不加bias
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        # 参数d_model, num_of_heads: h
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        
        # make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
   

    @staticmethod
    def attention(query, key, value):
        # q,k,v size (batch,h, seq_len, d_k）
        d_k = query.shape[-1]# 获取d_k的大小
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        attention_scores = attention_scores.softmax(dim=-1) #(batch, h, seq_len,seq_len) #softmax后是第四个维度加起来为1. 
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value)

    def forward(self, q, k, v):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        # 这个操作是为了方便计算， q@k^T =(batch, h, seq_len, d_k) *(batch, h, d_k, seq_len)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x = MultiHeadAttentionBlock.attention(query, key, value)
        
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
     
#S3.3 ResidualConnection 描述的是残差神经网络，需要传入两个维度一样的矩阵，进行加法运算
class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, residual):
        #传入两个维度一致的矩阵(batch,seq_len,d_model)
        assert x.size() == residual.size(), "Input tensor dimensions must match for residual connection"
        return x + residual

#S3.4 MLPlayer，含有两层线性连接层。(batch_size,seq_len,d_model),-->(batch_size,seq_len,d_hidde)-->(batch_size,seq_len,d_model)
class MultiLayerP(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(self.d_model, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.gelu(x)  #中间hidden_layer的神经元激活函数使用GELU
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)

#将上述步骤的S3.1 至 S3.4有效组合起来，形成一个TransformerEncoder,包含layer的深度
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, mlp_hidden, num_layers, dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_hidden = mlp_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.attention_blocks = nn.ModuleList([MultiHeadAttentionBlock(d_model, num_heads) for _ in range(num_layers)])
        self.mlp_layers = nn.ModuleList([MultiLayerP(d_model, mlp_hidden, dropout_rate) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([LayerNormalization(d_model) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            # MultiHeadAttentionBlock
            residual = x
            x = self.layer_norms[i](x)
            x = self.attention_blocks[i](x, x, x) + residual
            
            # MLP Layer
            residual = x
            x = self.layer_norms[i](x)
            x = self.mlp_layers[i](x) + residual

        return x

#利用上述PatchEmbedding, PositionalEmbedding, TransformerEncoder搭建 VisionTransformer
class VisionTransformer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_hidden, num_layers, patch_size, image_size, num_classes, dropout_rate):
        super().__init__()
        self.patch_embedder = PatchEmbedding(d_model, patch_size)
        self.pos_embedder = PositionalEmbedding((image_size // patch_size) ** 2, d_model)
        self.transformer_encoder = TransformerEncoder(d_model, num_heads, mlp_hidden, num_layers, dropout_rate)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        patches = self.patch_embedder(x)
        embeddings = self.pos_embedder(patches)
        features = self.transformer_encoder(embeddings)
        output = self.fc(features[:,0,:])  # 提取所有样本中的第一个class token作为分类的全连接层输入
        return output