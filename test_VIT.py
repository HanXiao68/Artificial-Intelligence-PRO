import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self,in_channels,patch_size,embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size),
            nn.Flatten(2),
            # torch.transpose(, 0, 1)
        )
    def forward(self,x):
        return self.projection(x)

class VisionTransformer(nn.Module):
    def __init__(self,in_channels,patch_size,embed_dim,num_layers,num_heads,mlp_ratio):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels,patch_size,embed_dim)
        self.cls_token == nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,(64 // (6* patch_size ** 2)),embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer = nn.Transformer(embed_dim,num_heads,num_layers,dim_feedforward=embed_dim * mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens,x),dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x[:,0])

        return x[:,0]


#example

model = VisionTransformer(3,patch_size=8,embed_dim=768,num_layers=12,num_heads=12,mlp_ratio=4)

inputs = torch.randn(10, 3, 224, 224)

outputs = model(inputs)

print(outputs.shape)