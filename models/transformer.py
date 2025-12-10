import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from  .unet_modules import TimeEmbedding
from .class_embedder import ClassEmbedder


def get_pos_embed_sincos1D(embed_dim, positions):
   

    T = positions.shape[0]
    positions = positions.float()

    dim_half = embed_dim // 2
    # frequencies
    omega = torch.arange(dim_half, dtype=torch.float32)
    omega = omega / dim_half
    omega = 1.0 / (10000 ** omega)  # (dim_half,)
    out = positions[:, None] * omega[None, :]  # (T, dim_half)

    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (T, embed_dim)
    return emb

def get_pos_embed_sincos2D(grid_size, embed_dim):
    h, w = grid_size
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    y = y.reshape(-1)
    x = x.reshape(-1)

    if embed_dim % 4 != 0:
        raise ValueError("Embed dimension must be even")
    
    dim_each = embed_dim // 2

    emb_y = get_pos_embed_sincos1D(dim_each, y)   # (T, D/2)
    emb_x = get_pos_embed_sincos1D(dim_each, x)

    pos_embed = torch.cat([emb_y, emb_x], dim=1)  # (T, D)
    return pos_embed.float()

    
class PatchEmbed(nn.Module):
    #(B, C, 32, 32) → (B, T, d)
    def __init__(self, img_size=(32, 32), patch_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, z):
        
        x = self.proj(z)  
        x = x.flatten(2)
        x = x.transpose(1, 2) 
        return x
    

class AdaLNZero(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 3 * dim)
            )

        def forward(self, x, cond):
            # x: (B, T, D); cond: (B, D)
            gamma, beta, alpha = self.mlp(cond).chunk(3, dim=-1)
            x_norm = self.norm(x)
            return x + alpha.unsqueeze(1) * (gamma.unsqueeze(1) * x_norm + beta.unsqueeze(1))

class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.AdaLN = AdaLNZero(dim)
        self.AdaLN2 = AdaLNZero(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
           
        )

    def forward(self, x, cond):
        # x: (B, T, D)

        x_norm = self.AdaLN(x, cond)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output  

        x_norm = self.AdaLN2(x, cond)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output

        return x
    
class ReversePatchEmbed(nn.Module):
    #(B, T, d) → (B, 2*C, H, W)
    def __init__(self, img_size=(32, 32), patch_size=2, out_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size
        self.C = out_chans 
        self.T = (self.H // patch_size) * (self.W // patch_size)

        self.proj = nn.Linear(embed_dim, (patch_size ** 2) * self.C * 2)

    def forward(self, x):
        x =self.proj(x)  # (B, T, patch_size*patch_size*2*C)
        B, T, _ = x.shape
        x = x.view(B, T, self.C * 2, self.patch_size, self.patch_size)  # (B, T, 2*C, p, p)
        x = x.view(B, self.H // self.patch_size, self.W // self.patch_size, self.C * 2, self.patch_size, self.patch_size)  # (B, H//p, W//p, 2*C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, 2*C, H//p, p, W//p, p)
        x = x.reshape(B, self.C * 2, self.H, self.W)  # (B, 2*C, H, W)

        return x
    

class DiT(nn.Module):
    def __init__(self,
                 embedding_steps,
                 n_classes,
                 patch_size = 2,
                 embed_dim = 768,
                 depth = 12,
                 heads = 12,
                 in_channel = 3,
                 image_size = (32, 32), 
                 dropout = 0.1):
        super().__init__()

        self.H = image_size[0] // patch_size
        self.W = image_size[1] // patch_size

        self.in_channel = in_channel
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.timeEmbed = TimeEmbedding(embedding_steps, d_model = embed_dim, dim = embed_dim)
        self.classEmbed = ClassEmbedder(embed_dim = embed_dim, n_classes = n_classes, cond_drop_rate = 0.1)
        self.patchEmbed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dim)

        #positional embedding
        pos_embed = get_pos_embed_sincos2D((self.H, self.W), embed_dim)  # (T, D)
        self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)  # (1, T, D)

        self.DiTBlocks = nn.ModuleList([
            DiTBlock(dim=embed_dim, heads=heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.reverseEmbed = ReversePatchEmbed(img_size=image_size, patch_size=patch_size, out_chans=in_channel, embed_dim=embed_dim)
        
        self.cond_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )


    def forward(self, x, t, y):
        B, C, _, _ = x.shape

        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        elif torch.is_tensor(t) and len(t.shape) == 0:
            t = t.unsqueeze(0).to(x.device)
        
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B)
        elif t.shape[0] != B:
            t = t[:B]

        patch_embed = self.patchEmbed(x)
        time_emb = self.timeEmbed(t)
        x = patch_embed + self.pos_embed
        class_emb = self.classEmbed(y)
        cond = time_emb + class_emb

        for block in self.DiTBlocks:
            x = block(x, cond)

        x = self.reverseEmbed(x)
        noise, logvar = x.chunk(2, dim=1)
        return noise, logvar
