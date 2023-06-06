import torch
import numpy as np

from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dimension, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dimension)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dimension, heads=8, head_dimension=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = head_dimension * heads
        project_out = not (heads == 1 and head_dimension == dimension)

        self.heads = heads
        self.scale = head_dimension ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dimension, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dimension),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dimension, hidden_dimension, dropout=0.):
        super(FeedForward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, dimension),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class Transformer(nn.Module):
    def __init__(self, dimension, layers, heads, head_dimension, mlp_dimension, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dimension, Attention(dimension, heads=heads, head_dimension=head_dimension, dropout=dropout)),
                PreNorm(dimension, FeedForward(dimension, mlp_dimension, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViViT(nn.Module):
    def __init__(
        self,
        height,
        width,
        frames,
        patch_height,
        patch_width,
        patch_frame,
        number_classes,
        dimension,
        layers=4,
        heads=3,
        in_channels=3,
        head_dimension=64,
        dropout=0.,
        embedding_dropout=0.,
        mlp_dimension=4
    ):
        super(ViViT, self).__init__()

        assert height % patch_height == 0 and width % patch_width == 0, 'Image dimensions must be divisible by the patch size'
        assert frames % patch_frame == 0, 'Frames must be divisible by frame patch size'
        
        number_image_patches = (height // patch_height) * (width // patch_width)
        number_frame_patches = (frames // patch_frame)

        patch_dimension = in_channels * patch_height * patch_width * patch_frame
        
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width, pf=patch_frame),
            nn.LayerNorm(patch_dimension),
            nn.Linear(patch_dimension, dimension),
            nn.LayerNorm(dimension)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, number_frame_patches, number_image_patches, dimension))
        self.dropout = nn.Dropout(embedding_dropout)
        
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dimension))
        self.spatial_transformer = Transformer(dimension, layers, heads, head_dimension, mlp_dimension, dropout)

        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dimension))
        self.temporal_transformer = Transformer(dimension, layers, heads, head_dimension, mlp_dimension, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, number_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        b, f, n, _ = x.shape

        x = x + self.pos_embedding[:, :f, :n]

        spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b=b, f=f)
        x = torch.cat((spatial_cls_tokens, x), dim = 2)

        x = self.dropout(x)

        x = rearrange(x, 'b f n d -> (b f) n d')

        x = self.spatial_transformer(x)
        x = rearrange(x, '(b f) n d -> b f n d', b=b)
        x = x[:, :, 0]

        temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b=b)
        x = torch.cat((temporal_cls_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        x = x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)


if __name__ == "__main__":
    
    img = torch.ones([1, 3, 16, 224, 224]).cuda()
    
    model = ViViT(224, 224, 16, 8, 8, 8, 2).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)
