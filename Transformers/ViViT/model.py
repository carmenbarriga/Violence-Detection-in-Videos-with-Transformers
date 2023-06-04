import torch.nn as nn

from einops.layers.torch import Rearrange


class TubeletEmbedding(nn.Module):
    def __init__(self, embedding_dimension, patch_t, patch_h, patch_w, channels):
        super(TubeletEmbedding, self).__init__()
        tubelet_dim = channels * patch_h * patch_w * patch_t
        self.tubelet_embedding = nn.Sequential(
            Rearrange(
                "b (t pt) c (h ph) (w pw) -> b t (h w) (pt ph pw c)",
                pt=patch_t,
                ph=patch_h,
                pw=patch_w,
            ),
            nn.Linear(tubelet_dim, embedding_dimension),
        )

    def forward(self, x):
        return self.tubelet_embedding(x)


class ViViT(nn.Module):
    def __init__(
        self,
        frames=40,
        height=256,
        width=256,
        patch_t=8,
        patch_h=8,
        patch_w=8,
        channels=3,
        embedding_dimension=512,
        device='cuda'
    ):
        super(ViViT, self).__init__()

        if frames % patch_t or height % patch_h or width % patch_w:
            print('Video dimensions should be divisible by tubelet size')
            return
        
        self.T = frames
        self.H = height
        self.W = width

        self.pt = patch_t
        self.ph = patch_h
        self.pw = patch_w

        self.channels = channels

        self.device = device

        self.tubelet_embedding = TubeletEmbedding(
            embedding_dimension=embedding_dimension,
            patch_t=self.pt,
            patch_h=self.ph,
            patch_w=self.pw,
            channels=self.channels
        )

    def forward(self, x):
        """ x is a video: (b, C, T, H, W) """

        tokens = self.tubelet_embedding(x)

        return tokens
