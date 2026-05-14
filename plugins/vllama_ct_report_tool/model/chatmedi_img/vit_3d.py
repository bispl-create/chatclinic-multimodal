import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels=1, patch_size=(12, 24, 24), emb_size=7):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch_size, in_channels, D, H, W)
        x = self.proj(x)  # (batch_size, emb_size, D', H', W')
        return x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, emb_size)

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, emb_size):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, emb_size))

    def forward(self, x):
        return x + self.pos_embed

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, ff_hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(emb_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, emb_size)
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # Self-attention block
        attn_output = self.attn(x)
        x = self.norm1(x + attn_output)

        # Feed-forward block
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class VisionTransformer3D(nn.Module):
    def __init__(self, in_channels=1, patch_size=(12, 24, 24), emb_size=768, num_layers=12, num_heads=12, ff_hidden_dim=3072, img_size=(120, 480, 480)):
        super().__init__()
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, emb_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.pos_embed = PositionalEmbedding(num_patches, emb_size)

        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(emb_size, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])

        #self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(emb_size),
        #     nn.Linear(emb_size, num_classes)
        # )

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)

        # Add CLS token
        batch_size = x.shape[0]
        #cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        #x = torch.cat((cls_tokens, x), dim=1)
        print('x.shape', x.shape)
        # Positional Embedding
        x = self.pos_embed(x)

        # Transformer Layers
        for encoder in self.transformer_encoders:
            x = encoder(x)

        # Classification Head (using CLS token output)
        # cls_output = x[:, 0]  # (batch_size, emb_size)
        # logits = self.mlp_head(cls_output)

        return x

# Example usage
# if __name__ == "__main__":
#     # Input: (batch_size, in_channels, D, H, W)
#     img = torch.randn(2, 1, 64, 128, 128)  # Example 3D volume input
#     vit3d = VisionTransformer3D(in_channels=1, img_size=(64, 128, 128))
#     out = vit3d(img)
#     print(out.shape)  # (batch_size, num_classes)
