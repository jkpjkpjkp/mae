from functools import partial

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import Block

class PatchEmbed1D(nn.Module):
    """ 1D sequence to Patch Embedding
    """
    def __init__(self, seq_len=12288, patch_size=64, in_chans=1, embed_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(patch_size * in_chans, embed_dim)

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.seq_len, f"Input sequence length ({L}) doesn't match model ({self.seq_len})."
        
        # Reshape to patches: (B, C, L) -> (B, num_patches, patch_size * C)
        x = x.reshape(B, C, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 3, 1)  # (B, num_patches, patch_size, C)
        x = x.reshape(B, self.num_patches, self.patch_size * C)
        x = self.proj(x)
        return x


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    Create 1D sinusoidal position embedding.
    
    Args:
        embed_dim: embedding dimension
        length: length of the sequence (number of patches)
        cls_token: whether to include cls token
    
    Returns:
        pos_embed: [length, embed_dim] or [length+1, embed_dim] if cls_token
    """
    pos = np.arange(length, dtype=np.float32)
    
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (embed_dim/2,)
    
    pos = pos.reshape(-1)  # (length,)
    pos = np.outer(pos, omega)  # (length, embed_dim/2), outer product
    
    emb_sin = np.sin(pos)  # (length, embed_dim/2)
    emb_cos = np.cos(pos)  # (length, embed_dim/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (length, embed_dim)
    
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


class MaskedAutoencoder1D(nn.Module):
    def __init__(self, seq_len=12288, patch_size=64, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=None, norm_loss=False):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(seq_len, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_loss = norm_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed projection like nn.Linear
        torch.nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.constant_(self.patch_embed.proj.bias, 0)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """
        x: (N, C, L) where L is sequence length
        patches: (N, num_patches, patch_size * C)
        """
        B, C, L = x.shape
        patch_size = self.patch_embed.patch_size
        num_patches = L // patch_size
        
        x = x.reshape(B, C, num_patches, patch_size)
        x = x.permute(0, 2, 3, 1)  # (B, num_patches, patch_size, C)
        x = x.reshape(B, num_patches, patch_size * C)
        return x

    def unpatchify(self, x):
        """
        x: (N, num_patches, patch_size * C)
        sequence: (N, C, L)
        """
        B, num_patches, patch_dim = x.shape
        patch_size = self.patch_embed.patch_size
        C = patch_dim // patch_size
        
        x = x.reshape(B, num_patches, patch_size, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, num_patches, patch_size)
        sequence = x.reshape(B, C, num_patches * patch_size)
        return sequence

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, sequences, pred, mask):
        """
        sequences: [N, C, L]
        pred: [N, num_patches, patch_size*C]
        mask: [N, num_patches], 0 is keep, 1 is remove, 
        """
        target = self.patchify(sequences)
        if self.norm_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, num_patches], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, sequences, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(sequences, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, num_patches, patch_size*C]
        loss = self.forward_loss(sequences, pred, mask)
        return loss, pred, mask


def mae_1d_tiny_patch64(**kwargs):
    model = MaskedAutoencoder1D(
        seq_len=12288, patch_size=64, in_chans=1, embed_dim=384, depth=6, num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_1d_small_patch64(**kwargs):
    model = MaskedAutoencoder1D(
        seq_len=12288, patch_size=64, in_chans=1, embed_dim=512, depth=8, num_heads=8,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_1d_tiny_patch64_short(**kwargs):
    """For nside=16, seq_len=3072, 48 patches"""
    model = MaskedAutoencoder1D(
        seq_len=3072, patch_size=64, in_chans=1, embed_dim=384, depth=6, num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_1d_small_patch64_short(**kwargs):
    """For nside=16, seq_len=3072, 48 patches"""
    model = MaskedAutoencoder1D(
        seq_len=3072, patch_size=64, in_chans=1, embed_dim=512, depth=8, num_heads=8,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_1d_base_patch16(**kwargs):
    model = MaskedAutoencoder1D(
        seq_len=12288, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_1d_base_patch32(**kwargs):
    model = MaskedAutoencoder1D(
        seq_len=12288, patch_size=32, in_chans=1, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_1d_base_patch64(**kwargs):
    model = MaskedAutoencoder1D(
        seq_len=12288, patch_size=64, in_chans=1, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_1d_large_patch16(**kwargs):
    model = MaskedAutoencoder1D(
        seq_len=12288, patch_size=16, in_chans=1, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_1d_large_patch32(**kwargs):
    model = MaskedAutoencoder1D(
        seq_len=12288, patch_size=32, in_chans=1, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_1d_large_patch64(**kwargs):
    model = MaskedAutoencoder1D(
        seq_len=12288, patch_size=64, in_chans=1, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_1d_tiny_patch64 = mae_1d_tiny_patch64   # 192 patches of size 64, ~5M params
mae_1d_small_patch64 = mae_1d_small_patch64 # 192 patches of size 64, ~15M params
mae_1d_base_patch16 = mae_1d_base_patch16  # 768 patches of size 16
mae_1d_base_patch32 = mae_1d_base_patch32  # 384 patches of size 32  
mae_1d_base_patch64 = mae_1d_base_patch64  # 192 patches of size 64
mae_1d_large_patch16 = mae_1d_large_patch16  # 768 patches of size 16
mae_1d_large_patch32 = mae_1d_large_patch32  # 384 patches of size 32
mae_1d_large_patch64 = mae_1d_large_patch64  # 192 patches of size 64