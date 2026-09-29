import torch
import torch.nn as nn
import numpy as np
from functools import partial
from timm.models.vision_transformer import Block
from monai.networks.blocks import PatchEmbeddingBlock

def to_3tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 3
        return x
    return (x, x, x)

class MAEViT3D(nn.Module):

    def __init__(self,
                 img_size=192,
                 patch_size=16,
                 in_chans=1,
                 pos_embed_type='learnable',
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 dropout_rate=0.0):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.mc_dropout_enabled = False

        self.patch_embed = PatchEmbeddingBlock(
            in_channels=in_chans,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=embed_dim,
            num_heads=num_heads,
            proj_type='conv',
            pos_embed_type=pos_embed_type,
            dropout_rate=0.0,
            spatial_dims=3
        )

        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  proj_drop=dropout_rate, attn_drop=dropout_rate,
                  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                  proj_drop=dropout_rate, attn_drop=dropout_rate,
                  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3 * in_chans, bias=True)

        self.decoder_logvar = nn.Linear(decoder_embed_dim, patch_size**3 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = to_3tuple(16)[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0

        d = imgs.shape[2] // p
        h = imgs.shape[3] // p
        w = imgs.shape[4] // p

        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], d, p, h, p, w, p))
        x = torch.einsum('ncdphqwr->ndhwpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p**3 * imgs.shape[1]))
        return x

    def unpatchify(self, x):
        p = to_3tuple(16)[0]

        d, h, w = 12, 16, 16
        assert d * h * w == x.shape[1], f"Expected {d*h*w} patches, got {x.shape[1]}"

        c = x.shape[2] // (p**3)

        x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, c))

        x = torch.einsum('ndhwpqrc->ncdphqwr', x)

        imgs = x.reshape(shape=(x.shape[0], c, d * p, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def custom_masking(self, x, mask):
        N, L, D = x.shape
        device = x.device

        mask = mask.to(device).float()
        mask = (mask > 0.5).float()

        len_keep = int((mask == 0).sum(dim=1).min().item())

        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_visible, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, mask_type='random', mask=None):

        x = self.patch_embed(x)

        if mask is not None:
            x, mask, ids_restore = self.custom_masking(x, mask)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.cls_pos_embed
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):

        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        mu = self.decoder_pred(x)
        logvar = self.decoder_logvar(x)

        mu = mu[:, 1:, :]
        logvar = logvar[:, 1:, :]

        return mu, logvar

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_loss_nll(self, imgs, mu, logvar, mask):
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        logvar = torch.clamp(logvar, min=-6.0, max=2.0)
        var = torch.exp(logvar)

        nll = 0.5 * (((target - mu) ** 2) / var + logvar)
        nll = nll.mean(dim=-1)

        loss = (nll * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def enable_mc_dropout(self):
        self.mc_dropout_enabled = True
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_mc_dropout(self):
        self.mc_dropout_enabled = False
        self.eval()

    def forward(self, imgs, mask_ratio=0.75, mask_type='random', mask=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, mask_type, mask=mask)
        mu, logvar = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss_nll(imgs, mu, logvar, mask)
        return loss, mu, logvar, mask

def mae_vit_small_patch16_3d(**kwargs):
    model = MAEViT3D(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_3d(**kwargs):
    model = MAEViT3D(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_3d(**kwargs):
    model = MAEViT3D(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
