# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Tuple, Type, Union
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualConvBlock3D(nn.Module):
   
    def __init__(self, channels: int, hidden_channels: int = None) -> None:
        super().__init__()
        hidden = hidden_channels or channels
        self.block = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        return self.conv(x)


class UNetBlock3D(nn.Module):
   
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.skip_proj = nn.Conv3d(skip_channels, in_channels, kernel_size=1, bias=False)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x, skip_feat):
        target_size = skip_feat.shape[2:]
        if x.shape[2:] != target_size:
            x_up = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        else:
            x_up = x
        s = self.skip_proj(skip_feat)
        out = x_up + s
        return self.block(out)


class MultiLayerFeatureFusion3D(nn.Module):
    def __init__(self, num_levels: int, in_channels_list: List[int], out_dim: int, target_level_idx: int = 2) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.target_idx = target_level_idx
        self.laterals = nn.ModuleList([
            nn.Conv3d(in_ch, out_dim, kernel_size=1, bias=False)
            for in_ch in in_channels_list
        ])
        reduce = max(out_dim // 4, 16)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(out_dim, reduce, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv3d(reduce, 1, kernel_size=1, bias=True),
            )
            for _ in range(num_levels)
        ])
        self.refine = nn.Sequential(
            nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )

    def forward(self, feats: List[Tensor]) -> Tensor:
        feats_l = [lat(f) for lat, f in zip(self.laterals, feats)]
        target_shape = feats_l[self.target_idx].shape[2:]
        aligned_feats = []
        for feat in feats_l:
            if feat.shape[2:] != target_shape:
                aligned_feats.append(F.interpolate(feat, size=target_shape, mode='trilinear', align_corners=False))
            else:
                aligned_feats.append(feat)
        weights = [gate(f) for gate, f in zip(self.gates, aligned_feats)]
        w = torch.stack(weights, dim=1)
        w = torch.softmax(w, dim=1)
        fused = 0.0
        for i in range(self.num_levels):
            fused = fused + w[:, i] * aligned_feats[i]
        return self.refine(fused)


class MLPBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class TwoWayTransformer3D(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock3D(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                ))
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: Tensor, image_pe: Tensor, point_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        bs, c, x, y, z = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class TwoWayAttentionBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock3D(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MaskDecoder3D(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_feature_levels: int = 4,
        feature_channels: List[int] = [48, 96, 192, 384],
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_feature_levels = num_feature_levels
        self.target_level = 2
        self.multi_layer_fusion = MultiLayerFeatureFusion3D(
            num_levels=num_feature_levels,
            in_channels_list=feature_channels,
            out_dim=transformer_dim,
            target_level_idx=self.target_level,
        )
        self.pre_transformer_refine = ResidualConvBlock3D(transformer_dim, hidden_channels=transformer_dim)
        self.transformer = TwoWayTransformer3D(depth=2, embedding_dim=self.transformer_dim, mlp_dim=2048, num_heads=8)
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.unet_blocks = nn.ModuleList()
        curr_dim = transformer_dim
        for i in range(self.target_level - 1, -1, -1):
            skip_ch = feature_channels[i]
            out_dim = max(curr_dim // 2, 16)
            self.unet_blocks.append(UNetBlock3D(in_channels=curr_dim, skip_channels=skip_ch, out_channels=out_dim))
            curr_dim = out_dim
        self.final_dim = curr_dim
        self.final_upsample = UpsampleBlock(self.final_dim, self.final_dim)
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, self.final_dim, 3)
            for _ in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def _as_channel_first(self, x: Tensor) -> Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor, got shape {x.shape}")
        return x

    def _fuse_image_embeddings(self, image_embeddings: List[Tensor]) -> Tensor:
        if isinstance(image_embeddings, (list, tuple)):
            feats = image_embeddings
            fused = self.multi_layer_fusion(feats)
            return fused
        else:
            raise ValueError("VSmix Encoder should return a list of features.")

    def forward(self, image_embeddings: Union[torch.Tensor, List[torch.Tensor]], image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor, multimask_output: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(image_embeddings=image_embeddings, image_pe=image_pe, sparse_prompt_embeddings=sparse_prompt_embeddings, dense_prompt_embeddings=dense_prompt_embeddings)
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks = masks[:, mask_slice, :, :, :]
        iou_pred = iou_pred[:, mask_slice]
        return masks, iou_pred

    def predict_masks(self, image_embeddings: Union[torch.Tensor, List[torch.Tensor]], image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(image_embeddings, list):
            raise ValueError("MaskDecoder3D now requires a LIST of image embeddings for U-Net connections.")
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = self._fuse_image_embeddings(image_embeddings)
        src = self.pre_transformer_refine(src)
        if dense_prompt_embeddings is not None:
            if dense_prompt_embeddings.shape[2:] != src.shape[2:]:
                dense_prompt_embeddings = F.interpolate(dense_prompt_embeddings, size=src.shape[2:], mode='trilinear', align_corners=False)
        if src.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(src, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        if image_pe.shape[2:] != src.shape[2:]:
            image_pe = F.interpolate(image_pe, size=src.shape[2:], mode='trilinear', align_corners=False)
        if image_pe.shape[0] != tokens.shape[0]:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe
        b, c, x, y, z = src.shape
        hs, src_tokens = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]
        current_feat = src_tokens.transpose(1, 2).view(b, c, x, y, z)
        for i, block in enumerate(self.unet_blocks):
            skip_idx = self.target_level - 1 - i
            skip_feat = image_embeddings[skip_idx]
            if skip_feat.shape[0] != current_feat.shape[0]:
                skip_feat = torch.repeat_interleave(skip_feat, current_feat.shape[0] // skip_feat.shape[0], dim=0)
            current_feat = block(current_feat, skip_feat)
        upscaled_embedding = self.final_upsample(current_feat)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c_up, x_up, y_up, z_up = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c_up, x_up * y_up * z_up)).view(b, -1, x_up, y_up, z_up)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred