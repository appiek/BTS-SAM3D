from .modeling.prompt_encoder_vssam import PromptEncoder3D

from .modeling.image_encoder_vssam import ImageEncoderViT3D
from .modeling.mask_decoder_vssam import MaskDecoder3D

from .modeling.sam_vsm import Sam3D
from typing import Tuple
import torch


def build_sam_3d_vsmix(
    checkpoint=None,
):

    return build_sam_vssam(
        encoder_embed_dim=48,
        encoder_depths=[2, 2, 2, 2],
        encoder_num_heads=[3, 6, 12, 24],
        encoder_window_size=7,
        encoder_feature_channels=[48, 96, 192, 384],
        prompt_embed_dim=256,
        image_size=128,
        pixel_mean=[0],
        pixel_std=[1],
        checkpoint=checkpoint,
    )


def build_sam_vssam(
    encoder_embed_dim,
    encoder_depths,
    encoder_num_heads,
    encoder_window_size,
    encoder_feature_channels,
    prompt_embed_dim,
    image_size,
    pixel_mean,
    pixel_std,
    checkpoint,
):
    image_encoder = ImageEncoderViT3D(
        img_size=(image_size, image_size, image_size),
        feature_size=encoder_embed_dim,
        vsmix_depths=encoder_depths,
        vsmix_num_heads=encoder_num_heads,
        window_size=encoder_window_size,
        out_chans=prompt_embed_dim,
    )

    fusion_scale = 8
    prompt_encoder = PromptEncoder3D(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_size // fusion_scale, image_size // fusion_scale, image_size // fusion_scale),
        input_image_size=(image_size, image_size, image_size),
        mask_in_chans=16,
    )

    mask_decoder = MaskDecoder3D(
        num_multimask_outputs=3,
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        num_feature_levels=4,
        feature_channels=encoder_feature_channels,
    )

    sam = Sam3D(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, strict=False)

    return sam


