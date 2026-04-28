from typing import Any, Dict, List, Tuple, Union, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .image_encoder_vssam import ImageEncoderViT3D as VSmixSAMEncoder
from .mask_decoder_vssam import MaskDecoder3D as MaskDecoderVSM

try:
    from .prompt_encoder_vssam import PromptEncoder3D
except ImportError:
    class PromptEncoder3D(nn.Module):
        def __init__(self, embed_dim, image_embedding_size, input_image_size, mask_in_chans):
            super().__init__()
            self.embed_dim = embed_dim
        def forward(self, points, boxes, masks):
            return torch.randn(1, 0, self.embed_dim), torch.randn(1, self.embed_dim, 12, 12, 12)
        def get_dense_pe(self):
            return torch.randn(1, self.embed_dim, 12, 12, 12)

class Sam3D(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "L"

    def __init__(
        self,
        image_encoder: Union[VSmixSAMEncoder, nn.Module],
        prompt_encoder: PromptEncoder3D,
        mask_decoder: Union[MaskDecoderVSM, nn.Module],
        pixel_mean: List[float] = [0],
        pixel_std: List[float] = [1],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1, 1), persistent=False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def _encoder_input_size_3d(self) -> Tuple[int, int, int]:
        s = getattr(self.image_encoder, "img_size", None)
        if s is None:
            pass
        if isinstance(s, int):
            return (s, s, s)
        if isinstance(s, (tuple, list)) and len(s) == 3:
            return int(s[0]), int(s[1]), int(s[2])
        raise ValueError(f"Unsupported image_encoder.img_size={s}")

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0).to(self.device)
        enc_outputs = self.image_encoder(input_images)
        feats_multi: Optional[List[torch.Tensor]] = None
        image_embeddings_single: Optional[torch.Tensor] = None
        if isinstance(enc_outputs, list):
            feats_multi = enc_outputs
        else:
            image_embeddings_single = enc_outputs
        outputs: List[Dict[str, torch.Tensor]] = []
        pe = self.prompt_encoder.get_dense_pe().to(self.device)
        B = len(batched_input)
        for i in range(B):
            image_record = batched_input[i]
            pc_raw = image_record.get("point_coords")
            pl_raw = image_record.get("point_labels")
            if pc_raw is not None and pl_raw is not None:
                pc = pc_raw.to(self.device, dtype=torch.float32)
                pl = pl_raw.to(self.device, dtype=torch.int64)
                if pc.dim() == 2:
                    pc = pc.unsqueeze(0)
                if pl.dim() == 1:
                    pl = pl.unsqueeze(0)
                points = (pc, pl)
            else:
                points = None
            boxes_raw = image_record.get("boxes")
            if boxes_raw is not None:
                boxes = boxes_raw.to(self.device, dtype=torch.float32)
                if boxes.dim() == 3 and boxes.shape[0] == 1:
                    boxes = boxes.squeeze(0)
            else:
                boxes = None
            mask_raw = image_record.get("mask_inputs")
            if mask_raw is not None:
                mask_inputs = mask_raw.to(self.device, dtype=torch.float32)
            else:
                mask_inputs = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_inputs,
            )
            if feats_multi is not None:
                curr_embeddings: List[torch.Tensor] = [lvl_feat[i : i + 1] for lvl_feat in feats_multi]
                decoder_image_embeddings = curr_embeddings
            else:
                curr_embedding = image_embeddings_single[i].unsqueeze(0)
                decoder_image_embeddings = curr_embedding
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=decoder_image_embeddings,
                image_pe=pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=tuple(image_record["image"].shape[-3:]),
                original_size=tuple(image_record["original_size"]),
            )
            if self.mask_threshold > 0.0:
                masks = masks > self.mask_threshold
            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            })
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        target_dhw = self._encoder_input_size_3d()
        masks = F.interpolate(masks, size=target_dhw, mode="trilinear", align_corners=False)
        masks = masks[..., :input_size[0], :input_size[1], :input_size[2]]
        masks = F.interpolate(masks, size=original_size, mode="trilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        d, h, w = x.shape[-3:]
        td, th, tw = self._encoder_input_size_3d()
        padd = max(td - d, 0)
        padh = max(th - h, 0)
        padw = max(tw - w, 0)
        x = F.pad(x, (0, padw, 0, padh, 0, padd))
        return x


def build_sam3d_vsm(
    checkpoint: Optional[str] = None,
    img_size: int = 128,
    in_chans: int = 4,
    feature_size: int = 48,
    out_chans: int = 256,
) -> Sam3D:
    image_encoder = VSmixSAMEncoder(
        img_size=(img_size, img_size, img_size),
        in_chans=in_chans,
        feature_size=feature_size,
        patch_size=(2, 2, 2),
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        out_chans=out_chans,
    )
    prompt_encoder_embed_dim = 256
    prompt_encoder = PromptEncoder3D(
        embed_dim=prompt_encoder_embed_dim,
        image_embedding_size=(img_size // 8, img_size // 8, img_size // 8),
        input_image_size=(img_size, img_size, img_size),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoderVSM(
        transformer_dim=out_chans,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    model = Sam3D(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        pixel_mean=[0.0] * in_chans,
        pixel_std=[1.0] * in_chans,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict, strict=False)
    return model