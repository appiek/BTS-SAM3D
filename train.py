from tqdm import tqdm
import os
import math
import json
import time
import argparse
import datetime
from typing import List, Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from dataloader import TrainingDataset3D, TestingDataset3D, collate_fn_3d,BraTS3DDataset
from segment_anything import build_sam3D_vit_b, build_sam3D_vit_l, build_sam3D_vit_h, sam_model_registry3D,build_sam3D_vit_l_ori
from segment_anything import build_sam_3d_vsmix
from utils.utils import get_3d_boxes_from_mask

def get_model(model_name: str, checkpoint: Optional[str] = None, image_size: Tuple[int, int, int] = (128, 128, 128)):
    builders = {
        "vit_b": build_sam3D_vit_b,
        "vit_l": build_sam3D_vit_l,
        "vit_h": build_sam3D_vit_h,
        "vsmix": build_sam_3d_vsmix,
        "vit_l_ori": build_sam3D_vit_l_ori,
        "default": build_sam3D_vit_b,
    }
    if model_name not in builders:
        print(f"[WARN] Unknown model_name={model_name}, fallback to vit_b")
        model_name = "vit_b"

    model = builders[model_name](checkpoint=checkpoint)

    if hasattr(model, "register_buffer"):
        if model.pixel_mean.numel() != 4:
            model.register_buffer("pixel_mean", torch.zeros(4, 1, 1, 1), persistent=False)
        else:
            model.pixel_mean[:] = 0.0
        if model.pixel_std.numel() != 4:
            model.register_buffer("pixel_std", torch.ones(4, 1, 1, 1), persistent=False)
        else:
            model.pixel_std[:] = 1.0
    return model


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight is not None:
        pred = pred * weight
        target = target * weight
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return 1.0 - dice_coeff(probs, target, eps=eps, weight=weight)


def compute_iou_3d(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred_bin & target_bin).float().sum(dim=(2, 3, 4))
    union = (pred_bin | target_bin).float().sum(dim=(2, 3, 4))
    iou = (inter + eps) / (union + eps)
    return iou.squeeze(1)


def split_brats_masks(labels: torch.Tensor):
    label_3d = labels.squeeze(1)
    wt_mask = (label_3d > 0).float().unsqueeze(1)
    tc_mask = ((label_3d == 1) | (label_3d == 4)).float().unsqueeze(1)
    et_mask = (label_3d == 4).float().unsqueeze(1)
    core_mask = (label_3d == 1).float().unsqueeze(1)
    return wt_mask, tc_mask, et_mask, core_mask


def build_prompt_batch(
    images: torch.Tensor,
    original_sizes,
    boxes=None,
    point_coords=None,
    point_labels=None,
    mask_inputs=None,
):
    B = images.shape[0]
    batched = []
    for i in range(B):
        batched.append(
            {
                "image": images[i],
                "original_size": tuple(original_sizes[i].tolist()) if isinstance(original_sizes, torch.Tensor) else tuple(original_sizes[i]),
                "boxes": None if boxes is None else boxes[i],
                "point_coords": None if point_coords is None else point_coords[i],
                "point_labels": None if point_labels is None else point_labels[i],
                "mask_inputs": None if mask_inputs is None else mask_inputs[i],
            }
        )
    return batched


def upsample_and_loss(low_res_logits: torch.Tensor, target_1ch: torch.Tensor, bce_loss_fn: nn.Module, args):
    target_float = target_1ch.unsqueeze(0).float()
    logits_up = F.interpolate(low_res_logits, size=target_1ch.shape[-3:], mode="trilinear", align_corners=False)
    bce = bce_loss_fn(logits_up, target_float)
    dice = dice_loss_from_logits(logits_up, target_float)
    return args.lambda_bce * bce + args.lambda_dice * dice, logits_up


def sample_positive_point(mask_1ch: torch.Tensor, device: torch.device):
    coords = torch.nonzero(mask_1ch > 0.5, as_tuple=False)
    if coords.numel() == 0:
        d, h, w = mask_1ch.shape
        pt = torch.tensor([w // 2, h // 2, d // 2], device=device, dtype=torch.float32)
    else:
        idx = torch.randint(0, coords.shape[0], (1,), device=device)
        z, y, x = coords[idx].squeeze(0)
        pt = torch.stack([x.float(), y.float(), z.float()])
    return pt.unsqueeze(0)


def build_batched_input_from_loader_batch(batch: Dict[str, Union[torch.Tensor, List]]) -> List[Dict[str, torch.Tensor]]:
    B = batch["image"].shape[0]
    out: List[Dict[str, torch.Tensor]] = []
    for i in range(B):
        sample: Dict[str, torch.Tensor] = {
            "image": batch["image"][i],
            "original_size": tuple(batch["original_sizes"][i].tolist()) if isinstance(batch["original_sizes"], torch.Tensor)
                             else tuple(batch["original_sizes"][i]),
            "point_coords": batch["point_coords"][i],
            "point_labels": batch["point_labels"][i],
            "boxes": batch["boxes"][i],
        }
        out.append(sample)
    return out




def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_samples = 0

    bce_loss_fn = nn.BCEWithLogitsLoss()
    stage_loss_meter = {"wt": 0.0, "tc_box": 0.0, "et_box": 0.0, "et_refine": 0.0}

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, ncols=100)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        B = images.shape[0]

        wt_mask, tc_mask, et_mask, core_mask = split_brats_masks(labels)

        wt_boxes = [get_3d_boxes_from_mask(wt_mask[i, 0]).to(device) for i in range(B)]
        batched_wt = build_prompt_batch(images, batch["original_sizes"], boxes=wt_boxes)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            outputs_wt = model(batched_wt, multimask_output=False)
            loss_wt = 0.0
            for i, out in enumerate(outputs_wt):
                l, _ = upsample_and_loss(out["low_res_logits"].to(device), wt_mask[i], bce_loss_fn, args)
                loss_wt += l
            loss_wt = loss_wt / len(outputs_wt)

        if args.amp:
            scaler.scale(loss_wt).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_wt.backward()
            optimizer.step()

        tc_boxes = [get_3d_boxes_from_mask(tc_mask[i, 0]).to(device) for i in range(B)]
        batched_tc_box = build_prompt_batch(images, batch["original_sizes"], boxes=tc_boxes)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            outputs_tc = model(batched_tc_box, multimask_output=False)
            loss_tc_box = 0.0
            prev_low_res_tc = []
            prev_logits_up = []
            for i, out in enumerate(outputs_tc):
                l, logits_up_tc = upsample_and_loss(out["low_res_logits"].to(device), tc_mask[i], bce_loss_fn, args)
                loss_tc_box += l
                prev_low_res_tc.append(out["low_res_logits"].detach())
                prev_logits_up.append(logits_up_tc.detach())
            loss_tc_box = loss_tc_box / len(outputs_tc)

        if args.amp:
            scaler.scale(loss_tc_box).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_tc_box.backward()
            optimizer.step()

        et_boxes = [get_3d_boxes_from_mask(et_mask[i, 0]).to(device) for i in range(B)]

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            batched_et_box = build_prompt_batch(
                images,
                batch["original_sizes"],
                boxes=et_boxes,
                mask_inputs=prev_low_res_tc,
            )
            outputs_et_box = model(batched_et_box, multimask_output=False)

            loss_et_box = 0.0
            point_coords_list = []
            point_labels_list = []
            for i, out in enumerate(outputs_et_box):
                l_et_box, logits_up_et = upsample_and_loss(out["low_res_logits"].to(device), et_mask[i], bce_loss_fn, args)
                loss_et_box += l_et_box

                pts, lbs = sample_points_from_error(logits_up_et.detach(), et_mask[i].unsqueeze(0))
                point_coords_list.append(pts.squeeze(0))
                point_labels_list.append(lbs.squeeze(0))

            loss_et_box = loss_et_box / len(outputs_et_box)

        if args.amp:
            scaler.scale(loss_et_box).backward()
        else:
            loss_et_box.backward()

        del outputs_et_box

        with autocast(enabled=args.amp):
            batched_et_refine = build_prompt_batch(
                images,
                batch["original_sizes"],
                boxes=et_boxes,
                point_coords=point_coords_list,
                point_labels=point_labels_list,
                mask_inputs=prev_low_res_tc,
            )
            outputs_et_refine = model(batched_et_refine, multimask_output=False)

            loss_et_refine = 0.0
            for i, out in enumerate(outputs_et_refine):
                l_et_refine, _ = upsample_and_loss(out["low_res_logits"].to(device), et_mask[i], bce_loss_fn, args)
                loss_et_refine += l_et_refine
            loss_et_refine = loss_et_refine / len(outputs_et_refine)

        if args.amp:
            scaler.scale(loss_et_refine).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_et_refine.backward()
            optimizer.step()

        del outputs_et_refine

        stage_loss_meter["wt"] += loss_wt.item() * B
        stage_loss_meter["tc_box"] += loss_tc_box.item() * B
        stage_loss_meter["et_box"] += loss_et_box.item() * B
        stage_loss_meter["et_refine"] += loss_et_refine.item() * B
        total_loss += (loss_wt.item() + loss_tc_box.item() + loss_et_box.item() + loss_et_refine.item()) * B
        num_samples += B

    return {
        "loss": total_loss / max(1, num_samples),
        "bce": stage_loss_meter["wt"] / max(1, num_samples),
        "dice": stage_loss_meter["tc_box"] / max(1, num_samples),
        "necrosis": stage_loss_meter["et_box"] / max(1, num_samples),
        "iou_reg": stage_loss_meter["et_refine"] / max(1, num_samples),
    }




def sample_points_from_error(pred_logits, target_mask, max_points=1):
    with torch.no_grad():
        pred_prob = torch.sigmoid(pred_logits)
        pred_binary = (pred_prob > 0.5).float()

        fn_mask = (target_mask == 1) & (pred_binary == 0)
        fp_mask = (target_mask == 0) & (pred_binary == 1)

        points = []
        labels = []

        device = pred_logits.device

        def get_random_point(mask_5d):
            mask_3d = mask_5d[0, 0]
            coords = torch.nonzero(mask_3d, as_tuple=False)
            if len(coords) > 0:
                idx = torch.randint(0, len(coords), (1,)).item()
                z, y, x = coords[idx].tolist()
                return torch.tensor([x, y, z], device=device, dtype=torch.float32)
            return None

        for _ in range(max_points):
            if fn_mask.sum() > 0:
                pt = get_random_point(fn_mask)
                if pt is not None:
                    points.append(pt)
                    labels.append(1)

            elif fp_mask.sum() > 0:
                pt = get_random_point(fp_mask)
                if pt is not None:
                    points.append(pt)
                    labels.append(0)

            else:
                pt = get_random_point(target_mask)
                if pt is not None:
                    points.append(pt)
                    labels.append(1)
                else:
                    d, h, w = pred_logits.shape[-3:]
                    points.append(torch.tensor([w // 2, h // 2, d // 2], device=device, dtype=torch.float32))
                    labels.append(0)

        if len(points) == 0:
            d, h, w = pred_logits.shape[-3:]
            points.append(torch.tensor([w // 2, h // 2, d // 2], device=device, dtype=torch.float32))
            labels.append(0)

        points = torch.stack(points).unsqueeze(0)
        labels = torch.tensor(labels, device=device).unsqueeze(0)

        return points, labels



@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args) -> Dict[str, float]:
    model.eval()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    dice_wt, dice_core, dice_ncr, dice_et = [], [], [], []
    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        B = images.shape[0]

        wt_mask, tc_mask, et_mask, core_mask = split_brats_masks(labels)

        wt_boxes = [get_3d_boxes_from_mask(wt_mask[i, 0]).to(device) for i in range(B)]
        batched_wt = build_prompt_batch(images, batch["original_sizes"], boxes=wt_boxes)
        outputs_wt = model(batched_wt, multimask_output=False)

        tc_boxes = [get_3d_boxes_from_mask(tc_mask[i, 0]).to(device) for i in range(B)]
        batched_tc_box = build_prompt_batch(images, batch["original_sizes"], boxes=tc_boxes)
        outputs_tc_box = model(batched_tc_box, multimask_output=False)

        prev_low_res_tc = []
        prev_logits_up = []
        for i, out in enumerate(outputs_tc_box):
            logits_up_tc = F.interpolate(
                out["low_res_logits"].to(device),
                size=tc_mask.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            prev_low_res_tc.append(out["low_res_logits"].detach())
            prev_logits_up.append(logits_up_tc.detach())

        et_boxes = [get_3d_boxes_from_mask(et_mask[i, 0]).to(device) for i in range(B)]

        batched_et_box = build_prompt_batch(
            images,
            batch["original_sizes"],
            boxes=et_boxes,
            mask_inputs=prev_low_res_tc,
        )
        outputs_et_box = model(batched_et_box, multimask_output=False)

        point_coords_list = []
        point_labels_list = []
        logits_up_box = []
        for i, out in enumerate(outputs_et_box):
            logits_et_up = F.interpolate(
                out["low_res_logits"].to(device),
                size=et_mask.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            logits_up_box.append(logits_et_up.detach())
            et_gt = et_mask[i].unsqueeze(0).float()
            dice_et_box_i = dice_coeff((logits_et_up.sigmoid() > 0.5).float(), et_gt)
            dice_ncr.append(dice_et_box_i.item())
            pts, lbs = sample_points_from_error(logits_et_up.detach(), et_mask[i].unsqueeze(0))
            point_coords_list.append(pts.squeeze(0))
            point_labels_list.append(lbs.squeeze(0))

        batched_et_refine = build_prompt_batch(
            images,
            batch["original_sizes"],
            boxes=et_boxes,
            point_coords=point_coords_list,
            point_labels=point_labels_list,
            mask_inputs=prev_low_res_tc,
        )
        outputs_et_refine = model(batched_et_refine, multimask_output=False)

        batch_loss = 0.0

        for i in range(B):
            logits_wt = outputs_wt[i]["low_res_logits"].to(device)
            logits_wt_up = F.interpolate(logits_wt, size=wt_mask.shape[-3:], mode="trilinear", align_corners=False)
            wt_gt = wt_mask[i].unsqueeze(0).float()
            bce_wt = bce_loss_fn(logits_wt_up, wt_gt)
            dice_wt_i = dice_coeff((logits_wt_up.sigmoid() > 0.5).float(), wt_gt)
            dice_wt.append(dice_wt_i.item())
            batch_loss += args.lambda_bce * bce_wt + args.lambda_dice * (1 - dice_wt_i)

            logits_core = prev_logits_up[i]
            core_gt = tc_mask[i].unsqueeze(0).float()
            dice_core_i = dice_coeff((logits_core.sigmoid() > 0.5).float(), core_gt)
            dice_core.append(dice_core_i.item())

            logits_et = outputs_et_refine[i]["low_res_logits"].to(device)
            logits_et_up = F.interpolate(logits_et, size=et_mask.shape[-3:], mode="trilinear", align_corners=False)
            et_gt = et_mask[i].unsqueeze(0).float()
            bce_et = bce_loss_fn(logits_et_up, et_gt)
            dice_et_i = dice_coeff((logits_et_up.sigmoid() > 0.5).float(), et_gt)
            dice_et.append(dice_et_i.item())
            batch_loss += args.lambda_bce * bce_et + args.lambda_dice * (1 - dice_et_i)

        total_loss += batch_loss.item()
        num_samples += B

    return {
        "val_loss": float(total_loss / max(1, num_samples)),
        "val_dice_wt": float(sum(dice_wt) / max(1, len(dice_wt))),
        "val_dice_core": float(sum(dice_core) / max(1, len(dice_core))),
        "val_dice_ncr": float(sum(dice_ncr) / max(1, len(dice_ncr))),
        "val_dice_et": float(sum(dice_et) / max(1, len(dice_et))),
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, out_dir: str, tag: str, val_dice: float = None):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_dice": val_dice if val_dice is not None else 0.0,
    }

    if val_dice is not None:
        dice_str = f"{val_dice:.4f}".replace('.', 'p')
        path = os.path.join(out_dir, f"sam3d_{tag}_epoch{epoch}_dice{dice_str}.pt")
    else:
        path = os.path.join(out_dir, f"sam3d_{tag}_epoch{epoch}.pt")

    torch.save(ckpt, path)
    print(f"[CKPT] Saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM-3D on BraTS")
    parser.add_argument("--device", type=str, default='cuda:0', help="使用GPU")
    parser.add_argument("--data_path", type=str, default="BraTS2021_3D_multimodal/case0", help="Path to BraTS dataset root (with dataset.json)")
    parser.add_argument("--save_dir", type=str, default="./workdir/models/brats2021/glioma-unet-ZAxisAttention", help="Checkpoint output directory")
    parser.add_argument("--log_dir", type=str, default="./workdir/logs/brats2021/glioma-unet-ZAxisAttention", help="Log output directory")
    parser.add_argument("--model", type=str, default="vsmix", choices=["vit_b", "vit_l", "vit_h", "vsmix", "vit_l_ori"], help="Backbone size")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional initial checkpoint to load")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-GPU batch size (3D often memory heavy)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--image_size", type=int, nargs=3, default=[128, 128, 128], help="Input size (D H W)")
    parser.add_argument("--point_num", type=int, default=3, help="Number of points per sample in dataset")
    parser.add_argument("--lambda_bce", type=float, default=1.0)
    parser.add_argument("--lambda_dice", type=float, default=1.0)
    parser.add_argument("--lambda_iou", type=float, default=0.1)
    parser.add_argument("--weighted_bce", action="store_true", help="Use weight map if provided by dataset")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("【错误】未检测到 GPU！请检查 CUDA 配置或显卡状态。")
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError(f"【错误】你要求使用 {args.device}，但 PyTorch 找不到 CUDA 设备。")

    print(f"【成功】正在使用设备: {torch.cuda.get_device_name(0)}")
    
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"train_log_{args.model}_{timestamp}.log")
    
    with open(log_file, "w") as f:
        f.write(f"Training Started at: {timestamp}\n")
        f.write("="*50 + "\n")
        f.write("Args:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write("="*50 + "\n\n")
        f.write("Epoch,Train_Loss,Val_Loss,Val_Dice_WT,Val_Dice_Core,Val_Dice_NCR,Val_Dice_ET,Time(s)\n")

    print(f"Logging training to: {log_file}")

    train_set = TrainingDataset3D(
        data_path=args.data_path,
        image_size=tuple(args.image_size),
        mode="train",
        requires_name=True,
        point_num=args.point_num,
        mask_num=1,
    )
    val_set = BraTS3DDataset(
        data_path=args.data_path,
        image_size=tuple(args.image_size),
        mode="val",
        requires_name=True,
        point_num=args.point_num,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_3d,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_fn_3d,
        drop_last=False,
    )

    model = get_model(args.model, checkpoint=args.checkpoint, image_size=tuple(args.image_size))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )

    def lr_lambda(current_step):
        warmup_steps = max(10, len(train_loader))
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, args.epochs * len(train_loader) - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler(enabled=args.amp)

    best_val = -1.0
    global_step = 0
    total_epochs = args.epochs

    with tqdm(total=total_epochs, desc="Training Progress", unit="epoch") as pbar:
        for epoch in range(1, total_epochs + 1):
            t0 = time.time()
            train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
            scheduler.step()
            dt = time.time() - t0

            val_stats = {
                "val_loss": 0.0,
                "val_dice_wt": 0.0,
                "val_dice_core": 0.0,
                "val_dice_ncr": 0.0,
                "val_dice_et": 0.0,
            }
            did_validate = False

            if (epoch % args.val_every) == 0:
                val_stats = evaluate(model, val_loader, device, args)
                did_validate = True

                print(
                    f"\n[Epoch {epoch}] VAL "
                    f"WT={val_stats['val_dice_wt']:.4f} "
                    f"Core={val_stats['val_dice_core']:.4f} "
                    f"NCR={val_stats['val_dice_ncr']:.4f} "
                    f"ET={val_stats['val_dice_et']:.4f} "
                    f"| loss={val_stats['val_loss']:.4f}"
                )

                if val_stats["val_dice_wt"] > best_val:
                    best_val = val_stats["val_dice_wt"]
                    save_checkpoint(model, optimizer, epoch, args.save_dir, tag="best", val_dice=val_stats["val_dice_wt"])

            with open(log_file, "a") as f:
                if did_validate:
                    log_line = (
                        f"{epoch},{train_stats['loss']:.4f},"
                        f"{val_stats['val_loss']:.4f},{val_stats['val_dice_wt']:.4f},"
                        f"{val_stats['val_dice_core']:.4f},{val_stats['val_dice_ncr']:.4f},"
                        f"{val_stats['val_dice_et']:.4f},{dt:.1f}\n"
                    )
                else:
                    log_line = (
                        f"{epoch},{train_stats['loss']:.4f},"
                        f",,,,,"
                        f"{dt:.1f}\n"
                    )
                f.write(log_line)

            pbar.set_postfix(
                loss=f"{train_stats['loss']:.4f}",
                dice_wt=f"{val_stats['val_dice_wt']:.4f}" if did_validate else "-",
                dice_core=f"{val_stats['val_dice_core']:.4f}" if did_validate else "-",
                dice_ncr=f"{val_stats['val_dice_ncr']:.4f}" if did_validate else "-",
                dice_et=f"{val_stats['val_dice_et']:.4f}" if did_validate else "-",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            pbar.update(1)
            if epoch % 10 == 0 or epoch == total_epochs:
                save_checkpoint(model, optimizer, epoch, args.save_dir, tag="last",
                                val_dice=val_stats["val_dice_wt"] if did_validate else 0.0)


    print("\nTraining finished.")
    print(f"Best validation Dice: {best_val:.4f}")
    print(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    main()
