import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.utils import get_3d_boxes_from_mask
from dataloader import TestingDataset3D, collate_fn_3d, BraTS3DDataset
from segment_anything import build_sam3D_vit_b, build_sam3D_vit_l, build_sam3D_vit_h
from segment_anything import build_sam_3d_vsmix


def get_model(model_name: str, checkpoint: str, image_size: Tuple[int, int, int]):
    builders = {
        "vit_b": build_sam3D_vit_b,
        "vit_l": build_sam3D_vit_l,
        "vit_h": build_sam3D_vit_h,
        "vsmix": build_sam_3d_vsmix,
    }
    print(f"[INFO] Building model: {model_name}...")
    model = builders.get(model_name, build_sam3D_vit_h)(checkpoint=None)
    if hasattr(model, "register_buffer"):
        if model.pixel_mean.numel() != 4:
            model.register_buffer("pixel_mean", torch.zeros(4, 1, 1, 1), persistent=False)
        if model.pixel_std.numel() != 4:
            model.register_buffer("pixel_std", torch.ones(4, 1, 1, 1), persistent=False)
    if checkpoint is not None:
        print(f"[INFO] Loading weights from {checkpoint}")
        ckpt_obj = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt_obj.get("model", ckpt_obj)
        model.load_state_dict(state_dict, strict=True)
    return model


def split_brats_masks(labels: torch.Tensor):
    label_3d = labels.squeeze(1)
    wt_mask = (label_3d > 0).float().unsqueeze(1)
    tc_mask = ((label_3d == 1) | (label_3d == 4)).float().unsqueeze(1)
    et_mask = (label_3d == 4).float().unsqueeze(1)
    return wt_mask, tc_mask, et_mask


def build_prompt_batch(images, original_sizes, boxes=None, point_coords=None, point_labels=None, mask_inputs=None):
    B = images.shape[0]
    batched = []
    for i in range(B):
        batched.append({
            "image": images[i],
            "original_size": tuple(original_sizes[i].tolist()) if torch.is_tensor(original_sizes) else tuple(original_sizes[i]),
            "boxes": None if boxes is None else boxes[i],
            "point_coords": None if point_coords is None else point_coords[i],
            "point_labels": None if point_labels is None else point_labels[i],
            "mask_inputs": None if mask_inputs is None else mask_inputs[i],
        })
    return batched


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


def dice_coeff_numpy(pred, target):
    intersection = np.logical_and(pred, target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)


def save_nifti(mask_data, affine, save_path):
    mask_data = mask_data.astype(np.uint8)
    mask_data = np.transpose(mask_data, (1, 2, 0))
    nifti_img = nib.Nifti1Image(mask_data, affine)
    nib.save(nifti_img, save_path)


def load_full_label_volume(label_path: str) -> np.ndarray:
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata()
    if label_data.ndim != 3:
        raise ValueError(f"Unexpected label shape: {label_data.shape}, path={label_path}")
    return np.transpose(label_data, (2, 0, 1)).astype(np.uint8)


def derive_case_name(data_item: Dict, index: int) -> str:
    label_path = data_item.get("label") if isinstance(data_item, dict) else None
    candidate_path = label_path or (data_item.get("image") if isinstance(data_item, dict) else None)
    if candidate_path:
        base_name = os.path.basename(candidate_path)
        if label_path is None:
            base_name = base_name.replace("_0000", "")
        if not base_name.endswith(".nii.gz"):
            base_root = base_name.split(".")[0]
            base_name = f"{base_root}.nii.gz"
        return base_name
    return f"sample_{index:04d}.nii.gz"


def select_tumor_slice(gt_volume: np.ndarray) -> int:
    tumor_sums = gt_volume.sum(axis=(1, 2))
    if tumor_sums.max() > 0:
        return int(np.argmax(tumor_sums))
    return gt_volume.shape[0] // 2


def normalize_slice(image_slice: np.ndarray) -> np.ndarray:
    slice_copy = image_slice.astype(np.float32, copy=False)
    valid_mask = slice_copy != 0
    if valid_mask.any():
        min_val = slice_copy[valid_mask].min()
        max_val = slice_copy[valid_mask].max()
    else:
        min_val = slice_copy.min()
        max_val = slice_copy.max()
    denom = max(max_val - min_val, 1e-6)
    return (slice_copy - min_val) / denom


def render_visualization(image_volume: np.ndarray,
                         gt_volume: np.ndarray,
                         pred_volume: np.ndarray,
                         slice_idx: int,
                         save_path: str,
                         channel: int = 0) -> None:
    slice_img = image_volume[channel, slice_idx]
    slice_img = normalize_slice(slice_img)
    gt_slice = (gt_volume[slice_idx] > 0).astype(np.float32)
    pred_slice = (pred_volume[slice_idx] > 0).astype(np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(slice_img, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(slice_img, cmap="gray")
    if gt_slice.sum() > 0:
        axes[1].contour(gt_slice, levels=[0.5], colors="lime", linewidths=1.5)
    axes[1].set_title("Image + GT")
    axes[1].axis("off")
    axes[2].imshow(slice_img, cmap="gray")
    if gt_slice.sum() > 0:
        axes[2].contour(gt_slice, levels=[0.5], colors="lime", linewidths=1.2)
    if pred_slice.sum() > 0:
        axes[2].contour(pred_slice, levels=[0.5], colors="red", linewidths=1.2)
    axes[2].set_title("Image + GT + Pred")
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test SAM-3D Interactive Inference")
    parser.add_argument("--data_path", type=str, default="BraTS2021", help="Dataset root")
    parser.add_argument("--checkpoint", type=str, default="./workdir/models/", help="Path to best checkpoint")
    parser.add_argument("--model", type=str, default="vsmix", choices=["vit_b", "vsmix"])
    parser.add_argument("--output_dir", type=str, default="./tetsdir/", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image_size", type=int, nargs=3, default=[128, 128, 128])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model = get_model(args.model, args.checkpoint, tuple(args.image_size))
    model.to(device)
    model.eval()
    test_set = BraTS3DDataset(
        data_path=args.data_path,
        image_size=tuple(args.image_size),
        mode='test',
        requires_name=True,
        point_num=1,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn_3d
    )
    if hasattr(test_set, "data_items") and hasattr(test_set, "valid_indices"):
        output_name_list = [derive_case_name(test_set.data_items[idx], order) for order, idx in enumerate(test_set.valid_indices)]
        label_path_list = []
        for idx in test_set.valid_indices:
            item = test_set.data_items[idx]
            label_rel = item.get("label") if isinstance(item, dict) else None
            label_path_list.append(os.path.join(args.data_path, label_rel) if label_rel else None)
    else:
        output_name_list = [f"sample_{i:04d}.nii.gz" for i in range(len(test_set))]
        label_path_list = [None for _ in range(len(test_set))]
    print(f"[INFO] Testing on {len(test_set)} samples...")
    metrics = {"WT": [], "TC": [], "ET": []}
    sample_counter = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inferencing"):
            images_cpu = batch["image"].cpu().numpy()
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            B = images.shape[0]
            label_path = label_path_list[sample_counter] if sample_counter < len(label_path_list) else None
            if label_path is None or (not os.path.exists(label_path)):
                raise FileNotFoundError(f"Missing original label path for sample index {sample_counter}: {label_path}")
            full_gt_label = load_full_label_volume(label_path)
            orig_shape = full_gt_label.shape
            wt_mask, tc_mask, et_mask = split_brats_masks(labels)
            wt_boxes = [get_3d_boxes_from_mask(wt_mask[i, 0]).to(device) for i in range(B)]
            batched_wt = build_prompt_batch(images, batch["original_sizes"], boxes=wt_boxes)
            outputs_wt = model(batched_wt, multimask_output=False)
            logits_wt = outputs_wt[0]["low_res_logits"]
            tc_boxes = [get_3d_boxes_from_mask(tc_mask[i, 0]).to(device) for i in range(B)]
            batched_tc = build_prompt_batch(images, batch["original_sizes"], boxes=tc_boxes)
            outputs_tc = model(batched_tc, multimask_output=False)
            logits_tc = outputs_tc[0]["low_res_logits"]
            prev_low_res_tc = [logits_tc.detach().clone()]
            logits_et_final = None
            has_et = (et_mask[0].sum() > 0)
            if has_et:
                et_boxes = [get_3d_boxes_from_mask(et_mask[i, 0]).to(device) for i in range(B)]
                batched_et_box = build_prompt_batch(images, batch["original_sizes"], boxes=et_boxes, mask_inputs=prev_low_res_tc)
                outputs_et_box = model(batched_et_box, multimask_output=False)
                logits_et_init = outputs_et_box[0]["low_res_logits"]
                logits_et_init_up = F.interpolate(logits_et_init, size=et_mask.shape[-3:], mode="trilinear", align_corners=False)
                points, labels = sample_points_from_error(logits_et_init_up, et_mask[0].unsqueeze(0), max_points=1)
                point_coords_list = [points.squeeze(0)]
                point_labels_list = [labels.squeeze(0)]
                batched_et_refine = build_prompt_batch(images, batch["original_sizes"], boxes=et_boxes, point_coords=point_coords_list, point_labels=point_labels_list, mask_inputs=prev_low_res_tc)
                outputs_et_refine = model(batched_et_refine, multimask_output=False)
                logits_et_final = outputs_et_refine[0]["low_res_logits"]
            else:
                logits_et_final = torch.zeros_like(logits_tc) - 100.0
            orig_size = batch["original_sizes"][0]
            target_size = (int(orig_size[0]), int(orig_size[1]), int(orig_size[2]))
            mask_wt_small = (torch.sigmoid(F.interpolate(logits_wt, size=target_size, mode="trilinear", align_corners=False)) > 0.5).cpu().numpy()[0, 0]
            mask_tc_small = (torch.sigmoid(F.interpolate(logits_tc, size=target_size, mode="trilinear", align_corners=False)) > 0.5).cpu().numpy()[0, 0]
            mask_et_small = (torch.sigmoid(F.interpolate(logits_et_final, size=target_size, mode="trilinear", align_corners=False)) > 0.5).cpu().numpy()[0, 0]
            mask_tc_small = np.logical_and(mask_tc_small, mask_wt_small)
            mask_et_small = np.logical_and(mask_et_small, mask_tc_small)
            final_label_small = np.zeros_like(mask_wt_small, dtype=np.uint8)
            final_label_small[mask_wt_small] = 2
            final_label_small[mask_tc_small] = 1
            final_label_small[mask_et_small] = 4
            crop_meta = batch.get("crop_info")
            if isinstance(crop_meta, list) and crop_meta:
                crop_info = crop_meta[0]
            elif isinstance(crop_meta, dict):
                crop_info = crop_meta
            else:
                crop_info = {}
            z_start = int(crop_info.get("z_start", 0))
            y_start = int(crop_info.get("y_start", 0))
            x_start = int(crop_info.get("x_start", 0))
            z_size = int(crop_info.get("z_size", final_label_small.shape[0]))
            y_size = int(crop_info.get("y_size", final_label_small.shape[1]))
            x_size = int(crop_info.get("x_size", final_label_small.shape[2]))
            z_size = max(0, min(z_size, final_label_small.shape[0], orig_shape[0] - z_start))
            y_size = max(0, min(y_size, final_label_small.shape[1], orig_shape[1] - y_start))
            x_size = max(0, min(x_size, final_label_small.shape[2], orig_shape[2] - x_start))
            final_label_full = np.zeros(orig_shape, dtype=np.uint8)
            if z_size > 0 and y_size > 0 and x_size > 0:
                final_label_full[z_start:z_start + z_size, y_start:y_start + y_size, x_start:x_start + x_size] = final_label_small[:z_size, :y_size, :x_size]
            gt_wt_full = full_gt_label > 0
            gt_tc_full = np.logical_or(full_gt_label == 1, full_gt_label == 4)
            gt_et_full = full_gt_label == 4
            pred_wt_full = final_label_full > 0
            pred_tc_full = np.logical_or(final_label_full == 1, final_label_full == 4)
            pred_et_full = final_label_full == 4
            metrics["WT"].append(dice_coeff_numpy(pred_wt_full, gt_wt_full))
            metrics["TC"].append(dice_coeff_numpy(pred_tc_full, gt_tc_full))
            if gt_et_full.sum() > 0:
                metrics["ET"].append(dice_coeff_numpy(pred_et_full, gt_et_full))
            else:
                if pred_et_full.sum() == 0:
                    metrics["ET"].append(1.0)
                else:
                    metrics["ET"].append(0.0)
            if sample_counter < len(output_name_list):
                output_name = output_name_list[sample_counter]
            else:
                output_name = f"sample_{sample_counter:04d}.nii.gz"
            vis_prefix = output_name[:-7] if output_name.endswith(".nii.gz") else os.path.splitext(output_name)[0]
            affine = batch["affine"][0]
            if isinstance(affine, torch.Tensor):
                affine = affine.cpu().numpy()
            mask_path = os.path.join(args.output_dir, output_name)
            save_nifti(final_label_full, affine, mask_path)
            image_volume = images_cpu[0]
            pred_volume = final_label_small
            gt_volume = wt_mask[0, 0].cpu().numpy()
            slice_idx = select_tumor_slice(gt_volume)
            vis_path = os.path.join(args.output_dir, f"{vis_prefix}_vis.png")
            render_visualization(image_volume, gt_volume, pred_volume, slice_idx, vis_path, channel=0)
            sample_counter += 1
    print("\n" + "=" * 30)
    print("Interactive Inference Summary (Box + 1 Point Refine for ET)")
    print(f"Mean Dice WT: {np.mean(metrics['WT']):.4f}")
    print(f"Mean Dice TC: {np.mean(metrics['TC']):.4f}")
    print(f"Mean Dice ET: {np.mean(metrics['ET']):.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()