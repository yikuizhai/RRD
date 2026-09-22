import os
from typing import List

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import cv2
from PIL import Image

from dataset import get_data_transforms, VisaDataset, MVTecDataset, HDDataset
from resnet import resnet101,wide_resnet50_2, resnet50, resnet34, resnet18
from de_resnet import de_resnet101, de_wide_resnet50_2, de_resnet50, de_resnet34, de_resnet18
from metrics import compute_all_metrics


def cal_anomaly_map(fs_list, ft_list, out_size):
    amap = None
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        cm = 1 - F.cosine_similarity(fs, ft)
        cm = torch.unsqueeze(cm, dim=1)
        cm = F.interpolate(cm, size=out_size, mode='bilinear', align_corners=True)
        if amap is None:
            amap = cm
        else:
            amap = amap + cm
    return amap


def _unpack_batch(batch):
    """Return (img, gt, label) from a batch that may be (img, gt, label) or (img, gt, label, type)."""
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    return batch, None, None


def build_test_loaders(classes: List[str], dataset_name: str, data_root: str, image_size: int, num_workers: int = 4):
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    loaders = []
    for cls in classes:
        root = os.path.join(data_root, cls)
        if dataset_name.lower() == 'visa':
            ds = VisaDataset(root=root, transform=data_transform, gt_transform=gt_transform, phase='test')
        elif dataset_name.lower() == 'mvtec':
            ds = MVTecDataset(root=root, transform=data_transform, gt_transform=gt_transform, phase='test')
        elif dataset_name.lower() == 'hd':
            ds = HDDataset(root=root, transform=data_transform, gt_transform=gt_transform, phase='test')
        else:
            raise ValueError('Unknown dataset_name: ' + dataset_name)
        loaders.append((cls, DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)))
    return loaders


def evaluate_full_metrics(encoder, bn, decoder, dataloader: DataLoader, device: str, top_ratio: float = 0.01):
    bn.eval(); decoder.eval()
    labels = []
    masks = []
    amaps_list = []
    with torch.no_grad():
        for batch in dataloader:
            img, gt, label = _unpack_batch(batch)
            img = img.to(device)
            t_feats = encoder(img)
            s_feats = decoder(bn(t_feats))
            amap = cal_anomaly_map(t_feats, s_feats, img.shape[-1])  # [B,1,H,W]
            amap_np = amap.squeeze(1).cpu().numpy()  # (B,H,W)
            amaps_list.append(amap_np)
            if gt is not None:
                masks.append(gt.squeeze(1).cpu().numpy())  # (B,H,W)
            if label is not None:
                labels.extend(np.array(label).astype(int).reshape(-1).tolist())
    amaps_np = np.concatenate(amaps_list, axis=0) if len(amaps_list) else np.zeros((0,1,1))
    masks_np = np.concatenate(masks, axis=0) if len(masks) else np.zeros((0,1,1))
    return compute_all_metrics(labels, masks_np, amaps_np, top_ratio=top_ratio)


# ---------------- Visualization Helpers ----------------
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def _denorm(img_tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize a single image tensor (3,H,W) back to 0-1."""
    return (img_tensor * _STD.to(img_tensor.device) + _MEAN.to(img_tensor.device)).clamp(0,1)


def save_heatmaps_for_loader(encoder, bn, decoder, loader: DataLoader, device: str, out_dir: str, class_name: str, filename_prefix: str = "", image_size: int = 256):
    """Save visualization in mytest-style: [original | overlay(heatmap) | optional GT] per image.
    Filenames: <prefix>_<idx>_<label>_<origFileName>
    """
    os.makedirs(out_dir, exist_ok=True)
    bn.eval(); decoder.eval()
    saved = 0
    for idx, batch in enumerate(loader):
        img, gt, label = _unpack_batch(batch)  # img: [1,3,H,W]
        img = img.to(device)
        with torch.no_grad():
            t_feats = encoder(img)
            s_feats = decoder(bn(t_feats))
            amap = cal_anomaly_map(t_feats, s_feats, img.shape[-1])  # [1,1,H,W]
        amap_np = amap.squeeze().cpu().numpy()
        # Normalize 0-1 and threshold like mytest
        amax, amin = amap_np.max(), amap_np.min()
        if amax > amin:
            heat = (amap_np - amin) / (amax - amin)
        else:
            heat = np.zeros_like(amap_np)
        heat = np.clip(heat, 0, 1)
        heat = np.where(heat < 0.5, 0, heat)
        heat_u8 = (heat * 255).astype(np.uint8)
        # Resolve image path and read original image
        if hasattr(loader.dataset, 'img_paths') and idx < len(loader.dataset.img_paths):
            image_path = loader.dataset.img_paths[idx]
            _, image_name = os.path.split(image_path)
            orig_img = cv2.imread(image_path)
        else:
            image_name = f'img_{idx:05d}.png'
            # fallback: reconstruct from tensor
            img_den = _denorm(img.squeeze(0)).cpu().numpy().transpose(1,2,0)
            orig_img = (img_den * 255).astype(np.uint8)[:, :, ::-1]  # to BGR
        # Resize original to target size
        orig_img = cv2.resize(orig_img, (image_size, image_size))
        # Colorize heatmap, boost intensity, and blur
        heat_boost = np.clip(heat_u8.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        colored_heat_bgr = cv2.applyColorMap(heat_boost, cv2.COLORMAP_JET)
        colored_heat_bgr = cv2.GaussianBlur(colored_heat_bgr, (5, 5), 0)
        # Overlay with stronger heat contribution (no contour lines)
        overlay = cv2.addWeighted(orig_img, 0.6, colored_heat_bgr, 0.4, 0)
        # Build merge panels: original (BGR)->RGB for saving by PIL
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        panels = [orig_rgb, overlay_rgb]
        # Optional GT panel if available
        if gt is not None:
            try:
                gt_np = gt.squeeze().cpu().numpy().astype(np.float32)
                # Ensure HxW, resize if needed
                if gt_np.ndim == 3:
                    gt_np = gt_np[0]
                if gt_np.shape[:2] != (image_size, image_size):
                    gt_np = cv2.resize(gt_np, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                gt_rgb = (np.repeat(gt_np[..., None], 3, axis=-1) * 255).astype(np.uint8)
                panels.append(gt_rgb)
            except Exception:
                pass
        # Concatenate horizontally and save
        merged = np.concatenate(panels, axis=1).astype(np.uint8)
        label_str = 'normal'
        try:
            # label tensor/list to int
            lbl = int(np.array(label).reshape(-1)[0]) if label is not None else 0
            label_str = 'normal' if lbl == 0 else 'abnormal'
        except Exception:
            pass
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        out_name = f"{prefix}{idx}_{label_str}_{image_name}"
        Image.fromarray(merged).save(os.path.join(out_dir, out_name))
        saved += 1
    return saved


def _strip_profile_keys(state_dict: dict) -> dict:
    """Remove profiling stats (flops/params) that are not part of module weights."""
    invalid_tokens = ('total_ops', 'total_params')
    return {k: v for k, v in state_dict.items() if not any(tok in k for tok in invalid_tokens)}


def evaluate_from_checkpoint(
    classes: List[str],
    dataset_name: str = 'mvtec',
    data_root: str = '',
    ckpt_path: str = '',
    image_size: int = 256,
    num_workers: int = 4,
    save_heatmaps: bool = False,
    vis_root: str = 'visualizations',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loaders = build_test_loaders(classes, dataset_name, data_root, image_size, num_workers)

    # encoder, bn = resnet101(pretrained=True)
    encoder, bn = wide_resnet50_2(pretrained=True)
    # encoder, bn = resnet50(pretrained=True)
    # encoder, bn = resnet34(pretrained=True)
    # encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    # decoder = de_resnet101(pretrained=False).to(device)
    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    # decoder = de_resnet50(pretrained=False).to(device)
    # decoder = de_resnet34(pretrained=False).to(device)
    # decoder = de_resnet18(pretrained=False).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    bn_state = _strip_profile_keys(ckpt['bn'])
    for k in list(bn_state.keys()):  # strip possible BN buffers
        if 'memory' in k:
            bn_state.pop(k)
    missing, unexpected = bn.load_state_dict(bn_state, strict=False)
    if missing or unexpected:
        print(f"[WARN] BN load -> missing: {missing}, unexpected: {unexpected}")

    dec_state = _strip_profile_keys(ckpt['decoder'])
    dec_missing, dec_unexpected = decoder.load_state_dict(dec_state, strict=False)
    if dec_missing or dec_unexpected:
        print(f"[WARN] Decoder load -> missing: {dec_missing}, unexpected: {dec_unexpected}")

    ckpt_base = os.path.splitext(os.path.basename(ckpt_path))[0]
    vis_ckpt_root = os.path.join(vis_root, ckpt_base)

    class_metrics: List[dict] = []
    save_counts = {}
    for cls, loader in test_loaders:
        # full metrics
        allm = evaluate_full_metrics(encoder, bn, decoder, loader, device, top_ratio=0.01)
        print(
            f'[{cls}] SP_AUROC={allm["auroc_sp"]:.3f} SP_AP={allm["ap_sp"]:.3f} SP_F1={allm["f1_sp"]:.3f} | '
            f'PX_AUROC={allm["auroc_px"]:.3f} PX_AP={allm["ap_px"]:.3f} PX_F1={allm["f1_px"]:.3f} AUPRO={allm["aupro_px"]:.3f}'
        )
        class_metrics.append(allm)
        if save_heatmaps:
            out_dir = os.path.join(vis_ckpt_root, cls)
            saved_n = save_heatmaps_for_loader(encoder, bn, decoder, loader, device, out_dir, cls, filename_prefix=ckpt_base, image_size=image_size)
            total_n = len(loader.dataset)
            save_counts[cls] = (saved_n, total_n)
            if saved_n != total_n:
                print(f'  [WARN] {cls}: saved {saved_n} / {total_n} samples (possible overwrite or skip).')
            else:
                print(f'  [OK] {cls}: all {saved_n} samples saved.')

    avg_full = {k: float(np.mean([m[k] for m in class_metrics])) for k in class_metrics[0].keys()} if class_metrics else {}
    if avg_full:
        print(
            f'Average -> SP_AUROC={avg_full["auroc_sp"]:.3f} SP_AP={avg_full["ap_sp"]:.3f} SP_F1={avg_full["f1_sp"]:.3f} | '
            f'PX_AUROC={avg_full["auroc_px"]:.3f} PX_AP={avg_full["ap_px"]:.3f} PX_F1={avg_full["f1_px"]:.3f} AUPRO={avg_full["aupro_px"]:.3f}'
        )
    if save_heatmaps:
        print(f'Visualizations saved under: {vis_ckpt_root}')
        print('Per-class save counts:', save_counts)
    return class_metrics, avg_full
if __name__ == '__main__':
  
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']

    evaluate_from_checkpoint(
        classes=item_list,
        dataset_name='mvtec',
        data_root='/media/li/EDF4EB7FA5C2BA70/IAD_datasets/mvtec_AD',
        ckpt_path='/media/li/EDF4EB7FA5C2BA70/IADProjects/RRD/checkpoints/0116_wres50_multi_mvtec.pth',
        image_size=256,
        num_workers=4,
        save_heatmaps=True,
        vis_root='visualizations',
    )