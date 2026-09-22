import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

from dataset import get_data_transforms, VisaDataset, MVTecDataset, HDDataset
from resnet import resnet101,wide_resnet50_2, resnet50, resnet34, resnet18
from de_resnet import de_resnet101 , de_wide_resnet50_2, de_resnet50, de_resnet34, de_resnet18
from metrics import compute_all_metrics
from thop import profile, clever_format


# -------------------- utils --------------------
def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_function(a: List[torch.Tensor], b: List[torch.Tensor]) -> torch.Tensor:
    """Cosine feature alignment across all scales."""
    cos = nn.CosineSimilarity()
    loss = 0.0
    for i in range(len(a)):
        loss = loss + torch.mean(1 - cos(a[i].view(a[i].shape[0], -1), b[i].view(b[i].shape[0], -1)))
    return loss



def latent_feature_recomposition(
    features,
    r_min=0.15,
    r_max=0.35,
    n_patches=(1,2),
    center_bias=True,
    center_sigma=0.20,
):
    assert isinstance(features, (list, tuple)) and len(features) > 0
    assert 0 < r_min <= r_max < 1.0
    assert isinstance(n_patches, (tuple, list)) and len(n_patches) == 2
    assert n_patches[0] >= 1 and n_patches[1] >= n_patches[0]

    # 批次大小与设备
    batch_size = features[0].shape[0]
    device = features[0].device

    # 随机打乱 batch 维：跨样本取源特征
    indices = torch.randperm(batch_size, device=device)

    # 复制一份作为输出
    augmented_features = [f.clone() for f in features]

    def _sample_center_biased_top_left(W, H, pw, ph, sigma_ratio):
        """
        在 [0, W-pw] / [0, H-ph] 里采样目标左上角坐标，中心偏置（高斯）
        """
        # 以中心为均值的高斯采样，然后转为左上角
        # sigma 用比例控制
        sig_x = max(1e-6, W * sigma_ratio)
        sig_y = max(1e-6, H * sigma_ratio)

        cx = int(random.gauss(W / 2.0, sig_x))
        cy = int(random.gauss(H / 2.0, sig_y))

        # 将中心点转换成左上角，并做边界裁剪
        tx = cx - pw // 2
        ty = cy - ph // 2

        tx = min(max(tx, 0), W - pw)
        ty = min(max(ty, 0), H - ph)
        return tx, ty

    for i in range(len(features)):
        # 当前层级特征图
        feature_map = augmented_features[i]
        B, C, H, W = feature_map.shape

        # 源特征图（跨样本置换）
        source_feature_map = features[i][indices]

        # 每个层级注入多个 patch（升级C）
        k = random.randint(n_patches[0], n_patches[1])

        for _ in range(k):
            # 多尺度 patch（升级A）
            ratio = random.uniform(r_min, r_max)
            patch_h = max(1, int(H * ratio))
            patch_w = max(1, int(W * ratio))

            # 防止极端小特征图导致越界
            patch_h = min(patch_h, H)
            patch_w = min(patch_w, W)

            # 随机选择源 patch 左上角
            src_x = random.randint(0, W - patch_w)
            src_y = random.randint(0, H - patch_h)

            patch = source_feature_map[:, :, src_y:src_y + patch_h, src_x:src_x + patch_w]

            # 目标位置：随机 or 中心偏置（升级B）
            if center_bias:
                tgt_x, tgt_y = _sample_center_biased_top_left(W, H, patch_w, patch_h, center_sigma)
            else:
                tgt_x = random.randint(0, W - patch_w)
                tgt_y = random.randint(0, H - patch_h)

            # 注入（重组）
            feature_map[:, :, tgt_y:tgt_y + patch_h, tgt_x:tgt_x + patch_w] = patch

    return augmented_features

# --------------- dataset wrappers ---------------
class GoodOnlyFolderDataset(Dataset):
    """Load images from '<data_root>/<class>/train/good' without needing sub-class folders."""

    def __init__(self, good_dir: str, transform=None):
        self.transform = transform
        self.paths: List[str] = []
        if os.path.isdir(good_dir):
            for root, _, files in os.walk(good_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.ppm', '.pnm'}:
                        self.paths.append(os.path.join(root, f))
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


class SourceIndexedDataset(Dataset):
    """
    Wrap an ImageFolder to return (img_tensor, source_class_idx) and discard original label.
    Only used to keep track of which class a sample came from during multi-class training.
    """

    def __init__(self, base: Dataset, source_idx: int):
        self.base = base
        self.source_idx = source_idx

    def __len__(self):
        return int(len(self.base))

    def __getitem__(self, idx):
        img = self.base[idx]
        return img, self.source_idx


# --------------- anomaly scoring ---------------
def cal_anomaly_map(fs_list: List[torch.Tensor], ft_list: List[torch.Tensor], out_size: int) -> torch.Tensor:
    """Compute per-pixel anomaly map from teacher/student multi-scale features via cosine distance."""
    anomaly_map = None
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[:, 0:1, :, :]  # [B,1,H,W]
        if anomaly_map is None:
            anomaly_map = a_map
        else:
            anomaly_map = anomaly_map + a_map
    return anomaly_map  # [B,1,H,W]


def image_level_scores(amap: torch.Tensor, reduce: str = 'max') -> torch.Tensor:
    """Reduce anomaly map [B,1,H,W] to image-level score: max or mean."""
    if reduce == 'max':
        return torch.amax(amap, dim=[2, 3]).squeeze(1)
    elif reduce == 'mean':
        return torch.mean(amap, dim=[2, 3]).squeeze(1)
    else:
        raise ValueError('reduce must be max or mean')


def evaluate_image_level(encoder, bn, decoder, dataloader: DataLoader, device: str, reduce: str = 'max') -> Tuple[
    float, float, float]:
    """Image-level AUROC/ACC/F1 using dataset-provided labels (0=good,1=anomaly)."""
    bn.eval();
    decoder.eval()
    gts: List[int] = []
    scores: List[float] = []
    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            t_feats = encoder(img)
            s_feats = decoder(bn(t_feats))
            amap = cal_anomaly_map(t_feats, s_feats, img.shape[-1])  # [B,1,H,W]
            img_scores = image_level_scores(amap, reduce=reduce)
            scores.extend(img_scores.detach().cpu().tolist())
            gts.extend(label.cpu().numpy().astype(int).tolist())
    # Normalize scores to 0-1
    scores_np = np.array(scores)
    if scores_np.max() > scores_np.min():
        scores_norm = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min())
    else:
        scores_norm = scores_np
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    auroc = float(roc_auc_score(gts, scores_norm)) if len(set(gts)) > 1 else 0.0
    best_acc, best_f1 = 0.0, 0.0
    for th in np.linspace(0, 1, 200):
        preds = (scores_norm >= th).astype(int)
        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds)
        if acc > best_acc:
            best_acc, best_f1 = acc, f1
    return auroc, best_acc, best_f1


# --------------- training entry ---------------
def train_multi(
        classes: List[str],
        dataset_name: str = 'visa',
        data_root: str = '',
        image_size: int = 256,
        batch_size: int = 32,
        epochs: int = 100,
        lr: float = 5e-3,
        ckpt_path: str = '',
        num_workers: int = 4,
):
    """
    Train a single model across multiple classes using only normal training samples.
    - classes: list of category names.
    - dataset_name: one of {'visa','mvtec','hd'} for how to build test loaders.
    - data_root: parent folder containing per-class folders.
    - returns: checkpoint path with best average SP_AUROC across classes, evaluated every 5 epochs.
    """
    setup_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_transform, gt_transform = get_data_transforms(image_size, image_size)

    # Build multi-class normal training set (ConcatDataset of ImageFolder per class)
    train_wrapped: List[Dataset] = []
    for i, cls in enumerate(classes):
        # load only normal images under train/good
        good_dir = os.path.join(data_root, cls, 'train', 'good')
        if os.path.isdir(good_dir):
            base_ds = GoodOnlyFolderDataset(good_dir, transform=data_transform)
        else:
            # Fallback: use ImageFolder at 'train' and rely on 'good' subfolder being the only one for training
            base_ds = ImageFolder(root=os.path.join(data_root, cls, 'train'), transform=data_transform)
        train_wrapped.append(SourceIndexedDataset(base_ds, source_idx=i))
    train_dataset = ConcatDataset(train_wrapped)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)

    # Per-class test dataloaders for image-level evaluation
    test_loaders: List[Tuple[str, DataLoader]] = []
    for cls in classes:
        cls_root = os.path.join(data_root, cls)
        if dataset_name.lower() == 'visa':
            test_set = VisaDataset(root=cls_root, transform=data_transform, gt_transform=gt_transform, phase='test')
        elif dataset_name.lower() == 'mvtec':
            test_set = MVTecDataset(root=cls_root, transform=data_transform, gt_transform=gt_transform, phase='test')
        elif dataset_name.lower() == 'hd':
            test_set = HDDataset(root=cls_root, transform=data_transform, gt_transform=gt_transform, phase='test')
        else:
            raise ValueError('Unknown dataset_name: ' + dataset_name)
        test_loaders.append((cls, DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers)))

    # Model
    # encoder, bn = resnet101(pretrained=True)
    encoder, bn = wide_resnet50_2(pretrained=True)
    # encoder, bn = resnet50(pretrained=True)
    # encoder, bn = resnet34(pretrained=True)
    # encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()  # teacher frozen

    # decoder = de_resnet101(pretrained=False).to(device)
    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    # decoder = de_resnet50(pretrained=False).to(device)
    # decoder = de_resnet34(pretrained=False).to(device)
    # decoder = de_resnet18(pretrained=False).to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=lr, betas=(0.5, 0.999))

    # Calculate FLOPs and Params
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    # We need to wrap bn and decoder into a single module to calculate total FLOPs easily,
    # or calculate them separately. Since bn takes encoder output, we need to run encoder first.
    with torch.no_grad():
        t_feats = encoder(dummy_input)

    bn_flops, bn_params = profile(bn, inputs=(t_feats, ), verbose=False)

    # Calculate FLOPs for Decoder
    # Decoder takes the output of BN layer
    with torch.no_grad():
        bn_output = bn(t_feats)

    decoder_flops, decoder_params = profile(decoder, inputs=(bn_output, ), verbose=False)

    total_flops = bn_flops + decoder_flops
    total_params = bn_params + decoder_params

    flops_str, params_str = clever_format([total_flops, total_params], "%.3f")
    print(f"Total Trainable Params: {params_str}")
    print(f"Total FLOPs: {flops_str}")

    # print('Trainable params (M):', round(count_parameters(decoder) / 1e6 + count_parameters(bn) / 1e6, 3))

    best_avg_sp_auroc = -1.0
    best_avg_px_auroc = -1.0

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_hist = []
        for imgs, src_idx in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                t_feats = encoder(imgs)
            # 指定类别在中心位置追加 CutPaste 噪声
            # noisy_inputs = feature_cutpaste(t_feats)
            noisy_inputs = latent_feature_recomposition(t_feats)
            s_feats = decoder(bn(noisy_inputs))
            loss = loss_function(t_feats, s_feats)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())

        print(f'Epoch [{epoch + 1}/{epochs}] loss={np.mean(loss_hist):.4f}')
        # Evaluate every 5 epochs using SP/PX metrics only
        if (epoch + 1) % 5 == 0:
            bn.eval();
            decoder.eval()
            full_metrics = []
            for cls, loader in test_loaders:
                # accumulate pixel-level anomaly maps and masks for full metrics
                labels = []
                masks = []
                amaps_list = []
                with torch.no_grad():
                    for img, gt, label, _ in loader:
                        img = img.to(device)
                        t_feats = encoder(img)
                        s_feats = decoder(bn(t_feats))
                        amap = cal_anomaly_map(t_feats, s_feats, img.shape[-1])  # [B,1,H,W]
                        amaps_list.append(amap.squeeze(1).cpu().numpy())
                        masks.append(gt.squeeze(1).cpu().numpy())
                        labels.extend(label.cpu().numpy().astype(int).tolist())
                amaps_np = np.concatenate(amaps_list, axis=0)
                masks_np = np.concatenate(masks, axis=0)
                allm = compute_all_metrics(labels, masks_np, amaps_np, top_ratio=0.01)
                print(
                    f'  [{cls}] SP_AUROC={allm["auroc_sp"]:.3f} SP_AP={allm["ap_sp"]:.3f} SP_F1={allm["f1_sp"]:.3f} | '
                    f'PX_AUROC={allm["auroc_px"]:.3f} PX_AP={allm["ap_px"]:.3f} PX_F1={allm["f1_px"]:.3f} AUPRO={allm["aupro_px"]:.3f}'
                )
                full_metrics.append(allm)
            # Average the kept metrics across classes
            avg_full = {k: float(np.mean([fm[k] for fm in full_metrics])) for k in full_metrics[0].keys()}
            print(
                f'  Average -> SP_AUROC={avg_full["auroc_sp"]:.3f} SP_AP={avg_full["ap_sp"]:.3f} SP_F1={avg_full["f1_sp"]:.3f} | '
                f'PX_AUROC={avg_full["auroc_px"]:.3f} PX_AP={avg_full["ap_px"]:.3f} PX_F1={avg_full["f1_px"]:.3f} AUPRO={avg_full["aupro_px"]:.3f}'
            )
            sp = avg_full['auroc_sp']
            px = avg_full['auroc_px']
            sp_r = round(sp + 1e-10, 3)
            px_r = round(px + 1e-10, 3)
            best_sp_r = round(best_avg_sp_auroc + 1e-10, 3)
            best_px_r = round(best_avg_px_auroc + 1e-10, 3)
            should_save = False
            if sp_r > best_sp_r:
                should_save = True
                reason = f"Rounded SP_AUROC improved {best_sp_r:.3f} -> {sp_r:.3f}"
            elif sp_r == best_sp_r and px_r > best_px_r:
                should_save = True
                reason = f"Rounded SP_AUROC tie at {sp_r:.3f}; PX_AUROC improved {best_px_r:.3f} -> {px_r:.3f}"
            if should_save:
                best_avg_sp_auroc = sp
                best_avg_px_auroc = px
                print(f'  {reason}, saving to {ckpt_path}')
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict(), 'classes': classes,
                            'dataset_name': dataset_name, 'image_size': image_size, 'avg_full': avg_full,
                            'best_sp_auroc': round(best_avg_sp_auroc, 6), 'best_px_auroc': round(best_avg_px_auroc, 6)},
                           ckpt_path)
            else:
                print(
                    f'  No save (rounded compare): SP {sp_r:.3f} (best {best_sp_r:.3f}), PX {px_r:.3f} (best {best_px_r:.3f})')
    # Final save
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict(), 'classes': classes,
                    'dataset_name': dataset_name, 'image_size': image_size}, ckpt_path)
    return ckpt_path


if __name__ == '__main__':

    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']

    ckpt = train_multi(
        classes=item_list,
        dataset_name='mvtec',
        data_root='/media/li/EDF4EB7FA5C2BA70/IAD_datasets/mvtec_AD',
        ckpt_path='/media/li/EDF4EB7FA5C2BA70/IADProjects/RRD/checkpoints/0116_wres50_multi_mvtec.pth',
        image_size=256,
        batch_size=48,
        epochs=250,
        lr=5e-3,
        num_workers=4,
    )
    print('Saved checkpoint:', ckpt)
