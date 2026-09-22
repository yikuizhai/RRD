import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from skimage import measure
from sklearn.metrics import auc as sk_auc


def f1_max_from_scores(y_true, y_score):
    """Max F1 from precision-recall curve.
    y_true: (M,), 0/1; y_score: (M,), float
    """
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thrs) = len(precs) - 1
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    if f1s.size <= 1:
        return 0.0
    return float(np.nanmax(f1s[:-1]))


def image_level_scores(amaps, top_ratio=0.0):
    """Reduce anomaly maps to image-level scores.
    amaps: (N,H,W)
    If top_ratio <= 0: use global max.
    If 0 < top_ratio < 1: treat as proportion of pixels (ceil).
    If top_ratio >= 1: treat as absolute number of top pixels.
    Returns (N,) scores.
    """
    N = amaps.shape[0]
    flat = amaps.reshape(N, -1)
    if N == 0:
        return np.zeros((0,), dtype=np.float32)
    total_pixels = flat.shape[1]
    if top_ratio <= 0:
        return flat.max(axis=1)
    if top_ratio < 1:
        k = int(np.ceil(total_pixels * float(top_ratio)))
    else:
        k = int(top_ratio)
    k = max(1, min(k, total_pixels))
    # partial selection of top-k values per row using argpartition
    idx = np.argpartition(-flat, kth=k - 1, axis=1)[:, :k]
    topk = np.take_along_axis(flat, idx, axis=1)
    return topk.mean(axis=1)


def aupro(masks, amaps, num_th=200, fpr_stop=0.3):
    """Area under the PRO curve up to FPR < fpr_stop.
    masks: (N,H,W) bool/0-1, amaps: (N,H,W) float
    """
    masks = masks.astype(bool)
    amaps = amaps.astype(np.float32)
    N, H, W = masks.shape
    if N == 0:
        return 0.0
    mn, mx = float(amaps.min()), float(amaps.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return 0.0
    thresholds = np.linspace(mn, mx, num=num_th, endpoint=False)[1:]  # skip min to avoid all-negative
    pros, fprs = [], []

    inv_masks = ~masks
    denom_bg = inv_masks.sum()
    for th in thresholds:
        pred = np.greater(amaps, th)  # numpy boolean array
        # region-wise PRO
        cur_pros = []
        for i in range(N):
            labeled = measure.label(masks[i].astype(np.uint8), connectivity=1)
            props = measure.regionprops(labeled)
            if not props:
                continue
            for reg in props:
                rr, cc = reg.coords[:, 0], reg.coords[:, 1]
                tp_pixels = pred[i][rr, cc].sum()
                cur_pros.append(tp_pixels / max(reg.area, 1))
        # background FPR
        if denom_bg > 0:
            fp = np.logical_and(inv_masks, pred).sum()
            cur_fpr = fp / denom_bg
        else:
            cur_fpr = 0.0

        pros.append(float(np.mean(cur_pros)) if len(cur_pros) > 0 else 0.0)
        fprs.append(float(cur_fpr))

    pros = np.array(pros, dtype=np.float32)
    fprs = np.array(fprs, dtype=np.float32)
    sel = fprs < fpr_stop
    if not np.any(sel):
        return 0.0
    fprs_sel = fprs[sel]
    pros_sel = pros[sel]
    if fprs_sel.max() > 0:
        fprs_sel = fprs_sel / fprs_sel.max()
    return float(sk_auc(fprs_sel, pros_sel))


def compute_all_metrics(labels, masks, amaps, top_ratio=0.01):
    """Compute image-level and pixel-level metrics.
    labels: (N,), 0/1; masks/amaps: (N,H,W)
    Returns dict with auroc_sp/ap_sp/f1_sp and auroc_px/ap_px/f1_px/aupro_px.
    """
    labels = np.asarray(labels).astype(np.int32).reshape(-1)
    masks = np.asarray(masks).astype(bool)
    amaps = np.asarray(amaps).astype(np.float32)
    N = labels.shape[0]
    assert masks.shape[0] == amaps.shape[0] == N

    # image-level
    sp_scores = image_level_scores(amaps, top_ratio=top_ratio)
    if len(np.unique(labels)) > 1:
        auroc_sp = roc_auc_score(labels, sp_scores)
        ap_sp = average_precision_score(labels, sp_scores)
        f1_sp = f1_max_from_scores(labels, sp_scores)
    else:
        auroc_sp = ap_sp = f1_sp = 0.0

    # pixel-level
    y_true_px = masks.reshape(-1).astype(np.uint8)
    y_score_px = amaps.reshape(-1).astype(np.float32)
    if len(np.unique(y_true_px)) > 1:
        auroc_px = roc_auc_score(y_true_px, y_score_px)
        ap_px = average_precision_score(y_true_px, y_score_px)
        f1_px = f1_max_from_scores(y_true_px, y_score_px)
    else:
        auroc_px = ap_px = f1_px = 0.0

    aupro_px = aupro(masks, amaps, num_th=200, fpr_stop=0.3)

    return dict(
        auroc_sp=float(auroc_sp), ap_sp=float(ap_sp), f1_sp=float(f1_sp),
        auroc_px=float(auroc_px), ap_px=float(ap_px), f1_px=float(f1_px), aupro_px=float(aupro_px)
    )
