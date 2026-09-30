import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _require_pil():
    if not _HAS_PIL:
        raise RuntimeError("PIL not available. Install pillow.")


def read_image_gray(path: str, resize: Optional[int] = 256) -> np.ndarray:
    _require_pil()
    img = Image.open(path).convert("L")
    if resize is not None:
        resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        img = img.resize((resize, resize), resample=resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin)


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    x = normalize_minmax(x)
    hist, bin_edges = np.histogram(x.ravel(), bins=bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    total = hist.sum()
    if total == 0:
        return 0.0

    sum_total = (bin_centers * hist).sum()
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 0.0

    for i in range(bins):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += bin_centers[i] * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = bin_centers[i]
    return float(threshold)


@torch.no_grad()
def _fit_line_torch(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if y.ndim == 1:
        y = y.unsqueeze(0)
    B, M = y.shape
    x = x.view(1, M)
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    xc = x - x_mean
    yc = y - y_mean
    var_x = (xc * xc).mean(dim=1, keepdim=True)
    cov_xy = (yc * xc).mean(dim=1, keepdim=True)

    b = (cov_xy / (var_x + 1e-12)).squeeze(1)
    a = (y_mean - b.view(-1, 1) * x_mean).squeeze(1)

    y_pred = a.view(-1, 1) + b.view(-1, 1) * x
    ss_res = ((y - y_pred) ** 2).sum(dim=1)
    ss_tot = ((y - y.mean(dim=1, keepdim=True)) ** 2).sum(dim=1) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return b.squeeze(0), a.squeeze(0), r2.squeeze(0)


@torch.no_grad()
def sobel_edge_map(gray: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    g = normalize_minmax(gray)
    if g.max() <= 1e-12:
        return np.zeros_like(g, dtype=np.uint8)
    t = torch.from_numpy(g).float().view(1, 1, g.shape[0], g.shape[1])
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy).squeeze().numpy()
    mag = normalize_minmax(mag)

    if threshold is None:
        threshold = otsu_threshold(mag)
    edge = (mag >= threshold).astype(np.uint8)
    return edge


@torch.no_grad()
def box_counting_fd(binary_map: torch.Tensor, ks: List[int]) -> Tuple[float, float, List[int], List[int]]:
    assert binary_map.ndim == 4 and binary_map.shape[0] == 1 and binary_map.shape[1] == 1
    device = binary_map.device
    dtype = torch.float32
    x = binary_map.to(dtype)
    _, _, H, W = x.shape

    ks_sorted = sorted(set(int(k) for k in ks if k >= 1))
    ks_used, counts = [], []

    for k in ks_sorted:
        Hk = (H // k) * k
        Wk = (W // k) * k
        if Hk < k or Wk < k:
            continue
        xk = x[:, :, :Hk, :Wk]
        w = torch.ones((1, 1, k, k), device=device, dtype=dtype)
        sums = F.conv2d(xk, w, stride=k, padding=0)
        nonempty = (sums > 0).to(dtype)
        n = int(nonempty.flatten().sum().item())
        ks_used.append(k)
        counts.append(n)

    if len(ks_used) < 2 or all(c == 0 for c in counts):
        return 0.0, 0.0, ks_used, counts

    ks_t = torch.tensor(ks_used, device=device, dtype=dtype)
    c_t = torch.tensor(counts, device=device, dtype=dtype)
    x_log = torch.log(1.0 / ks_t)
    y_log = torch.log(c_t + 1e-12)
    D, _, r2 = _fit_line_torch(x_log, y_log)
    return float(D.item()), float(r2.item()), ks_used, counts


def _split_indices(n: int, parts: int = 3) -> List[List[int]]:
    base = n // parts
    rem = n % parts
    out = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < rem else 0)
        out.append(list(range(start, start + size)))
        start += size
    return out


@torch.no_grad()
def multi_scale_fd(binary_map: torch.Tensor, ks: List[int]) -> Dict[str, float]:
    D_all, r2_all, ks_used, counts = box_counting_fd(binary_map, ks)
    out = {
        "D_all": D_all,
        "r2_all": r2_all,
        "D_small": 0.0,
        "r2_small": 0.0,
        "D_mid": 0.0,
        "r2_mid": 0.0,
        "D_large": 0.0,
        "r2_large": 0.0,
    }
    if len(ks_used) < 4:
        return out

    ks_t = torch.tensor(ks_used, device=binary_map.device, dtype=torch.float32)
    c_t = torch.tensor(counts, device=binary_map.device, dtype=torch.float32)
    x_log = torch.log(1.0 / ks_t)
    y_log = torch.log(c_t + 1e-12)

    splits = _split_indices(len(ks_used), parts=3)
    for name, idx in zip(["small", "mid", "large"], splits):
        if len(idx) < 2:
            continue
        idx_t = torch.tensor(idx, device=binary_map.device)
        mask = c_t[idx_t] > 0
        if mask.sum().item() < 2:
            continue
        D, _, r2 = _fit_line_torch(x_log[idx_t][mask], y_log[idx_t][mask])
        out[f"D_{name}"] = float(D.item())
        out[f"r2_{name}"] = float(r2.item())
    return out


@torch.no_grad()
def local_fd_stats(edge_map: torch.Tensor, patch: int = 64, ks: List[int] = [2, 4, 8, 16]) -> Dict[str, float]:
    assert edge_map.ndim == 4 and edge_map.shape[0] == 1 and edge_map.shape[1] == 1
    _, _, H, W = edge_map.shape
    Hn = (H // patch) * patch
    Wn = (W // patch) * patch
    if Hn < patch or Wn < patch:
        return {"lfd_mean": 0.0, "lfd_std": 0.0, "lfd_skew": 0.0, "lfd_kurt": 0.0, "lfd_high_ratio": 0.0}

    x = edge_map[0, 0, :Hn, :Wn]
    x = x.reshape(Hn // patch, patch, Wn // patch, patch)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(-1, 1, patch, patch)

    Ds = []
    for i in range(x.shape[0]):
        D, r2, _, _ = box_counting_fd(x[i:i + 1], ks)
        if D > 0 and r2 > 0:
            Ds.append(D)
    if len(Ds) == 0:
        return {"lfd_mean": 0.0, "lfd_std": 0.0, "lfd_skew": 0.0, "lfd_kurt": 0.0, "lfd_high_ratio": 0.0}

    d = np.array(Ds, dtype=np.float32)
    mean = float(d.mean())
    std = float(d.std() + 1e-12)
    skew = float(((d - mean) ** 3).mean() / (std ** 3))
    kurt = float(((d - mean) ** 4).mean() / (std ** 4) - 3.0)
    high_ratio = float((d > (mean + std)).mean())
    return {"lfd_mean": mean, "lfd_std": std, "lfd_skew": skew, "lfd_kurt": kurt, "lfd_high_ratio": high_ratio}


@torch.no_grad()
def laplacian_pyramid_slope(img_g: torch.Tensor, levels: int = 4) -> Dict[str, float]:
    assert img_g.ndim == 4 and img_g.shape[0] == 1 and img_g.shape[1] == 1
    x = img_g
    energies, scales = [], []

    gk = torch.tensor([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]], dtype=x.dtype, device=x.device)
    gk = (gk / gk.sum()).view(1, 1, 5, 5)

    for l in range(levels):
        xb = F.conv2d(x, gk, padding=2)
        xd = F.avg_pool2d(xb, kernel_size=2, stride=2)
        xu = F.interpolate(xd, scale_factor=2, mode="bilinear", align_corners=False)
        xu = xu[:, :, :x.shape[2], :x.shape[3]]
        lap = x - xu
        E = lap.abs().mean().item()
        energies.append(max(E, 1e-12))
        scales.append(2 ** l)
        x = xd
        if x.shape[2] < 16 or x.shape[3] < 16:
            break

    if len(energies) < 2:
        return {"lp_slope": 0.0, "lp_r2": 0.0}

    e = torch.tensor(energies, dtype=torch.float32, device=img_g.device)
    s = torch.tensor(scales, dtype=torch.float32, device=img_g.device)
    x_log = torch.log(s)
    y_log = torch.log(e)
    b, _, r2 = _fit_line_torch(x_log, y_log)
    return {"lp_slope": float(b.item()), "lp_r2": float(r2.item())}


@dataclass
class FractalGrayConfig:
    box_sizes: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)
    local_patch: int = 64
    local_box_sizes: Tuple[int, ...] = (2, 4, 8, 16)
    pyramid_levels: int = 4
    resize: Optional[int] = 256
    edge_threshold: Optional[float] = None


class FractalGrayExtractor:
    def __init__(self, cfg: FractalGrayConfig = FractalGrayConfig(), device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

    def extract_from_gray(self, gray: np.ndarray, return_debug: bool = False):
        # 优化：只计算需要的特征，节省时间
        g = normalize_minmax(gray)
        H, W = g.shape
        g_t = torch.from_numpy(g.astype(np.float32)).to(self.device).view(1, 1, H, W)
        
        # 1 & 2: 拉普拉斯金字塔特征 (最重要)
        lp = laplacian_pyramid_slope(g_t, levels=self.cfg.pyramid_levels)
        
        # 3: 边缘密度特征
        edge = sobel_edge_map(g, threshold=self.cfg.edge_threshold)
        edge_density = float(edge.sum()) / float(H * W + 1e-12)

        # 4: 全局分形维数
        edge_t = torch.from_numpy(edge.astype(np.float32)).to(self.device).view(1, 1, H, W)
        D_all, r2_all, ks_used, counts = box_counting_fd(edge_t, list(self.cfg.box_sizes))

        feats = [
            lp["lp_slope"], lp["lp_r2"],
            edge_density,
            D_all
        ]
        out = np.array(feats, dtype=np.float32)

        if not return_debug:
            return out

        D_all, r2_all, ks_used, counts = box_counting_fd(edge_t, list(self.cfg.box_sizes))
        if len(ks_used) >= 2:
            ks_t = torch.tensor(ks_used, dtype=torch.float32)
            c_t = torch.tensor(counts, dtype=torch.float32)
            x_log = torch.log(1.0 / ks_t)
            y_log = torch.log(c_t + 1e-12)
            b, a, _ = _fit_line_torch(x_log, y_log)
            y_fit = a + b * x_log
            log_x = x_log.cpu().numpy()
            log_y = y_log.cpu().numpy()
            log_fit = y_fit.cpu().numpy()
        else:
            log_x = np.array([])
            log_y = np.array([])
            log_fit = np.array([])

        debug = {
            "gray": g,
            "edge": edge,
            "ks_used": ks_used,
            "counts": counts,
            "log_x": log_x,
            "log_y": log_y,
            "log_fit": log_fit,
            "D_all": D_all,
            "r2_all": r2_all,
        }
        return out, debug

    def extract_from_path(self, path: str, return_debug: bool = False):
        gray = read_image_gray(path, resize=self.cfg.resize)
        return self.extract_from_gray(gray, return_debug=return_debug)


def visualize(path: str, extractor: FractalGrayExtractor, save_path: Optional[str] = None, show: bool = True):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib not available. Install matplotlib to visualize.") from exc

    feats, debug = extractor.extract_from_path(path, return_debug=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(debug["gray"], cmap="gray")
    axes[0].set_title("Grayscale")
    axes[0].axis("off")

    axes[1].imshow(debug["edge"], cmap="gray")
    axes[1].set_title("Edge Map (Sobel)")
    axes[1].axis("off")

    axes[2].plot(debug["log_x"], debug["log_y"], "o", label="counts")
    if debug["log_fit"].size > 0:
        axes[2].plot(debug["log_x"], debug["log_fit"], "-", label="fit")
    axes[2].set_title(f"log-log fit (D={debug['D_all']:.3f}, R2={debug['r2_all']:.3f})")
    axes[2].set_xlabel("log(1/k)")
    axes[2].set_ylabel("log(N(k))")
    axes[2].legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return feats


def extract_top4_fractal_features(image_path: str, resize: int = 256, device: str = "cpu") -> np.ndarray:
    """
    提供给外部调用的简洁接口：
    读取图像并提取重要性排名前4的分形特征。
    返回的特征维度: (4,)
    顺序为: [lp_slope, lp_r2, edge_density, D_all]
    """
    cfg = FractalGrayConfig(resize=resize)
    extractor = FractalGrayExtractor(cfg, device=device)
    feats = extractor.extract_from_path(image_path, return_debug=False)
    return feats

def main():
    # Edit these values as needed.
    image_path = r"Anthrax Leaf00129.JPG"
    resize = 224
    device = "cpu"
    save_fig = None
    show = True

    resize_value = None if resize == 0 else int(resize)
    cfg = FractalGrayConfig(resize=resize_value)
    extractor = FractalGrayExtractor(cfg, device=device)

    feats = visualize(image_path, extractor, save_path=save_fig, show=show)
    print("Feature dim:", feats.shape)
    print(feats)


if __name__ == "__main__":
    main()
