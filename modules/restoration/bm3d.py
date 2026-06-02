"""自定义 BM3D (Block-Matching and 3D Collaborative Filtering) 去噪实现。

移植自 VisionRestorePro (v4.9) 的自实现 BM3D，不依赖外部 bm3d 库。
"""

import numpy as np
from itertools import product
from scipy.fft import dct as _dct, idct as _idct

# ── YCbCr 转换辅助 ────────────────────────────────────────────────

def _rgb_to_ycbcr(img: np.ndarray) -> tuple:
    """RGB → YCbCr，返回 (y, cb, cr)"""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 + (-0.168736 * r - 0.331264 * g + 0.5 * b) * 255
    cr = 128 + (0.5 * r - 0.418688 * g - 0.081312 * b) * 255
    return y, cb, cr


def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray, shape: tuple) -> np.ndarray:
    """YCbCr → RGB"""
    out = np.zeros(shape, dtype=np.float64)
    out[:, :, 0] = y + 0.0 * (cb - 128) / 255 + 1.402 * (cr - 128) / 255
    out[:, :, 1] = y - 0.344136 * (cb - 128) / 255 - 0.714136 * (cr - 128) / 255
    out[:, :, 2] = y + 1.772 * (cb - 128) / 255 + 0.0 * (cr - 128) / 255
    return np.clip(out, 0, 1)


# ── DCT 基 ──────────────────────────────────────────────────────────

def compute_dct_basis(size: int) -> np.ndarray:
    """预计算 4D DCT 基矩阵 (size, size, size, size)"""
    basis = np.zeros((size, size, size, size))
    for k, n in product(range(size), range(size)):
        for i, j in product(range(size), range(size)):
            basis[k, n, i, j] = (
                np.cos((np.pi * (2 * i + 1) * k) / (2 * size))
                * np.cos((np.pi * (2 * j + 1) * n) / (2 * size))
            )
    # 归一化
    for k in range(size):
        for n in range(size):
            ck = 1.0 / np.sqrt(size) if k == 0 else np.sqrt(2.0 / size)
            cn = 1.0 / np.sqrt(size) if n == 0 else np.sqrt(2.0 / size)
            basis[k, n] *= ck * cn
    return basis


def dct_2d(patch: np.ndarray, dct_basis: np.ndarray) -> np.ndarray:
    """2D DCT: 使用预计算基"""
    size = patch.shape[0]
    result = np.zeros((size, size))
    for k, n in product(range(size), range(size)):
        result[k, n] = np.sum(patch * dct_basis[k, n])
    return result


def idct_2d(transformed_patch: np.ndarray, dct_basis: np.ndarray) -> np.ndarray:
    """逆 2D DCT"""
    size = transformed_patch.shape[0]
    result = np.zeros((size, size))
    for i, j in product(range(size), range(size)):
        result[i, j] = np.sum(transformed_patch * dct_basis[:, :, i, j])
    return result


# ── Block Matching ──────────────────────────────────────────────────

def block_matching(
    img: np.ndarray,
    ref_patch: np.ndarray,
    ref_pos: tuple,
    search_window: int,
    max_matches: int,
    patch_size: int,
    sigma: float,
) -> list:
    """在 search_window 内寻找相似块（L2 距离），返回 (position, dist) 列表"""
    h, w = img.shape
    ry, rx = ref_pos
    half = search_window // 2
    y_start = max(0, ry - half)
    y_end = min(h - patch_size, ry + half)
    x_start = max(0, rx - half)
    x_end = min(w - patch_size, rx + half)

    matches = []
    threshold = 3.0 * sigma * sigma  # L2 距离阈值

    for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
            candidate = img[y:y + patch_size, x:x + patch_size].astype(np.float64)
            diff = ref_patch - candidate
            dist = np.sum(diff * diff) / (patch_size * patch_size)
            if dist < threshold:
                matches.append(((y, x), dist))

    matches.sort(key=lambda m: m[1])
    matches = matches[:max_matches]

    if not matches:
        matches = [(ref_pos, 0.0)]

    return matches


# ── 3D Transform / Inverse ─────────────────────────────────────────

def apply_3d_transform(group_3d: np.ndarray, dct_basis: np.ndarray) -> np.ndarray:
    """对 3D 块组应用 2D DCT + 1D DCT"""
    n_patches = group_3d.shape[0]
    patch_size = group_3d.shape[1]
    transformed = np.zeros_like(group_3d)

    for i in range(n_patches):
        transformed[i] = dct_2d(group_3d[i], dct_basis)

    transformed = np.apply_along_axis(lambda x: _dct(x, norm='ortho'), 0, transformed)
    return transformed


def apply_inverse_3d_transform(transformed_group: np.ndarray, dct_basis: np.ndarray) -> np.ndarray:
    """逆 3D 变换：1D IDCT + 2D IDCT"""
    result = np.apply_along_axis(lambda x: _idct(x, norm='ortho'), 0, transformed_group)
    n_patches = result.shape[0]
    for i in range(n_patches):
        result[i] = idct_2d(result[i], dct_basis)
    return result


def hard_thresholding(coefficients: np.ndarray, threshold: float) -> np.ndarray:
    """硬阈值滤波"""
    return np.where(np.abs(coefficients) < threshold, 0, coefficients)


# ── Collaborative Filtering ─────────────────────────────────────────

def collaborative_filtering_3d(
    patches_positions: list,
    img: np.ndarray,
    sigma: float,
    hard_threshold: float,
    dct_basis: np.ndarray,
) -> tuple:
    """3D 协同滤波：分组 → 变换 → 阈值 → 逆变换 → 聚合"""
    patch_size = dct_basis.shape[0]
    threshold = hard_threshold * sigma

    group_3d = np.zeros((len(patches_positions), patch_size, patch_size))
    for i, (pos, _) in enumerate(patches_positions):
        y, x = pos
        group_3d[i] = img[y:y + patch_size, x:x + patch_size].astype(np.float64)

    transformed = apply_3d_transform(group_3d, dct_basis)
    filtered = hard_thresholding(transformed, threshold)
    nonzero_count = np.count_nonzero(filtered)
    reconstructed = apply_inverse_3d_transform(filtered, dct_basis)

    return reconstructed, nonzero_count


# ── 主 BM3D 算法 ───────────────────────────────────────────────────

def bm3d_denoising(
    img: np.ndarray,
    sigma: float,
    patch_size: int = 8,
    step_size: int = 3,
    search_window: int = 39,
    max_matches: int = 16,
    hard_threshold: float = 2.7,
) -> np.ndarray:
    """BM3D 去噪主算法（第一阶段）

    Args:
        img: 二维灰度图 (H, W)，float64 范围 [0, 1]
        sigma: 噪声标准差
        patch_size: 块大小（默认 8）
        step_size: 步长（默认 3）
        search_window: 搜索窗口大小（默认 39）
        max_matches: 最大匹配块数（默认 16）
        hard_threshold: 硬阈值系数（默认 2.7）

    Returns:
        去噪后的二维灰度图 (H, W)
    """
    if img.ndim != 2:
        raise ValueError(f"bm3d_denoising expects 2D array, got {img.ndim}D")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    h, w = img.shape
    if patch_size > min(h, w):
        raise ValueError(f"patch_size {patch_size} larger than image {h}x{w}")
    dct_basis = compute_dct_basis(patch_size)

    denoised = np.zeros_like(img, dtype=np.float64)
    weight_accum = np.zeros_like(img, dtype=np.float64)

    for y in range(0, h - patch_size + 1, step_size):
        for x in range(0, w - patch_size + 1, step_size):
            ref_patch = img[y:y + patch_size, x:x + patch_size].astype(np.float64)

            matches = block_matching(img, ref_patch, (y, x), search_window, max_matches, patch_size, sigma)

            if len(matches) < 1:
                continue

            reconstructed, nonzeros = collaborative_filtering_3d(matches, img, sigma, hard_threshold, dct_basis)

            weight = max(nonzeros, 1)
            for i, (pos, _) in enumerate(matches):
                py, px = pos
                denoised[py:py + patch_size, px:px + patch_size] += reconstructed[i] * weight
                weight_accum[py:py + patch_size, px:px + patch_size] += weight

    mask = weight_accum > 0
    denoised[mask] /= weight_accum[mask]
    denoised[~mask] = img[~mask]
    return np.clip(denoised, 0, 1)


def bm3d_rgb_denoising(img_rgb: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """RGB 图像 BM3D 去噪：仅对 Y 通道（亮度）做 BM3D"""
    img_rgb = np.clip(img_rgb, 0, 1)
    y, cb, cr = _rgb_to_ycbcr(img_rgb)
    sigma = max(noise_level, 0.01)
    denoised_y = bm3d_denoising(y, sigma)
    return _ycbcr_to_rgb(denoised_y, cb, cr, img_rgb.shape)


def denoise_luminance(img_rgb: np.ndarray, sigma: float, use_bm3d: bool = True) -> np.ndarray:
    """对亮度通道去噪，保留色度"""
    img_rgb = np.clip(img_rgb, 0, 1).astype(np.float64)
    y, cb, cr = _rgb_to_ycbcr(img_rgb)

    if use_bm3d:
        denoised_y = bm3d_denoising(y, max(sigma, 0.01))
    else:
        from scipy.ndimage import gaussian_filter
        denoised_y = gaussian_filter(y, sigma=sigma)

    return _ycbcr_to_rgb(denoised_y, cb, cr, img_rgb.shape)
