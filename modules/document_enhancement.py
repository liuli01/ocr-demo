"""
文档图像增强模块 — 扫描王风格预处理

提供背景漂白、阴影去除、文字增强、自适应二值化、去摩尔纹等功能，
专为 OCR 前处理优化，提升文字可识别性。

依赖: numpy, opencv-python, scikit-image, scipy
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import exposure, morphology


def ensure_float01(img: np.ndarray) -> np.ndarray:
    """确保图像为 float64 [0, 1] 范围"""
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0, 1)


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """确保图像为 uint8 [0, 255] 范围"""
    if img.dtype == np.float64 or img.dtype == np.float32:
        if img.max() <= 1.0:
            img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


# ── 1. 背景漂白 / 光照归一化 ─────────────────────────────


def estimate_background(img_gray: np.ndarray, kernel_size: int = 61) -> np.ndarray:
    """估计文档图像的背景光照场

    使用超大核形态学闭运算估计背景亮度分布，
    适用于文档扫描件的背景均匀化。

    Args:
        img_gray: (H, W) float64 [0, 1] 灰度图
        kernel_size: 形态学核大小（奇数），越大背景估计越平滑

    Returns:
        背景光照场 (H, W) float64 [0, 1]
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # 闭运算：先膨胀后腐蚀，填补文字区域
    background = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    # 高斯模糊平滑
    background = cv2.GaussianBlur(background, (k, k), k * 0.3)

    return np.clip(background, 0.01, 1.0)


def background_whitening(img_rgb: np.ndarray, strength: float = 1.0,
                         kernel_size: int = 61) -> np.ndarray:
    """背景漂白 — 文档扫描件背景均匀化

    通过估计背景光照场并归一化，消除扫描件的阴影、黄斑、不均匀光照。
    效果类似扫描王的"增强"或"增亮"功能。

    Args:
        img_rgb: (H, W, 3) float64 [0, 1]
        strength: 漂白强度 [0, 1]，0=不处理，1=完全均匀
        kernel_size: 背景估计核大小

    Returns:
        漂白后的 (H, W, 3) float64 [0, 1]
    """
    if strength <= 0:
        return img_rgb

    img = ensure_float01(img_rgb).copy()
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    # 估计背景
    bg = estimate_background(gray, kernel_size)

    # 除法归一化: 去除光照不均匀
    # 背景亮 => 不变; 背景暗 => 提亮
    normalized = gray / bg
    normalized = np.clip(normalized, 0, 1)

    # 拉伸到 [0, 1] 充分利用动态范围
    lo, hi = np.percentile(normalized[normalized > 0], (1, 99))
    if hi > lo:
        normalized = (normalized - lo) / (hi - lo)
    normalized = np.clip(normalized, 0, 1)

    # 按强度混合
    result = img.copy()
    for c in range(3):
        channel_bg = cv2.GaussianBlur(img[:, :, c], (kernel_size, kernel_size),
                                      kernel_size * 0.3)
        corrected = img[:, :, c] / np.clip(channel_bg, 0.01, 1.0)
        corrected = np.clip(corrected, 0, 1)
        # 拉伸
        lo_c, hi_c = np.percentile(corrected[corrected > 0], (1, 99))
        if hi_c > lo_c:
            corrected = (corrected - lo_c) / (hi_c - lo_c)
        result[:, :, c] = (1 - strength) * img[:, :, c] + strength * corrected

    return np.clip(result, 0, 1)


# ── 2. 阴影去除 ─────────────────────────────────────────


def remove_shadow(img_rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """去除文档阴影

    用超大核高斯模糊估计阴影场，通过除法校正消除阴影。
    比 background_whitening 的核更大，专门处理大范围阴影。

    Args:
        img_rgb: (H, W, 3) float64 [0, 1]
        strength: 处理强度 [0, 1]

    Returns:
        去阴影后的 (H, W, 3) float64 [0, 1]
    """
    if strength <= 0:
        return img_rgb

    img = ensure_float01(img_rgb).copy()
    h, w = img.shape[:2]

    # 自适应核大小（基于图像尺寸）
    k = max(151, min(h, w) // 4)
    if k % 2 == 0:
        k += 1

    result = np.zeros_like(img)
    for c in range(3):
        # 超大核高斯模糊估计阴影
        shadow_map = cv2.GaussianBlur(img[:, :, c], (k, k), k * 0.5)
        # 除法校正
        corrected = img[:, :, c] / np.clip(shadow_map, 0.01, 1.0)
        # 自适应拉伸
        lo, hi = np.percentile(corrected[corrected > 0], (1, 99))
        if hi > lo:
            corrected = (corrected - lo) / (hi - lo)
        result[:, :, c] = (1 - strength) * img[:, :, c] + strength * np.clip(corrected, 0, 1)

    return np.clip(result, 0, 1)


# ── 3. 自适应对比度增强 ─────────────────────────────────


def clahe_enhance(img_rgb: np.ndarray, clip_limit: float = 3.0,
                  tile_size: int = 8) -> np.ndarray:
    """CLAHE 自适应对比度增强

    限制对比度自适应直方图均衡化，增强局部对比度
    同时避免过度放大噪声。

    Args:
        img_rgb: (H, W, 3) float64 [0, 1]
        clip_limit: 对比度限制，越大对比度越强
        tile_size: 分块大小

    Returns:
        增强后的 (H, W, 3) float64 [0, 1]
    """
    img_u8 = ensure_uint8(img_rgb)

    # 转 LAB 色彩空间，只在 L 通道做 CLAHE
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return ensure_float01(result)


def adaptive_stretch(img_rgb: np.ndarray, low_pct: float = 0.5,
                     high_pct: float = 99.5) -> np.ndarray:
    """自适应对比度拉伸

    去除极端亮暗像素后做直方图拉伸。

    Args:
        img_rgb: (H, W, 3) float64 [0, 1]
        low_pct: 低端百分位
        high_pct: 高端百分位

    Returns:
        拉伸后的 (H, W, 3) float64 [0, 1]
    """
    img = ensure_float01(img_rgb).copy()
    result = np.zeros_like(img)

    for c in range(3):
        ch = img[:, :, c]
        lo, hi = np.percentile(ch, (low_pct, high_pct))
        if hi > lo:
            ch = (ch - lo) / (hi - lo)
        result[:, :, c] = np.clip(ch, 0, 1)

    return result


# ── 4. 文字增强 / 自适应二值化 ──────────────────────────


def sauvola_threshold(img_gray: np.ndarray, window_size: int = 31,
                      k: float = 0.2, r: float = 128) -> np.ndarray:
    """Sauvola 局部自适应阈值二值化

    比全局 Otsu 更好处理光照不均的文档图像。
    效果类似扫描王的"增强"模式。

    Args:
        img_gray: (H, W) float64 [0, 1] 灰度图
        window_size: 局部窗口大小（奇数）
        k: Sauvola 参数（默认 0.2），越大阈值越低
        r: 动态范围（默认 128）

    Returns:
        二值图 (H, W) uint8 {0, 255}
    """
    if window_size % 2 == 0:
        window_size += 1

    img = ensure_uint8(img_gray).astype(np.float64)

    # 局部均值 + 局部标准差
    mean = cv2.blur(img, (window_size, window_size))
    sq_mean = cv2.blur(img ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))

    # Sauvola 阈值
    threshold = mean * (1 + k * (std / r - 1))
    binary = (img >= threshold).astype(np.uint8) * 255

    return binary


def text_enhance_binary(img_rgb: np.ndarray, method: str = 'sauvola',
                        window_size: int = 31, k: float = 0.2) -> np.ndarray:
    """文字增强：灰度化 → 自适应二值化 → 可选反色

    Args:
        img_rgb: (H, W, 3) float64 [0, 1]
        method: 'sauvola' | 'otsu'
        window_size: Sauvola 窗口大小
        k: Sauvola 参数

    Returns:
        RGB 三通道图像 (H, W, 3) float64 [0, 1]，文字黑底白字或白底黑字
    """
    img = ensure_float01(img_rgb)
    gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])
    gray = np.clip(gray, 0, 1)

    if method == 'sauvola':
        binary = sauvola_threshold(gray, window_size, k)
    else:
        gray_u8 = ensure_uint8(gray)
        _, binary = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 返回 3 通道
    result = np.stack([binary, binary, binary], axis=2)
    return ensure_float01(result)


def text_sharpening(img_rgb: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """文字锐化 — unsharp mask

    增强文字边缘，使文字更清晰锐利。

    Args:
        img_rgb: (H, W, 3) float64 [0, 1]
        strength: 锐化强度 [0, 1]

    Returns:
        锐化后的 (H, W, 3) float64 [0, 1]
    """
    if strength <= 0:
        return img_rgb

    img = ensure_float01(img_rgb)
    # 高斯模糊
    blurred = np.zeros_like(img)
    for c in range(3):
        blurred[:, :, c] = cv2.GaussianBlur(img[:, :, c], (0, 0), 1.0)

    # USM: 原图 + strength * (原图 - 模糊)
    result = img + strength * (img - blurred)
    return np.clip(result, 0, 1)


# ── 5. 去摩尔纹（FFT 频域滤波）────────────────────────


def moire_removal_fft(img_rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """FFT 频域去摩尔纹

    文档扫描/拍照时产生的摩尔纹在频域表现为周期性峰值。
    通过检测并抑制这些峰值来去除摩尔纹。

    Args:
        img_rgb: (H, W, 3) float64 [0, 1]
        strength: 处理强度 [0, 1]

    Returns:
        去摩尔纹后的 (H, W, 3) float64 [0, 1]
    """
    if strength <= 0:
        return img_rgb

    img = ensure_float01(img_rgb).copy()
    h, w = img.shape[:2]

    # 对每个通道的亮度分量处理
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    # 构建频域掩码：抑制中高频的周期性峰值
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    # 中心低频区域（保留）
    inner_radius = min(h, w) * 0.08
    # 高频区域（可能包含摩尔纹）
    outer_radius = min(h, w) * 0.4

    # 环形区域：内半径到外半径之间
    mask = np.ones((h, w), dtype=np.float64)

    # 中高频区域的峰值抑制
    ring = (dist > inner_radius) & (dist < outer_radius)

    # 在这个环形区域检测并抑制异常峰值
    mag_ring = magnitude[ring]
    if len(mag_ring) > 0:
        median_mag = np.median(mag_ring)
        std_mag = np.std(mag_ring)
        # 找出异常高的点（摩尔纹峰值）
        threshold = median_mag + 2.5 * std_mag * (1.5 - strength * 0.5)

        # 标记峰值位置
        peak_mask = np.zeros((h, w), dtype=bool)
        peak_mask[ring] = magnitude[ring] > threshold

        # 对峰值区域应用抑制
        mask[peak_mask] = 1.0 - 0.7 * strength

    # 应用掩码
    fshift_filtered = fshift * mask

    # 逆变换
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    # 归一化
    img_filtered = np.clip(img_filtered, 0, 1)

    # 按强度混合到原图
    result = img.copy()
    for c in range(3):
        result[:, :, c] = (1 - strength) * img[:, :, c] + strength * img_filtered

    return np.clip(result, 0, 1)


# ── 6. 综合管线 ─────────────────────────────────────────


def document_enhance_pipeline(
    img_rgb: np.ndarray,
    whitening: float = 0.0,
    shadow_removal: float = 0.0,
    clahe: float = 0.0,
    sharpening: float = 0.0,
    binarize: bool = False,
    moire: float = 0.0,
    **kwargs
) -> np.ndarray:
    """综合文档增强管线

    按顺序执行：去摩尔纹 → 阴影去除 → 背景漂白 → CLAHE → 锐化 → 二值化

    Args:
        img_rgb: (H, W, 3) float64 [0, 1]
        whitening: 背景漂白强度 [0, 1]
        shadow_removal: 阴影去除强度 [0, 1]
        clahe: CLAHE 对比度增强 clip_limit（0=跳过）
        sharpening: 锐化强度 [0, 1]
        binarize: 是否二值化
        moire: 去摩尔纹强度 [0, 1]
        **kwargs: 其他参数

    Returns:
        增强后的 (H, W, 3) float64 [0, 1]
    """
    img = ensure_float01(img_rgb).copy()

    # 1. 去摩尔纹（先做，避免增强后摩尔纹更明显）
    if moire > 0:
        img = moire_removal_fft(img, moire)

    # 2. 阴影去除
    if shadow_removal > 0:
        img = remove_shadow(img, shadow_removal)

    # 3. 背景漂白
    if whitening > 0:
        kernel_size = kwargs.get('whitening_kernel', 61)
        img = background_whitening(img, whitening, kernel_size)

    # 4. CLAHE 对比度增强
    if clahe > 0:
        tile_size = kwargs.get('clahe_tile', 8)
        img = clahe_enhance(img, clip_limit=clahe, tile_size=tile_size)

    # 5. 锐化
    if sharpening > 0:
        img = text_sharpening(img, sharpening)

    # 6. 二值化（最后做）
    if binarize:
        method = kwargs.get('binary_method', 'sauvola')
        window_size = kwargs.get('binary_window', 31)
        img = text_enhance_binary(img, method=method, window_size=window_size)

    return np.clip(img, 0, 1)
