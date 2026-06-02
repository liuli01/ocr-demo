"""图像恢复滤波器：维纳滤波、Richardson-Lucy 反卷积、PSF 估计。"""

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import median_filter, laplace


# ── FFT 卷积工具 ────────────────────────────────────────────────────

def fft_conv(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT 加速卷积，same 模式"""
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = h + kh - 1, w + kw - 1
    fft_img = np.fft.fft2(img, s=(pad_h, pad_w))
    fft_k = np.fft.fft2(kernel, s=(pad_h, pad_w))
    conv = np.real(np.fft.ifft2(fft_img * fft_k))
    y_start = (kh - 1) // 2
    x_start = (kw - 1) // 2
    return conv[y_start:y_start + h, x_start:x_start + w]


def circular_mask(h: int, w: int, radius: float = None) -> np.ndarray:
    """创建圆形掩码，用于减少边界伪影"""
    if radius is None:
        radius = min(h, w) * 0.45
    Y, X = np.ogrid[:h, :w]
    center = (h / 2, w / 2)
    dist = np.sqrt((Y - center[0]) ** 2 + (X - center[1]) ** 2)
    mask = np.clip(1 - (dist / radius) ** 2, 0, 1)
    return mask


# ── PSF 生成 ────────────────────────────────────────────────────────

def motion_blur_psf(size: int = 21, angle: float = 45, length: int = 15) -> np.ndarray:
    """运动模糊 PSF"""
    psf = np.zeros((size, size))
    center = size // 2
    angle_rad = np.deg2rad(angle)
    half_len = length // 2
    for i in range(-half_len, half_len + 1):
        x = int(round(center + i * np.cos(angle_rad)))
        y = int(round(center + i * np.sin(angle_rad)))
        if 0 <= x < size and 0 <= y < size:
            psf[y, x] = 1
    total = np.sum(psf)
    return psf / total if total > 0 else psf


def gaussian_psf(size: int = 21, sigma: float = 3.0) -> np.ndarray:
    """高斯模糊 PSF"""
    ax = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    x, y = np.meshgrid(ax, ax)
    psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return psf / np.sum(psf)


def disk_psf(radius: int = 5) -> np.ndarray:
    """圆盘（平均）模糊 PSF"""
    size = 2 * radius + 1
    Y, X = np.ogrid[:size, :size]
    center = radius
    mask = np.sqrt((Y - center) ** 2 + (X - center) ** 2) <= radius
    psf = mask.astype(np.float64)
    return psf / np.sum(psf)


# ── PSF 估计 ────────────────────────────────────────────────────────

def estimate_motion_psf_robust(image_gray: np.ndarray, max_len: int = 120) -> tuple:
    """鲁棒的运动模糊 PSF 估计：Radon 变换 → 角度，倒频谱 → 长度

    Returns:
        (psf, psf_len, psf_angle, confidence)
    """
    from skimage.transform import radon

    small = image_gray.copy()

    # 梯度增强
    grad_x = np.abs(np.diff(small, axis=1))
    grad_y = np.abs(np.diff(small, axis=0))
    edges = np.zeros_like(small)
    edges[:, :-1] += grad_x
    edges[:-1, :] += grad_y

    # Radon 变换找角度
    theta = np.arange(0, 180, 1)
    sinogram = radon(edges, theta=theta, circle=False)
    variances = np.var(sinogram, axis=0)
    best_idx = np.argmax(variances)
    psf_angle = theta[best_idx]

    # 倒频谱分析找长度
    f = np.log(np.abs(np.fft.fft2(small)) + 1e-8)
    cepstrum = np.abs(np.fft.ifft2(f))
    cepstrum = np.fft.fftshift(cepstrum)

    cy, cx = cepstrum.shape[0] // 2, cepstrum.shape[1] // 2
    angle_rad = np.deg2rad(psf_angle)
    profile = []
    for d in range(1, min(max_len, min(cepstrum.shape) // 2)):
        y = int(cy + d * np.sin(angle_rad))
        x = int(cx + d * np.cos(angle_rad))
        if 0 <= y < cepstrum.shape[0] and 0 <= x < cepstrum.shape[1]:
            profile.append(cepstrum[y, x])
        else:
            break

    if profile:
        profile = np.array(profile, dtype=np.float64)
        gradient = np.diff(profile)
        zero_crossings = np.where(np.diff(np.signbit(gradient)))[0]
        if len(zero_crossings) > 0:
            psf_len = int(zero_crossings[0]) + 2
        else:
            psf_len = np.argmax(profile) + 1
    else:
        psf_len = 10

    psf_len = max(3, min(psf_len, max_len))
    confidence = min(1.0, variances[best_idx] / max(np.mean(variances), 1e-8) * 0.1)

    psf = motion_blur_psf(size=21, angle=psf_angle, length=psf_len)
    return psf, psf_len, psf_angle, confidence


# ── Richardson-Lucy 反卷积 ─────────────────────────────────────────

def rl_tv_deconvolution(
    y_obs: np.ndarray,
    psf: np.ndarray,
    init: np.ndarray | None = None,
    max_iters: int = 30,
    tv_interval: int = 5,
    tv_weight: float = 0.02,
) -> np.ndarray:
    """Richardson-Lucy 反卷积 + 总变分正则化

    Args:
        y_obs: 观测图像 (H, W)
        psf: 点扩散函数
        init: 初始估计（默认 = y_obs）
        max_iters: 最大迭代次数
        tv_interval: TV 正则化间隔
        tv_weight: TV 权重

    Returns:
        恢复后的图像 (H, W)
    """
    y_obs = y_obs.astype(np.float64)
    psf = psf.astype(np.float64)
    psf_rot = np.rot90(psf, 2)

    f = init.astype(np.float64) if init is not None else y_obs.copy()
    eps = 1e-12

    for i in range(max_iters):
        est_conv = fft_conv(f, psf) + eps
        ratio = y_obs / est_conv
        f = f * fft_conv(ratio, psf_rot)
        f = np.clip(f, eps, None)

        if tv_weight > 0 and (i + 1) % tv_interval == 0:
            dx = np.diff(f, axis=1)
            dy = np.diff(f, axis=0)
            dx_pad = np.pad(dx, ((0, 0), (0, 1)))
            dy_pad = np.pad(dy, ((0, 1), (0, 0)))
            grad_norm = np.sqrt(dx_pad ** 2 + dy_pad ** 2) + eps
            div_x = np.diff(dx_pad / grad_norm, axis=1)
            div_y = np.diff(dy_pad / grad_norm, axis=0)
            div_x = np.pad(div_x, ((0, 0), (0, 1)))
            div_y = np.pad(div_y, ((0, 1), (0, 0)))
            div = div_x + div_y
            denom = np.maximum(1 - tv_weight * div, eps)
            f = f / denom
            f = np.clip(f, eps, None)

    return f


# ── 维纳滤波 ────────────────────────────────────────────────────────

def wiener_deconvolution_rgb(
    img_rgb: np.ndarray,
    psf: np.ndarray,
    balance: float = 0.01,
) -> np.ndarray:
    """维纳滤波反卷积（仅处理 Y 通道）

    Args:
        img_rgb: (H, W, 3) 输入 RGB，范围 [0, 1]
        psf: 点扩散函数
        balance: 噪声平衡参数

    Returns:
        (H, W, 3) 恢复后 RGB
    """
    img_rgb = np.clip(img_rgb, 0, 1).astype(np.float64)

    # RGB → YCbCr，仅对 Y 通道做维纳滤波
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 + (-0.168736 * r - 0.331264 * g + 0.5 * b) * 255
    cr = 128 + (0.5 * r - 0.418688 * g - 0.081312 * b) * 255

    # Wiener 滤波
    psf = psf.astype(np.float64) / max(np.sum(psf), 1e-12)
    psf_pad = np.zeros_like(y)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf

    fft_y = np.fft.fft2(y)
    fft_psf = np.fft.fft2(psf_pad)
    fft_psf_conj = np.conj(fft_psf)
    denominator = fft_psf * fft_psf_conj + balance
    fft_restored = (fft_psf_conj / denominator) * fft_y
    restored_y = np.real(np.fft.ifft2(fft_restored))
    restored_y = np.clip(restored_y, 0, 1)

    # YCbCr → RGB
    out = np.zeros_like(img_rgb)
    out[:, :, 0] = restored_y + 0.0 * (cb - 128) / 255 + 1.402 * (cr - 128) / 255
    out[:, :, 1] = restored_y - 0.344136 * (cb - 128) / 255 - 0.714136 * (cr - 128) / 255
    out[:, :, 2] = restored_y + 1.772 * (cb - 128) / 255 + 0.0 * (cr - 128) / 255
    return np.clip(out, 0, 1)


def estimate_snr(y_channel: np.ndarray) -> float:
    """通过中位绝对偏差估计 SNR"""
    noise = laplace(y_channel)
    sigma_noise = np.median(np.abs(noise)) / 0.6745
    sigma_signal = np.std(y_channel)
    return 20 * np.log10(sigma_signal / max(sigma_noise, 1e-8))


def adaptive_wiener_balance(y_channel: np.ndarray, base: float = 0.01) -> float:
    """SNR 自适应维纳平衡参数"""
    snr = estimate_snr(y_channel)
    if snr > 30:
        return base * 0.5
    elif snr > 20:
        return base
    else:
        return base * 2.0
