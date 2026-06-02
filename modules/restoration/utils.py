"""图像质量检测、退化分析、指标计算和增强工具。"""

import numpy as np
from scipy.ndimage import sobel, laplace, median_filter, gaussian_filter


# ── 退化检测 ────────────────────────────────────────────────────────

def analyze_noise_pattern(gray_img: np.ndarray) -> dict:
    """分析噪声模式

    Returns:
        dict with keys: noise_type, noise_level, confidence
    """
    gray = gray_img.astype(np.float64)
    lap = laplace(gray)
    noise_var = np.var(lap) * 0.5
    noise_std = np.sqrt(max(noise_var, 1e-8)) / 255.0

    salt_ratio = np.sum(gray > 245) / gray.size
    pepper_ratio = np.sum(gray < 10) / gray.size

    if salt_ratio > 0.01 and pepper_ratio > 0.01:
        noise_type = "salt_pepper"
    elif salt_ratio > 0.02:
        noise_type = "salt"
    elif pepper_ratio > 0.02:
        noise_type = "pepper"
    elif noise_std > 0.02:
        noise_type = "gaussian"
    else:
        noise_type = "low"

    level = min(1.0, noise_std * 10)
    return {
        "noise_type": noise_type,
        "noise_level": level,
        "noise_std": noise_std,
        "salt_ratio": salt_ratio,
        "pepper_ratio": pepper_ratio,
        "confidence": min(1.0, level * 2),
    }


def analyze_blur_severity(gray_img: np.ndarray) -> dict:
    """分析模糊程度

    Returns:
        dict with keys: is_blurry, blur_severity, blur_type
    """
    gray = gray_img.astype(np.float64) / 255.0
    lap_var = np.var(laplace(gray))

    if lap_var < 0.002:
        severity = 0.9
        is_blurry = True
    elif lap_var < 0.005:
        severity = 0.7
        is_blurry = True
    elif lap_var < 0.01:
        severity = 0.4
        is_blurry = True
    elif lap_var < 0.02:
        severity = 0.2
        is_blurry = False
    else:
        severity = 0.0
        is_blurry = False

    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    gx_mean = np.mean(np.abs(gx))
    gy_mean = np.mean(np.abs(gy))

    if gx_mean > 2 * gy_mean:
        blur_type = "vertical_motion"
    elif gy_mean > 2 * gx_mean:
        blur_type = "horizontal_motion"
    else:
        blur_type = "gaussian"

    return {
        "is_blurry": is_blurry,
        "blur_severity": severity,
        "blur_type": blur_type,
        "laplacian_variance": lap_var,
        "confidence": min(1.0, severity * 1.2),
    }


def detect_salt_pepper_noise(gray_img: np.ndarray) -> dict:
    """检测椒盐噪声"""
    gray = gray_img.astype(np.float64)
    salt = np.sum(gray > 245) / gray.size
    pepper = np.sum(gray < 10) / gray.size
    total = salt + pepper
    return {
        "has_sp": total > 0.005,
        "salt_ratio": salt,
        "pepper_ratio": pepper,
        "total_ratio": total,
        "severity": min(1.0, total * 5),
    }


def detect_degradation_type(img_rgb: np.ndarray) -> dict:
    """综合检测图像退化类型

    组合噪声检测、模糊检测、椒盐检测。

    Returns:
        dict with keys: noise_type, noise_level, is_blurry, blur_severity,
                       blur_type, has_sp, salt_ratio, pepper_ratio,
                       overall_quality, recommendations
    """
    h, w = img_rgb.shape[:2]
    gray = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2])
    if gray.max() <= 1.0:
        gray = gray * 255.0

    noise = analyze_noise_pattern(gray)
    blur = analyze_blur_severity(gray)
    sp = detect_salt_pepper_noise(gray)

    overall = max(noise["noise_level"] * 0.4, blur["blur_severity"] * 0.6, sp["severity"] * 0.3)
    if overall < 0.2:
        quality = "good"
    elif overall < 0.5:
        quality = "fair"
    else:
        quality = "poor"

    recs = []
    if noise["noise_level"] > 0.1:
        recs.append(f"denoise({noise['noise_type']})")
    if blur["is_blurry"]:
        recs.append(f"deblur({blur['blur_type']})")
    if sp["has_sp"]:
        recs.append("remove_salt_pepper")

    return {
        **noise,
        **blur,
        **sp,
        "overall_quality": quality,
        "overall_score": overall,
        "recommendations": recs,
    }


# ── 自适应处理 ──────────────────────────────────────────────────────

def analyze_texture_simple(img_rgb: np.ndarray) -> dict:
    """简单纹理分析，推荐参数"""
    gray = np.mean(img_rgb, axis=2).astype(np.float64)
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    energy = np.mean(grad_mag)
    if energy < 0.05:
        tv_weight = 0.05
        sharpening = 0.3
        noise_sigma = 0.05
    elif energy < 0.1:
        tv_weight = 0.03
        sharpening = 0.5
        noise_sigma = 0.03
    else:
        tv_weight = 0.01
        sharpening = 0.8
        noise_sigma = 0.01

    return {
        "texture_energy": energy,
        "recommended_tv_weight": tv_weight,
        "recommended_sharpening": sharpening,
        "recommended_noise_sigma": noise_sigma,
    }


def adaptive_denoising(img_rgb: np.ndarray, degradation_info: dict, params: dict = None) -> np.ndarray:
    """自适应去噪：椒盐→中值滤波；高斯→BM3D"""
    if params is None:
        params = {}
    img = np.clip(img_rgb, 0, 1).astype(np.float64)

    if degradation_info.get("has_sp"):
        kernel_size = params.get("median_kernel", 3)
        for c in range(3):
            img[:, :, c] = median_filter(img[:, :, c], size=kernel_size)

    if degradation_info.get("noise_level", 0) > 0.05:
        from .bm3d import bm3d_rgb_denoising
        noise_level = degradation_info["noise_level"]
        img = bm3d_rgb_denoising(img, noise_level)

    return np.clip(img, 0, 1)


def wavelet_rgb_denoising(img_rgb: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """小波去噪（每个通道独立）"""
    import pywt
    img = np.clip(img_rgb, 0, 1).astype(np.float64)
    out = np.zeros_like(img)
    threshold = noise_level * 0.5

    for c in range(3):
        coeffs = pywt.wavedec2(img[:, :, c], 'db4', level=3)
        coeffs_list = list(coeffs)
        for j in range(1, len(coeffs_list)):
            detail = coeffs_list[j]
            detail = tuple(
                pywt.threshold(d, threshold * np.std(d), mode='soft')
                for d in detail
            )
            coeffs_list[j] = detail
        out[:, :, c] = pywt.waverec2(tuple(coeffs_list), 'db4')[:img.shape[0], :img.shape[1]]
    return np.clip(out, 0, 1)


def adaptive_deblurring(img_rgb: np.ndarray, degradation_info: dict, params: dict = None) -> np.ndarray:
    """自适应去模糊"""
    if params is None:
        params = {}
    img = np.clip(img_rgb, 0, 1).astype(np.float64)
    severity = degradation_info.get("blur_severity", 0)

    if severity < 0.2:
        return img

    gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])
    blur_type = degradation_info.get("blur_type", "gaussian")
    psf_size = params.get("psf_size", 21)

    if blur_type == "horizontal_motion":
        from .filters import motion_blur_psf
        psf = motion_blur_psf(psf_size, angle=0, length=min(int(psf_size * 0.7), psf_size - 2))
    elif blur_type == "vertical_motion":
        from .filters import motion_blur_psf
        psf = motion_blur_psf(psf_size, angle=90, length=min(int(psf_size * 0.7), psf_size - 2))
    else:
        from .filters import gaussian_psf
        psf = gaussian_psf(psf_size, sigma=max(1.0, severity * 5))

    from .filters import wiener_deconvolution_rgb
    balance = params.get("wiener_balance", max(0.001, severity * 0.05))
    result = wiener_deconvolution_rgb(img, psf, balance)

    if severity > 0.7:
        from .filters import rl_tv_deconvolution
        gray_result = (0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2])
        rl_result = rl_tv_deconvolution(
            gray_result, psf,
            init=gray_result,
            max_iters=20,
            tv_interval=5,
            tv_weight=params.get("tv_weight", 0.02),
        )
        scale = np.std(rl_result) / max(np.std(gray_result), 1e-8)
        for c in range(3):
            result[:, :, c] = np.clip(
                (result[:, :, c] - np.mean(result[:, :, c])) * scale + np.mean(rl_result), 0, 1
            )

    return np.clip(result, 0, 1)


def enhance_salt_pepper_recovery(img_rgb: np.ndarray, degradation_info: dict) -> np.ndarray:
    """椒盐噪声后处理：双边滤波清理"""
    if not degradation_info.get("has_sp", False):
        return img_rgb
    from skimage.restoration import denoise_bilateral
    img = np.clip(img_rgb, 0, 1).astype(np.float64)
    return denoise_bilateral(img, sigma_color=0.05, sigma_spatial=2)


def restore_image_with_degradation_awareness(img_rgb: np.ndarray, params: dict = None) -> np.ndarray:
    """完整修复管线：退化感知的自动图像恢复

    流程：退化检测 → 去噪 → 去模糊 → 锐化 → 椒盐后处理

    Args:
        img_rgb: (H, W, 3) float64，范围 [0, 1]
        params: 可选参数覆盖

    Returns:
        修复后的 (H, W, 3) float64
    """
    if params is None:
        params = {}
    img = np.clip(img_rgb, 0, 1).astype(np.float64)
    degradation = detect_degradation_type(img)
    texture = analyze_texture_simple(img)

    # 1. 去噪
    denoise_params = {
        "median_kernel": params.get("median_kernel", 3),
    }
    img = adaptive_denoising(img, degradation, denoise_params)

    # 2. 去模糊
    deblur_params = {
        "psf_size": params.get("psf_size", 21),
        "wiener_balance": params.get("wiener_balance", 0.01),
        "tv_weight": params.get("tv_weight", texture["recommended_tv_weight"]),
    }
    img = adaptive_deblurring(img, degradation, deblur_params)

    # 3. 锐化
    sharpening = params.get("sharpening", texture["recommended_sharpening"])
    if sharpening > 0:
        gray = np.mean(img, axis=2)
        blur_gray = gaussian_filter(gray, sigma=1)
        detail = gray - blur_gray
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c] + sharpening * detail, 0, 1)

    # 4. 椒盐后处理
    img = enhance_salt_pepper_recovery(img, degradation)

    return img


# ── 指标计算 ────────────────────────────────────────────────────────

def compute_comprehensive_metrics(
    gt_rgb: np.ndarray,
    input_rgb: np.ndarray,
    restored_rgb: np.ndarray,
) -> dict:
    """计算 PSNR / SSIM / MSE / Sharpness

    Args:
        gt_rgb: 参考图像 (H, W, 3)
        input_rgb: 输入图像 (H, W, 3)
        restored_rgb: 修复结果 (H, W, 3)

    Returns:
        metrics dict
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    def _to_gray(img):
        return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])

    def _sharpness(img):
        return np.var(laplace(img))

    gt_g = _to_gray(np.clip(gt_rgb, 0, 1))
    in_g = _to_gray(np.clip(input_rgb, 0, 1))
    res_g = _to_gray(np.clip(restored_rgb, 0, 1))

    psnr_input = peak_signal_noise_ratio(gt_g, in_g, data_range=1.0)
    psnr_restored = peak_signal_noise_ratio(gt_g, res_g, data_range=1.0)
    ssim_input = structural_similarity(gt_g, in_g, data_range=1.0)
    ssim_restored = structural_similarity(gt_g, res_g, data_range=1.0)
    mse_input = np.mean((gt_g - in_g) ** 2)
    mse_restored = np.mean((gt_g - res_g) ** 2)
    sharp_input = _sharpness(in_g)
    sharp_restored = _sharpness(res_g)

    return {
        "psnr_input": psnr_input,
        "psnr_restored": psnr_restored,
        "psnr_improvement": psnr_restored - psnr_input,
        "ssim_input": ssim_input,
        "ssim_restored": ssim_restored,
        "ssim_improvement": ssim_restored - ssim_input,
        "mse_input": mse_input,
        "mse_restored": mse_restored,
        "sharpness_input": sharp_input,
        "sharpness_restored": sharp_restored,
    }


def generate_synthetic_sample(size: int = 256, sigma_noise: float = 0.05) -> np.ndarray:
    """生成含可控退化的合成测试图"""
    np.random.seed(42)
    img = np.random.rand(size, size, 3).astype(np.float64) * 0.5 + 0.25
    img[30:70, 30:70] = 0.8
    img[100:140, 100:140] = 0.2
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=2)
    noise = np.random.randn(size, size, 3) * sigma_noise
    return np.clip(img + noise, 0, 1)


def ensure_image_range(img: np.ndarray) -> np.ndarray:
    """确保图像在 [0, 1] float64 范围"""
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0, 1)


def safe_image_display(img: np.ndarray, caption: str = "", width: int = None) -> np.ndarray:
    """安全处理图像用于显示（转 uint8）"""
    img = ensure_image_range(img)
    return (img * 255).astype(np.uint8)
