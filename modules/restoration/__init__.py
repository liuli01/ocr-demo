"""图像修复预处理模块 — 基于 VisionRestorePro 算法

提供 BM3D 去噪、维纳滤波/RL 反卷积、退化检测、质量评估等功能。
"""

from .bm3d import (
    bm3d_rgb_denoising,
    bm3d_denoising,
)
from .filters import (
    wiener_deconvolution_rgb,
    rl_tv_deconvolution,
    estimate_motion_psf_robust,
    motion_blur_psf,
    gaussian_psf,
    disk_psf,
    estimate_snr,
    adaptive_wiener_balance,
)
from .utils import (
    detect_degradation_type,
    analyze_noise_pattern,
    analyze_blur_severity,
    detect_salt_pepper_noise,
    compute_comprehensive_metrics,
    adaptive_denoising,
    adaptive_deblurring,
    restore_image_with_degradation_awareness,
    analyze_texture_simple,
    generate_synthetic_sample,
)

__all__ = [
    "bm3d_rgb_denoising",
    "bm3d_denoising",
    "wiener_deconvolution_rgb",
    "rl_tv_deconvolution",
    "estimate_motion_psf_robust",
    "motion_blur_psf",
    "gaussian_psf",
    "disk_psf",
    "detect_degradation_type",
    "analyze_noise_pattern",
    "analyze_blur_severity",
    "detect_salt_pepper_noise",
    "compute_comprehensive_metrics",
    "adaptive_denoising",
    "adaptive_deblurring",
    "restore_image_with_degradation_awareness",
    "analyze_texture_simple",
    "estimate_snr",
    "adaptive_wiener_balance",
    "generate_synthetic_sample",
]
