# SiliconFlow OCR 测试台 + 图像修复预处理

基于 Streamlit 的 OCR 在线测试工具，集成经典图像修复算法作为 OCR 前预处理步骤。

## 功能

- **多模型 OCR**：支持 PaddleOCR-VL、DeepSeek-OCR、Qwen3-VL、GLM-4.5V 等多模型对比
- **图像修复预处理**：OCR 前对模糊/噪声图像进行自动修复，提升识别质量
- **退化检测**：自动分析噪声类型、模糊程度、椒盐噪声
- **前后对比**：原图与修复效果并排对比查看
- **管道表输出**：自动将 OCR 结果规整为 Markdown 管道表格式
- **历史记录**：保存识别历史，含预处理标记

## 快速开始

```bash
# 安装依赖
cd py/ocr-demo
uv pip install -r requirements.txt

# 启动
uv run streamlit run app.py
```

打开 http://localhost:8501

## Docker 部署

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down

# 首次启动需等待模型下载（约 1-2 分钟）
docker compose logs -f | grep "Uvicorn server started"
```

模型缓存持久化在 Docker volumes 中，重启/更新容器不需要重新下载。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SILICONFLOW_API_KEY` | — | SiliconFlow API 密钥（云端模型需配置） |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | 50 | 上传文件大小限制 (MB) |

## 项目结构

```
py/ocr-demo/
├── app.py                          # Streamlit 主应用
├── modules/
│   ├── restoration/                # 图像修复算法模块
│   │   ├── __init__.py             # 公共 API
│   │   ├── bm3d.py                 # 自定义 BM3D 去噪
│   │   ├── filters.py              # 维纳滤波 / RL 反卷积 / PSF 估计
│   │   └── utils.py                # 退化检测 / 指标 / 增强管线
│   └── document_enhancement.py     # 文档增强（扫描王风格）
├── Dockerfile                      # Docker 构建文件
├── docker-compose.yml              # Docker Compose 配置
├── _sample/                        # 测试样本图片
├── _temp/                          # 临时文件
└── requirements.txt                # Python 依赖
```

## 图像增强技术参考

图像修复算法移植自 [VisionRestorePro-Classical_Image_Restoration_System](https://github.com/C-loud-Nine/VisionRestorePro-Classical_Image_Restoration_System-IPCV_Project) (MIT License)，包含以下经典算法：

| 算法 | 用途 | 参考 |
|------|------|------|
| **BM3D** | 高斯噪声去除（自实现，无外部依赖） | Dabov et al., 2007 |
| **维纳滤波** | 去模糊/反卷积 | Wiener, 1949 |
| **Richardson-Lucy 反卷积 + TV 正则化** | 严重模糊恢复 | Richardson, 1972; Lucy, 1974 |
| **PSF 估计** | 运动模糊参数估计（Radon 变换 + 倒频谱分析） | — |
| **退化检测** | 自动分析噪声类型、模糊程度 | — |
| **小波去噪** | 多尺度噪声去除（PyWavelets） | — |

算法模块位于 `modules/restoration/`，与 Streamlit UI 解耦，可独立使用：

```python
from modules.restoration import restore_image_with_degradation_awareness
import cv2, numpy as np

img = cv2.imread("blurry.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
restored = restore_image_with_degradation_awareness(img_rgb)
```


## 依赖

- `streamlit` — Web UI 框架
- `openai` — SiliconFlow API 客户端
- `opencv-python`, `numpy`, `scipy` — 图像处理
- `scikit-image` — SSIM/PSNR 指标、Radon 变换
- `PyWavelets` — 小波去噪

## 使用提示

1. 上传图片或点击测试样本缩略图
2. 左侧栏勾选"启用修复预处理"
3. 点击"自动检测退化"分析图像质量
4. 调节参数或直接"应用修复"
5. 选择模型后点击"开始识别"

## License

MIT
