# syntax=docker/dockerfile:1
FROM python:3.12-slim

LABEL maintainer="OCR Demo"
LABEL description="SiliconFlow OCR 测试台 + PP-OCRv6 本地识别"

# 镜像加速（中国用户指定国内源，默认清华源）
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# Streamlit 配置
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50
ENV PADDLEX_DISABLE_MODEL_SOURCE_CHECK=True

# 安装系统依赖（OpenCV / PaddlePaddle 所需）
RUN sed -i "s/deb.debian.org/${APT_MIRROR}/g" /etc/apt/sources.list.d/debian.sources 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖文件并安装（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建运行时目录
RUN mkdir -p _temp && chmod 777 _temp

EXPOSE 8501

# 健康检查（首次启动需等模型下载，约 60s）
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.fileWatcherType", "none"]
