#!/usr/bin/env python3
"""
SiliconFlow OCR 在线测试 Demo — Streamlit 版
支持多模型对比、参数调节、历史记录

启动：
    cd AIVault-Code/py/ocr-demo
    uv run streamlit run app.py
"""

import base64
import os
# PP-OCRv6 本地模型跳过模型源检测
os.environ.setdefault('PADDLEX_DISABLE_MODEL_SOURCE_CHECK', 'True')
import time
import shutil
import zipfile
import io
from datetime import datetime

import requests
import streamlit as st
from PIL import Image
from openai import OpenAI

# === 图像修复预处理模块 ===
from modules.restoration import (
    restore_image_with_degradation_awareness,
    detect_degradation_type,
    ensure_image_range,
)
import cv2
import numpy as np
import functools

# PP-OCRv6 本地模型标识
PADDLE_LOCAL_MODEL_ID = "_paddle_local_"

# === 常量 ===
try:
    API_KEY = st.secrets.get("API_KEY") or os.getenv("SILICONFLOW_API_KEY") or "sk-yvfxmydciruabzrpxmnaptmrvkgitkjhjjgroibaudfprhrw"
except Exception:
    API_KEY = os.getenv("SILICONFLOW_API_KEY") or "sk-yvfxmydciruabzrpxmnaptmrvkgitkjhjjgroibaudfprhrw"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

MODELS = {
    "PaddleOCR-VL-1.5 (0.9B)": "PaddlePaddle/PaddleOCR-VL-1.5",
    "DeepSeek-OCR (3B)": "deepseek-ai/DeepSeek-OCR",
    "Qwen3-VL-8B": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3-VL-32B": "Qwen/Qwen3-VL-32B-Instruct",
    "GLM-4.5V (106B)": "zai-org/GLM-4.5V",
    "中建电商模型 (1.2B)": "_file_parse_",
    # 本地模型（纯本地运行，无需 API）
    "🏠 PP-OCRv6 Medium (34.5M)": PADDLE_LOCAL_MODEL_ID,
}

FILE_PARSE_URL = "https://ms-9t6wthr2-100032905193-sw.gw.ap-beijing.ti.tencentcs.com/ms-9t6wthr2/file_parse"

DEFAULT_PROMPT = "请识别这张表格中的所有文字，按表格结构逐行输出，不要重复，不要额外说明"
PROMPT_PIPE = (
    "请识别这张表格中的所有文字，"
    "严格按 Markdown 管道表格式输出，用 | 分隔各列，"
    "第二行用 |---|--- 作为分隔线。"
    "不要输出额外说明文字，只输出表格。"
    "保持原始表格结构，空白单元格保留为空。"
)
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "_sample", "测试.jpg")

# === 初始化 session state ===
if "history" not in st.session_state:
    st.session_state.history = []
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = PROMPT_PIPE
if "base_url" not in st.session_state:
    st.session_state.base_url = DEFAULT_BASE_URL
if "custom_model" not in st.session_state:
    st.session_state.custom_model = ""

# 预处理状态
if "restoration_enabled" not in st.session_state:
    st.session_state.restoration_enabled = False
if "restored_image_path" not in st.session_state:
    st.session_state.restored_image_path = None
if "degradation_info" not in st.session_state:
    st.session_state.degradation_info = None
if "restoration_params" not in st.session_state:
    st.session_state.restoration_params = {
        "median_kernel": 3,
        "psf_size": 21,
        "wiener_balance": 0.01,
        "tv_weight": 0.02,
        "sharpening": 0.5,
    }
if "show_comparison" not in st.session_state:
    st.session_state.show_comparison = False

import re

# 默认选中第一张样本
_sample_dir = os.path.join(os.path.dirname(__file__), "_sample")
_first_sample = sorted([
    f for f in os.listdir(_sample_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])[0] if os.path.isdir(_sample_dir) else "测试.jpg"
_first_sample_path = os.path.join(_sample_dir, _first_sample)

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = _first_sample_path


def _normalize_pipe_table(text: str) -> str:
    """规整管道表：所有行列数对齐，分隔线修正"""
    lines = text.strip().split("\n")
    normalized = []
    col_count = 0
    sep_row_idx = None

    # 第一遍：解析行类型，找最大列数
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # 判断是否为分隔行（只含 | - 空格）
        if re.match(r"^\|[\s\-|]+\|$", stripped):
            cells = [c for c in stripped.split("|") if c.strip() and c.strip().startswith("-")]
            sep_row_idx = len(normalized)
            normalized.append(("sep", cells))
        else:
            cells = [c.strip() for c in stripped.split("|")][1:-1]  # 去掉首尾空
            col_count = max(col_count, len(cells))
            normalized.append(("data", cells))

    if col_count == 0:
        return text

    # 第二遍：填充对齐
    out = []
    for kind, cells in normalized:
        if kind == "sep":
            out.append("|" + "|".join("---" for _ in range(col_count)) + "|")
        else:
            padded = cells + [""] * (col_count - len(cells))
            out.append("| " + " | ".join(padded) + " |")
    return "\n".join(out)


def _is_pipe_like(text: str) -> bool:
    """检测文本是否所有非空行都以 | 开头（无分隔线的 pipe 表）"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return len(lines) > 1 and all(l.startswith("|") for l in lines)


def _pipe_to_table(text: str) -> str:
    """将 | 分隔的无分隔线文本转为标准管道表（首行后加分隔线）"""
    lines = text.strip().split("\n")
    rows = []
    col_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|")][1:-1]
        col_count = max(col_count, len(cells))
        rows.append(cells)
    if col_count == 0:
        return text
    # 单列模式（PaddleOCR 逐行输出）：两两配对为 key-value
    if col_count == 1:
        flat = [c[0] for c in rows if c[0]]
        paired = []
        i = 0
        while i < len(flat):
            paired.append([flat[i], flat[i + 1] if i + 1 < len(flat) else ""])
            i += 2
        if paired:
            max_c = max(len(r) for r in paired)
            out = []
            for j, cells in enumerate(paired):
                padded = cells + [""] * (max_c - len(cells))
                out.append("| " + " | ".join(padded) + " |")
                if j == 0:
                    out.append("|" + "|".join("---" for _ in range(max_c)) + "|")
            return "\n".join(out)
    out = []
    for i, cells in enumerate(rows):
        padded = cells + [""] * (col_count - len(cells))
        out.append("| " + " | ".join(padded) + " |")
        if i == 0:  # 首行后加分隔线
            out.append("|" + "|".join("---" for _ in range(col_count)) + "|")
    return "\n".join(out)


def clean_ocr_output(text: str) -> str:
    """清理 OCR 输出：HTML 表格 / 管道表 / 纯文本 → 标准管道表"""
    text = re.sub(r"^.*?(<table|<tr)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<\|LOC_\d+\|>", "", text)
    text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()

    # 情况1：标准管道表（含 |---| 分隔线），规整列数
    if re.search(r"\|[-]+\|", text):
        return _normalize_pipe_table(text)

    # 情况1.5：无分隔线的管道表（所有行 | 开头）
    if _is_pipe_like(text):
        return _pipe_to_table(text)

    # 情况2：DeepSeek-OCR 的 HTML <tr><td> 格式
    if "<tr>" in text:
        rows = re.findall(r"<tr>(.*?)</tr>", text, re.DOTALL)
        pipe_lines = []
        for i, tr in enumerate(rows):
            cells = re.findall(r"<td>(.*?)</td>", tr, re.DOTALL)
            texts = [c.strip() for c in cells]
            pipe_lines.append("| " + " | ".join(texts) + " |")
            if i == 0 and len(rows) > 1:  # 首行后加分隔线
                pipe_lines.append("|" + "|".join("---" for _ in range(len(texts))) + "|")
        return "\n".join(pipe_lines)

    # 情况2.5：纯文本逐行内容 → 单列表格（PaddleOCR 常见输出）
    lines_raw = [l.strip() for l in text.split("\n") if l.strip()]
    if (not text.startswith("|") and "|" not in text
            and len(lines_raw) > 3
            and max(len(l) for l in lines_raw) < 100):
        col_single = max(len(l) for l in lines_raw)
        out = "| 内容 |\n|---|---|\n"
        for l in lines_raw:
            out += f"| {l} |\n"
        return out.strip()

    # 情况3：Tab/空格分隔文本转管道表
    lines = text.split("\n")
    pipe_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cells = re.split(r"\t|  +", line)
        cells = [c.strip() for c in cells if c.strip()]
        if cells:
            pipe_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(pipe_lines) if pipe_lines else text


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "bmp": "bmp"}.get(
        ext.lstrip("."), "jpeg"
    )
    return f"data:image/{mime};base64,{b64}"


def ocr_recognize(image_path: str, model_id: str, prompt: str,
                  temperature: float, max_tokens: int,
                  top_p: float = 0.7, top_k: int = 50,
                  frequency_penalty: float = 0.0,
                  enable_thinking: bool = False,
                  base_url: str = None) -> dict:
    # 本地 PP-OCRv6 走独立路由
    if model_id == PADDLE_LOCAL_MODEL_ID:
        return _ocr_paddle_local(image_path)

    # 文件解析 API 走独立路由
    if model_id == "_file_parse_":
        return _call_file_parse(image_path)

    client = OpenAI(api_key=API_KEY, base_url=base_url or DEFAULT_BASE_URL)
    image_data = encode_image(image_path)

    # DeepSeek-OCR 和 PaddleOCR 不理解管道表指令
    if "DeepSeek-OCR" in model_id or "PaddleOCR" in model_id:
        prompt = ""  # 无 text prompt，匹配 Web 界面行为

    # 构建 content：有 prompt 时包含 text，否则仅图片
    content = [{"type": "image_url", "image_url": {"url": image_data}}]
    if prompt.strip():
        content.insert(0, {"type": "text", "text": prompt})

    start = time.time()
    kwargs = dict(
        model=model_id,
        messages=[{
            "role": "user",
            "content": content,
        }],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
    )
    # 模型特定参数
    extra = {}
    if "DeepSeek" in model_id:
        extra["top_k"] = top_k
    if "Qwen3.5-9B" in model_id:
        extra.update({"top_k": top_k, "enable_thinking": enable_thinking, "min_p": 0, "repetition_penalty": 1})
    if extra:
        kwargs["extra_body"] = extra
    response = client.chat.completions.create(**kwargs)
    elapsed = time.time() - start
    result = response.choices[0].message.content
    usage = response.usage

    return {
        "result": result,
        "result_cleaned": clean_ocr_output(result),
        "elapsed": f"{elapsed:.1f}s",
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "model": model_id,
        "time": datetime.now().strftime("%H:%M:%S"),
    }


def _call_file_parse(image_path: str) -> dict:
    """调用文件解析 API，返回 markdown 结果"""
    start = time.time()
    with open(image_path, "rb") as f:
        resp = requests.post(
            FILE_PARSE_URL,
            files={"files": (os.path.basename(image_path), f, "image/jpeg")},
            data={
                "response_format_zip": "true",
                "return_md": "true",
                "return_images": "true",
            },
            timeout=120,
        )
    elapsed = time.time() - start

    if resp.status_code != 200:
        return {
            "result": f"API 错误: HTTP {resp.status_code}\n{resp.text[:500]}",
            "result_cleaned": f"API 返回错误: HTTP {resp.status_code}",
            "elapsed": f"{elapsed:.1f}s",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": "file_parse",
            "time": datetime.now().strftime("%H:%M:%S"),
        }

    try:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        md_files = [n for n in z.namelist() if n.endswith(".md")]
        md_content = z.read(md_files[0]).decode("utf-8") if md_files else "(未返回 markdown)"
        z.extractall("_temp/file_parse_output")
    except Exception as e:
        md_content = f"解压失败: {e}"

    return {
        "result": md_content,
        "result_cleaned": clean_ocr_output(md_content),
        "elapsed": f"{elapsed:.1f}s",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model": "file_parse",
        "time": datetime.now().strftime("%H:%M:%S"),
    }


# ===== PP-OCRv6 本地 OCR =====

@functools.lru_cache(maxsize=1)
def _get_paddle_ocr():
    """懒加载 PaddleOCR 实例（首次选择本地模型时初始化，之后缓存）"""
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_textline_orientation=True,
        lang='ch',
        text_det_thresh=0.3,
        text_recognition_batch_size=6,
    )


def _ocr_paddle_local(image_path: str) -> dict:
    """使用 PP-OCRv6 本地识别图片，返回与 ocr_recognize 兼容的 dict"""
    start = time.time()
    ocr = _get_paddle_ocr()

    # 大图自动缩放到合理尺寸（PaddleOCR 内部 max_side_limit=4000）
    img_bgr = _cv_imread(image_path)
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        MAX_SIDE = 2000
        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        input_data = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        input_data = image_path  # fallback: 传路径

    result = ocr.predict(input_data, use_textline_orientation=True)
    elapsed = time.time() - start

    if not result or not result[0]:
        raw_output = "(未识别到文字)"
        return {
            "result": raw_output,
            "result_cleaned": raw_output,
            "elapsed": f"{elapsed:.1f}s",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": "PP-OCRv6 Medium (🏠 本地)",
            "time": datetime.now().strftime("%H:%M:%S"),
        }

    # PaddleOCR 3.x 返回 OCRResult 对象（类 dict 结构）
    page_result = result[0]
    rec_texts = page_result.get('rec_texts', [])
    rec_scores = page_result.get('rec_scores', [])
    rec_boxes = page_result.get('rec_boxes', [])

    if not rec_texts:
        raw_output = "(未识别到文字)"
        return {
            "result": raw_output,
            "result_cleaned": raw_output,
            "elapsed": f"{elapsed:.1f}s",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": "PP-OCRv6 Medium (🏠 本地)",
            "time": datetime.now().strftime("%H:%M:%S"),
        }

    # 组合 bbox + text + score，按 Y 坐标排序
    lines = []
    for i, text in enumerate(rec_texts):
        bbox = rec_boxes[i] if i < len(rec_boxes) else None
        conf = rec_scores[i] if i < len(rec_scores) else 0.0
        # bbox 格式（numpy.ndarray）: [x1, y1, x2, y2]（四点坐标压平）
        if bbox is not None and len(bbox) >= 4:
            y_center = (bbox[1] + bbox[3]) / 2  # (y1 + y2) / 2
            x_start = bbox[0]
        else:
            y_center = 0
            x_start = 0
        lines.append((y_center, x_start, bbox, text, conf))

    # Y 坐标排序（自上而下），同一行按 X 排序
    lines.sort(key=lambda x: (round(x[0]), x[1]))

    # 原始输出：每行带坐标和置信度
    raw_lines = []
    for _, _, bbox, text, conf in lines:
        if bbox is not None and len(bbox) >= 4:
            raw_lines.append(
                f"({int(bbox[0])},{int(bbox[1])})-({int(bbox[2])},{int(bbox[3])}) "
                f"\"{text}\" (conf={conf:.2f})"
            )
        else:
            raw_lines.append(f"\"{text}\" (conf={conf:.2f})")
    raw_output = "\n".join(raw_lines)

    # 管道表输出：仅文本 + 置信度
    pipe_lines = ["| 内容 | 置信度 |", "|---|---|"]
    for _, _, _, text, conf in lines:
        pipe_lines.append(f"| {text} | {conf:.2f} |")

    return {
        "result": raw_output,
        "result_cleaned": "\n".join(pipe_lines),
        "elapsed": f"{elapsed:.1f}s",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model": "PP-OCRv6 Medium (🏠 本地)",
        "time": datetime.now().strftime("%H:%M:%S"),
    }


# ===== 工具函数 =====
def _cv_imread(path: str):
    """支持中文路径的 _cv_imread"""
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

# ===== 页面布局 =====
st.set_page_config(page_title="SiliconFlow OCR 测试台", layout="wide")
st.title("SiliconFlow OCR 在线测试台")

tab_ocr, tab_enhance = st.tabs(["📝 OCR 识别", "🎨 图像增强"])

# ── 共享变量 ──
uploaded_file = None
temp_path = st.session_state.get("selected_sample") or DEFAULT_IMAGE
recognize_btn = False

# ──────────────────────── OCR 识别 ────────────────────────
with tab_ocr:
    col_left, col_right = st.columns([1.2, 2])

    with col_left:
        st.subheader("📤 输入图片")
        uploaded_file = st.file_uploader("上传图片（不选则使用默认测试图）",
                                         type=["jpg", "jpeg", "png", "webp", "bmp"])

        if uploaded_file:
            os.makedirs("_temp", exist_ok=True)
            temp_path = f"_temp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.image(temp_path, width="stretch")
        else:
            temp_path = st.session_state.get("selected_sample") or DEFAULT_IMAGE
            caption = "测试样本" if st.session_state.get("selected_sample") else "默认测试图片（干部任免审批表）"
            if os.path.exists(temp_path):
                st.image(temp_path, width="stretch", caption=caption)
            else:
                st.info("📤 请上传图片")

        # 测试样本缩略图
        sample_dir = os.path.join(os.path.dirname(__file__), "_sample")
        sample_images = sorted([
            f for f in os.listdir(sample_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if sample_images:
            st.image(Image.open(os.path.join(sample_dir, sample_images[0])), width=150)

        st.divider()
        st.subheader("⚙️ 参数设置")

        model_label = st.selectbox("模型选择", list(MODELS.keys()), index=1, key="model_select")
        model_id = MODELS[model_label]
        is_local_model = model_id == PADDLE_LOCAL_MODEL_ID

        # ── 本地模型提示 ──
        if is_local_model:
            st.info("📌 **本地推理** — 纯本地运行，数据不出本机。PP-OCRv6 Medium 约需 1-2GB 内存。")

        show_custom_api = st.checkbox("自定义 API 配置", value=False,
                                       disabled=is_local_model,
                                       help="启用后可自定义 API 地址和模型名称")
        if show_custom_api and not is_local_model:
            with st.container():
                st.text_input("API 地址",
                              value=st.session_state.base_url,
                              key="base_url",
                              help="OpenAI 兼容 API 的 base URL，如 https://api.siliconflow.cn/v1")
                st.text_input("自定义模型名（优先级高于下拉框）",
                              value=st.session_state.get("custom_model", ""),
                              key="custom_model",
                              placeholder="留空则使用下拉框选择的模型",
                              help="如填写，将覆盖下拉框的模型选择，直接使用此模型名请求")
            if st.session_state.custom_model:
                model_id = st.session_state.custom_model

        st.caption("PaddleOCR-VL(0.9B) → DeepSeek-OCR(3B) → Qwen3-VL(8B) → Qwen3.5-9B(9B) → Qwen3-VL(32B) → GLM-4.5V(106B) → 中建电商模型(1.2B) → PP-OCRv6 Medium(34.5M) 本地")

        with st.expander("📊 模型对比", expanded=False):
            st.markdown("""
| 模型 | 参数量 | 耗时 | 表格输出 | 推荐场景 |
|------|--------|------|----------|----------|
| **PaddleOCR-VL-1.5** | 0.9B | ~90s | ❌ 逐行无结构 | 不推荐表格场景 |
| **DeepSeek-OCR** | 3B | ~5-9s | ⚠️ 需清洗 | **速度最快，默认推荐** |
| **Qwen3-VL-8B** | 8B | ~46s | ✅ 原生管道表 | 平衡之选 |
| **Qwen3.5-9B** | 9B | ~15-30s | ✅ 管道表 | 多模态模型，不止 OCR |
| **Qwen3-VL-32B** | 32B | ~108s | ✅ 原生管道表 | 效果优先 |
| **GLM-4.5V** | 106B | ~43s | ✅ 原生管道表 | 复杂表格首选 |
| **中建电商模型** | 1.2B | ~5-15s | ✅ 管道表 | 文档解析专用 |
| **PP-OCRv6 Medium** | 34.5M | ~2-8s | ⚠️ 仅文本行 | **纯本地，无需网络** |
""")

        if "prev_model" not in st.session_state or st.session_state.prev_model != model_label:
            new_prompt = PROMPT_PIPE if ("GLM" in model_id or "Qwen" in model_id or "3.5-9B" in model_id) else DEFAULT_PROMPT
            st.session_state.current_prompt = new_prompt
            st.session_state.prompt_input = new_prompt
            st.session_state.prev_model = model_label

        prompt = st.text_area("识别提示词", st.session_state.current_prompt, height=100,
                              key="prompt_input",
                              disabled=is_local_model,
                              help=("PP-OCRv6 本地模型不支持文本 prompt，此输入框已禁用"
                                    if is_local_model else "切换模型时提示词自动匹配"))

        if is_local_model:
            st.info("🏠 PP-OCRv6 是传统 OCR 管线（检测→识别），不支持语义 prompt。输出为检测到的文本行 + 置信度。")
        elif "GLM" in model_id:
            st.info("✅ GLM-4.5V 使用**管道表 prompt**，直接输出 `|` 格式表格")
        elif "Qwen" in model_id:
            st.info("✅ Qwen3-VL/3.5-9B 使用**管道表 prompt**，直接输出 `|` 格式表格")
        else:
            st.info("📝 DeepSeek/PaddleOCR/中建电商 使用**通用 prompt**，输出后自动转管道表")

        if not is_local_model:
            is_deepseek = "DeepSeek" in model_id
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                temperature = st.slider("Temperature", 0.0, 1.0, 0.0 if is_deepseek else 0.1, 0.05)
                top_p = st.slider("Top-P", 0.0, 1.0, 0.7, 0.05)
            with col_p2:
                max_tokens = st.slider("最大令牌数", 512, 8192, 4096, 512)
                top_k = st.slider("Top-K", 0, 100, 50, 1)
            frequency_penalty = st.slider("频率惩罚", 0.0, 2.0, 0.0, 0.1)

            is_qwen35 = "Qwen3.5-9B" in model_id
            enable_thinking = st.checkbox("🧠 启用思考模式 (enable_thinking)",
                                           value=False,
                                           disabled=not is_qwen35,
                                           help="Qwen3.5-9B 专用：启用后模型会先思考再回答，效果更好但耗时大幅增加（约4分钟）")
        else:
            # 本地模型固定值（不会被使用，但保持变量存在避免引用错误）
            temperature = 0.0
            max_tokens = 4096
            top_p = 0.7
            top_k = 50
            frequency_penalty = 0.0
            enable_thinking = False

        recognize_btn = st.button("🚀 开始识别", type="primary", width="stretch")

    with col_right:
        st.subheader("📄 识别结果")

        final_image_path = temp_path
        if st.session_state.restoration_enabled and st.session_state.restored_image_path:
            if os.path.exists(st.session_state.restored_image_path):
                final_image_path = st.session_state.restored_image_path

        if recognize_btn:
            if not os.path.exists(final_image_path):
                st.error("请先上传图片")
            else:
                with st.spinner(f"正在识别...（模型：{model_label}）"):
                    try:
                        data = ocr_recognize(final_image_path, model_id, prompt, temperature, max_tokens,
                                             top_p, top_k, frequency_penalty,
                                             enable_thinking,
                                             base_url=st.session_state.base_url)

                        col_info1, col_info2, col_info3 = st.columns(3)
                        col_info1.metric("模型", model_label)
                        col_info2.metric("耗时", data["elapsed"])
                        col_info3.metric("Total Tokens", data["total_tokens"])

                        if is_local_model:
                            st.caption("🏠 本地推理完成，无 Token 消耗")

                        with st.expander("ℹ️ Token 详情"):
                            st.write(f"Prompt: {data['prompt_tokens']}, "
                                     f"Completion: {data['completion_tokens']}, "
                                     f"Total: {data['total_tokens']}")

                        tab1, tab2 = st.tabs(["📊 渲染表格", "📝 原始文本"])
                        with tab1:
                            st.markdown(data["result_cleaned"])
                        with tab2:
                            st.code(data["result_cleaned"], language="text")
                        with st.expander("📋 复制管道表"):
                            st.code(data["result_cleaned"], language="markdown")
                            st.download_button("下载 .md 文件", data["result_cleaned"],
                                               file_name="ocr_result.md", mime="text/markdown")

                        with st.expander("📝 查看原始输出"):
                            st.code(data["result"], language="text")

                        data["preprocessing"] = {
                            "enabled": st.session_state.restoration_enabled,
                            "degradation": {
                                "quality": st.session_state.degradation_info["overall_quality"],
                                "noise_type": st.session_state.degradation_info["noise_type"],
                                "is_blurry": st.session_state.degradation_info.get("is_blurry", False),
                            } if st.session_state.degradation_info else None,
                            "params": st.session_state.restoration_params.copy()
                            if st.session_state.restoration_enabled else None,
                        }

                        st.session_state.history.insert(0, data)

                    except Exception as e:
                        st.error(f"识别失败：{e}")

        else:
            st.info("点击「开始识别」使用默认测试图片")

# ──────────────────────── 图像增强 ────────────────────────
with tab_enhance:
    # ── 顶部：图片上传 + 预览 ──
    col_upload, col_preview = st.columns([1.2, 2])

    with col_upload:
        st.subheader("📤 输入图片")
        uploaded_file_e = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                           key="upload_enhance")

        if uploaded_file_e:
            os.makedirs("_temp", exist_ok=True)
            temp_path = f"_temp/{uploaded_file_e.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file_e.getvalue())
        else:
            temp_path = st.session_state.get("selected_sample") or DEFAULT_IMAGE

        # 测试样本
        sample_dir = os.path.join(os.path.dirname(__file__), "_sample")
        sample_images = sorted([
            f for f in os.listdir(sample_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if sample_images:
            st.image(Image.open(os.path.join(sample_dir, sample_images[0])), width=150,
                     caption="测试样本")

    with col_preview:
        st.subheader("🖼️ 图像预览")

        if st.session_state.show_comparison and st.session_state.restored_image_path:
            tab_before, tab_after = st.tabs(["📷 原图", "✨ 修复后"])
            with tab_before:
                if os.path.exists(temp_path):
                    st.image(temp_path, width="stretch", caption="修复前")
            with tab_after:
                if os.path.exists(st.session_state.restored_image_path):
                    st.image(st.session_state.restored_image_path, width="stretch", caption="修复后")
                    if st.session_state.degradation_info:
                        info = st.session_state.degradation_info
                        st.caption(
                            f"质量: {info['overall_quality'].upper()} | "
                            f"噪声: {info['noise_type']} | "
                            f"模糊: {'是' if info.get('is_blurry') else '否'}"
                        )
        else:
            if os.path.exists(temp_path):
                st.image(temp_path, width="stretch",
                         caption="测试样本" if st.session_state.get("selected_sample") else "上传图片")
            else:
                st.info("📤 请上传图片")

    # ── 底部：修复控制 ──
    st.divider()
    st.subheader("🔧 图像预处理")

    enable_restoration = st.checkbox("启用修复预处理",
                                      value=st.session_state.restoration_enabled,
                                      help="开启后，图片先经修复算法处理再送入 OCR")

    if enable_restoration:
        st.session_state.restoration_enabled = True

        col_params, col_actions = st.columns([1, 1])

        with col_params:
            if st.button("🔍 自动检测退化", use_container_width=True):
                if os.path.exists(temp_path):
                    img_bgr = _cv_imread(temp_path)
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
                        info = detect_degradation_type(img_rgb)
                        st.session_state.degradation_info = info
                        st.session_state.show_comparison = False
                        st.rerun()

            if st.session_state.degradation_info:
                info = st.session_state.degradation_info
                quality = info["overall_quality"]
                color = {"good": "green", "fair": "orange", "poor": "red"}
                st.markdown(f"**质量评分:** :{color.get(quality, 'gray')}[{quality.upper()}]")
                st.markdown(f"- 噪声: {info['noise_type']} (程度: {info['noise_level']:.2f})")
                if info.get("is_blurry"):
                    st.markdown(f"- 模糊: {info['blur_type']} (程度: {info['blur_severity']:.2f})")
                if info.get("has_sp"):
                    st.markdown(f"- 椒盐噪声: {info['total_ratio']:.4f}")
                if info.get("recommendations"):
                    st.markdown(f"- 建议: {', '.join(info['recommendations'])}")

        with col_actions:
            with st.expander("⚙️ 手动参数调节", expanded=False):
                params = st.session_state.restoration_params
                params["median_kernel"] = st.slider("中值滤波核大小", 3, 11, params["median_kernel"], 2)
                params["psf_size"] = st.slider("PSF 核大小", 11, 51, params["psf_size"], 2)
                params["wiener_balance"] = st.slider("维纳平衡参数", 0.001, 0.1, params["wiener_balance"], 0.001,
                                                      format="%.3f")
                params["tv_weight"] = st.slider("TV 正则化权重", 0.0, 0.1, params["tv_weight"], 0.001, format="%.3f")
                params["sharpening"] = st.slider("锐化强度", 0.0, 1.0, params["sharpening"], 0.05)

            if st.button("🔄 应用修复", type="primary", use_container_width=True):
                if os.path.exists(temp_path):
                    with st.spinner("正在执行图像修复..."):
                        try:
                            img_bgr = _cv_imread(temp_path)
                            if img_bgr is not None:
                                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
                                params = st.session_state.restoration_params
                                restored = restore_image_with_degradation_awareness(img_rgb, params)
                                restored_bgr = cv2.cvtColor(
                                    (restored * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                                )
                                restored_path = os.path.join("_temp", "_restored_temp.jpg")
                                cv2.imwrite(restored_path, restored_bgr)
                                st.session_state.restored_image_path = restored_path
                                st.session_state.show_comparison = True
                                st.rerun()
                        except Exception as e:
                            st.error(f"修复失败: {e}")

            if st.session_state.restored_image_path:
                if st.button("↩️ 使用原图", use_container_width=True):
                    st.session_state.restored_image_path = None
                    st.session_state.show_comparison = False
                    st.rerun()
    else:
        st.session_state.restoration_enabled = False
        st.session_state.restored_image_path = None
        st.session_state.show_comparison = False

# === 历史记录 ===
st.divider()
st.subheader("📋 历史记录")

if st.session_state.history:
    # 清空按钮
    if st.button("清空历史记录", type="secondary"):
        st.session_state.history = []
        st.rerun()

    for i, record in enumerate(st.session_state.history):
        model_short = record["model"].split("/")[-1]
        with st.expander(f"#{i+1} [{record['time']}] {model_short} — {record['elapsed']}"):
            st.markdown(f"**模型**: {record['model']}")
            st.markdown(f"**耗时**: {record['elapsed']}  |  "
                        f"**Token**: {record['total_tokens']} "
                        f"(prompt={record['prompt_tokens']}, "
                        f"completion={record['completion_tokens']})")
            if record.get("preprocessing", {}).get("enabled"):
                pp = record["preprocessing"]
                deg = pp.get("degradation") or {}
                st.markdown(f"**预处理**: 启用 | 质量: {deg.get('quality', '?')} | "
                            f"{'模糊' if deg.get('is_blurry') else '清晰'}")
            t1, t2 = st.tabs(["📊 表格", "📝 原始"])
            with t1:
                st.markdown(record.get("result_cleaned", record["result"]))
            with t2:
                st.code(record.get("result_cleaned", record["result"]), language="text")
else:
    st.caption("暂无历史记录")
