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
import time
from datetime import datetime

import streamlit as st
from openai import OpenAI

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
}

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


import re


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


# ===== 页面布局 =====
st.set_page_config(page_title="SiliconFlow OCR 测试台", layout="wide")
st.title("SiliconFlow OCR 在线测试台")

col1, col2 = st.columns([1, 1.8])

with col1:
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
        temp_path = DEFAULT_IMAGE
        if os.path.exists(temp_path):
            st.image(temp_path, width="stretch", caption="默认测试图片（干部任免审批表）")
        else:
            st.info("📤 请上传图片进行识别")

    st.divider()
    st.subheader("⚙️ 参数设置")

    model_label = st.selectbox("模型选择", list(MODELS.keys()), index=1, key="model_select")
    model_id = MODELS[model_label]

    # 自定义 API 配置（默认收起）
    show_custom_api = st.checkbox("自定义 API 配置", value=False,
                                   help="启用后可自定义 API 地址和模型名称")
    if show_custom_api:
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

    st.caption("PaddleOCR-VL(0.9B) → DeepSeek-OCR(3B) → Qwen3-VL(8B) → Qwen3.5-9B(9B) → Qwen3-VL(32B) → GLM-4.5V(106B)")

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
""")

    # 模型切换时自动更新提示词
    if "prev_model" not in st.session_state or st.session_state.prev_model != model_label:
        new_prompt = PROMPT_PIPE if ("GLM" in model_id or "Qwen" in model_id or "3.5-9B" in model_id) else DEFAULT_PROMPT
        st.session_state.current_prompt = new_prompt
        st.session_state.prompt_input = new_prompt  # 同步更新 widget 状态
        st.session_state.prev_model = model_label

    prompt = st.text_area("识别提示词", st.session_state.current_prompt, height=100,
                          key="prompt_input",
                          help="切换模型时提示词自动匹配")

    # 显示当前模型使用的提示词类型
    if "GLM" in model_id:
        st.info("✅ GLM-4.5V 使用**管道表 prompt**，直接输出 `|` 格式表格")
    elif "Qwen" in model_id:
        st.info("✅ Qwen3-VL/3.5-9B 使用**管道表 prompt**，直接输出 `|` 格式表格")
    else:
        st.info("📝 DeepSeek/PaddleOCR 使用**通用 prompt**，输出后自动转管道表")

    # 模型参数（各模型默认值不同）
    is_deepseek = "DeepSeek" in model_id
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0 if is_deepseek else 0.1, 0.05)
        top_p = st.slider("Top-P", 0.0, 1.0, 0.7, 0.05)
    with col_p2:
        max_tokens = st.slider("最大令牌数", 512, 8192, 4096, 512)
        top_k = st.slider("Top-K", 0, 100, 50, 1)
    frequency_penalty = st.slider("频率惩罚", 0.0, 2.0, 0.0, 0.1)

    # Qwen3.5-9B 思考模式开关（默认关闭）
    is_qwen35 = "Qwen3.5-9B" in model_id
    enable_thinking = st.checkbox("🧠 启用思考模式 (enable_thinking)",
                                   value=False,
                                   disabled=not is_qwen35,
                                   help="Qwen3.5-9B 专用：启用后模型会先思考再回答，效果更好但耗时大幅增加（约4分钟）")

    recognize_btn = st.button("🚀 开始识别", type="primary", width="stretch")

with col2:
    st.subheader("📄 识别结果")

    if recognize_btn:
        if not os.path.exists(temp_path):
            st.error("请先上传图片")
        else:
            with st.spinner(f"正在识别...（模型：{model_label}）"):
                try:
                    data = ocr_recognize(temp_path, model_id, prompt, temperature, max_tokens,
                                         top_p, top_k, frequency_penalty,
                                         enable_thinking,
                                         base_url=st.session_state.base_url)

                    # 显示识别信息
                    col_info1, col_info2, col_info3 = st.columns(3)
                    col_info1.metric("模型", model_label)
                    col_info2.metric("耗时", data["elapsed"])
                    col_info3.metric("Total Tokens", data["total_tokens"])

                    with st.expander("ℹ️ Token 详情"):
                        st.write(f"Prompt: {data['prompt_tokens']}, "
                                 f"Completion: {data['completion_tokens']}, "
                                 f"Total: {data['total_tokens']}")

                    # 显示识别文本
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

                    # 记录历史
                    st.session_state.history.insert(0, data)

                except Exception as e:
                    st.error(f"识别失败：{e}")

    else:
        st.info("点击「开始识别」使用默认测试图片")

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
            t1, t2 = st.tabs(["📊 表格", "📝 原始"])
            with t1:
                st.markdown(record.get("result_cleaned", record["result"]))
            with t2:
                st.code(record.get("result_cleaned", record["result"]), language="text")
else:
    st.caption("暂无历史记录")
