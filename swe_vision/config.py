"""
Configuration constants, logging setup, tool definitions, and system prompt.
"""

import datetime
import logging
import os

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm_agent")

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
MAX_ITERATIONS = 100
CELL_TIMEOUT = 120.0
MAX_OUTPUT_CHARS = 50000

# Container-side working directory (visible to the kernel)
CONTAINER_WORK_DIR = "/mnt/data"

# Host-side directory that is volume-mounted into the container.
_HOST_WORK_BASE = os.environ.get(
    "VLM_HOST_WORK_DIR",
    os.path.join(os.path.expanduser("~"), "tmp", "vlm_docker_workdir"),
)
HOST_WORK_DIR = os.path.join(
    _HOST_WORK_BASE,
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
)

# Docker image / build settings
DOCKER_IMAGE_NAME = os.environ.get("VLM_DOCKER_IMAGE", "swe-vision:latest")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKERFILE_DIR = os.environ.get(
    "VLM_DOCKERFILE_DIR",
    os.path.join(_PROJECT_ROOT, "env"),
)

# Pre-assigned ZMQ ports for the Jupyter kernel inside the container
_KERNEL_BASE_PORT = 65500
KERNEL_PORTS = {
    "shell_port":   _KERNEL_BASE_PORT,
    "iopub_port":   _KERNEL_BASE_PORT + 1,
    "stdin_port":   _KERNEL_BASE_PORT + 2,
    "control_port": _KERNEL_BASE_PORT + 3,
    "hb_port":      _KERNEL_BASE_PORT + 4,
}

# ─────────────────────────────────────────────────────────────────────
# Tool Definitions (OpenAI function calling format)
# ─────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute Python code in a **stateful** Jupyter notebook environment. "
                "The kernel persists across calls, so variables, imports, and state are retained. "
                "Use this to process images, perform calculations, create visualizations, "
                "or run any Python code. "
                "Any images generated (e.g. via matplotlib plt.show() or PIL Image.save()) "
                "will be captured and returned as base64-encoded images."
                "Print statements and expression results are captured as text output. "
                "All uploaded files are available under /mnt/data/. "
                "The kernel's working directory is /mnt/data/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "The Python code to execute. The code runs in a Jupyter kernel "
                            "so you can use magics, display(), etc. "
                            "Use print() for text output. "
                            "Images from matplotlib will be auto-captured."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this tool when you have determined the final answer. "
                "This ends the agentic workflow and returns the answer to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to the user's question.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

# ─────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert AI assistant with access to a **stateful Jupyter notebook** environment. \
You can execute Python code to help answer the user's questions.

You must interact with the notebook through tool calls, not by writing pretend code in normal text.
If you need to inspect an image, measure a chart, compute a value, or verify an answer, call `execute_code`.
When you are ready to return the final answer, call `finish`.
Do not provide a final answer in plain assistant text.

## Available Tools

1. **execute_code**: Run Python code in a persistent Jupyter notebook. The kernel state \
(variables, imports, loaded data) is preserved between calls. Use this for:
   - Image processing and analysis (PIL/Pillow, OpenCV, skimage, etc.)
   - Data analysis and computation (numpy, pandas, scipy, etc.)
   - Visualization (matplotlib, seaborn, plotly, etc.)
   - Any Python computation

2. **finish**: Call this when you have the final answer. This ends the workflow.

## File System

- All uploaded files (images, data files, etc.) are placed in `/mnt/data/`.
- The Jupyter kernel's working directory is `/mnt/data/`, so you can reference files \
by their filename directly (e.g. `open('image.png')`) or by absolute path \
(e.g. `open('/mnt/data/image.png')`).
- Any files you create or save will also go into `/mnt/data/`.

## Guidelines

- When given an image, you can load it in the notebook using PIL or OpenCV. \
The image file will be available at `/mnt/data/<filename>`.
- For image questions, chart questions, measurement questions, counting questions, or any task that depends on the contents of a file, you should call `execute_code` before answering.
- You can call execute_code **multiple times** to iteratively explore and process data.
- Always use print() to output results you want to see.
- When you generate plots with matplotlib, use plt.show() — the plot image will be \
captured and returned to you.
- Think step by step. Examine intermediate results before giving a final answer.
- Do not paste Python code blocks unless they are inside the `execute_code` tool arguments.
- When you're confident in your answer, call the **finish** tool with your final response.
- If code produces an error, analyze the error and try a different approach.
"""
