# Local vLLM Qwen Baseline

This document describes the local Qwen implementation added on top of SWE-Vision.

It is specifically for:
- local OpenAI-compatible serving through `vllm`
- Qwen2.5-VL models
- batch experiments through `scripts/run_experiments.py`
- Docker-backed tool execution through the existing `VLMToolCallAgent`

## What This Implementation Adds

- A batch runner: [`scripts/run_experiments.py`](/home/zmy/Repos/tool-vision/SWE-Vision/scripts/run_experiments.py)
- A local vLLM serve helper: [`scripts/start_vllm_qwen.sh`](/home/zmy/Repos/tool-vision/SWE-Vision/scripts/start_vllm_qwen.sh)
- A Conda environment spec: [`environment.yml`](/home/zmy/Repos/tool-vision/SWE-Vision/environment.yml)
- A Docker image for the notebook kernel: [`env/Dockerfile`](/home/zmy/Repos/tool-vision/SWE-Vision/env/Dockerfile)

## Default Local Setup

Batch runner defaults:

- `base_url`: `http://localhost:8000/v1`
- `api_key`: `EMPTY`
- `model`: `Qwen/Qwen2.5-VL-7B-Instruct`
- `reasoning`: off by default

Practical note:
- On this machine, `Qwen/Qwen2.5-VL-7B-Instruct` did not fit in GPU memory.
- `Qwen/Qwen2.5-VL-3B-Instruct` was the working local fallback.

## Host Prerequisites

- Conda
- Docker
- vLLM
- NVIDIA GPU with enough memory for the chosen Qwen VL model

Install the Python environment:

```bash
conda env create -f environment.yml
conda activate swe-vision
```

Build the notebook image:

```bash
docker build -t swe-vision:latest -f ./env/Dockerfile ./env
```

Install vLLM:

```bash
pip install vllm
```

## Start the Local Model Server

Default helper command:

```bash
bash scripts/start_vllm_qwen.sh
```

This starts:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --limit-mm-per-prompt '{"image":4}'
```

Use the lighter model if needed:

```bash
bash scripts/start_vllm_qwen.sh Qwen/Qwen2.5-VL-3B-Instruct
```

Override memory-related defaults if needed:

```bash
MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.80 \
  bash scripts/start_vllm_qwen.sh Qwen/Qwen2.5-VL-3B-Instruct
```

## Run a Batch Experiment

Manifest format:
- one JSON object per line
- required: `id`, `query`
- optional: `image_paths`, `metadata`, `expected_answer`

Example manifest:

```jsonl
{"id":"chart_001","query":"What is the gap between GPT5.2 and 6-year-olds from the chart?","image_paths":["assets/test_image.png"]}
```

Run the batch job:

```bash
python scripts/run_experiments.py \
  --manifest ./examples.jsonl \
  --output-dir ./outputs/qwen_run
```

Run against the 3B local model:

```bash
python scripts/run_experiments.py \
  --manifest ./examples.jsonl \
  --output-dir ./outputs/qwen_3b_run \
  --model Qwen/Qwen2.5-VL-3B-Instruct
```

Useful options:

```bash
python scripts/run_experiments.py \
  --manifest ./examples.jsonl \
  --output-dir ./outputs/qwen_run \
  --resume \
  --limit 10 \
  --fail-fast
```

## Output Files

Each run writes:

```text
<output-dir>/
├── predictions.jsonl
├── summary.json
└── runs/
    └── <id>_<timestamp>/
        ├── trajectory.json
        ├── messages_raw.json
        └── images/
```

## Important Caveats

This implementation is working at the infrastructure level, but model behavior still matters.

Observed during local testing:
- Ollama `qwen2.5vl:7b` did not support this repo's tool-calling workflow.
- vLLM `Qwen/Qwen2.5-VL-7B-Instruct` exceeded available GPU memory here.
- vLLM `Qwen/Qwen2.5-VL-3B-Instruct` served successfully.
- The 3B model completed a smoke run, but it did not reliably emit tool calls for the tested chart query.

That means:
- the serving path works
- the batch runner works
- the Docker notebook path works
- tool-use reliability still needs prompt/protocol hardening for Qwen

## Single Query Usage

CLI example:

```bash
python -m swe_vision.cli \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --no-reasoning \
  --image assets/test_image.png \
  "What is the gap between GPT5.2 and 6-year-olds from the chart?"
```

## Related Files

- [`scripts/run_experiments.py`](/home/zmy/Repos/tool-vision/SWE-Vision/scripts/run_experiments.py)
- [`scripts/start_vllm_qwen.sh`](/home/zmy/Repos/tool-vision/SWE-Vision/scripts/start_vllm_qwen.sh)
- [`run.sh`](/home/zmy/Repos/tool-vision/SWE-Vision/run.sh)
- [`swe_vision/agent.py`](/home/zmy/Repos/tool-vision/SWE-Vision/swe_vision/agent.py)
- [`README.md`](/home/zmy/Repos/tool-vision/SWE-Vision/README.md)
