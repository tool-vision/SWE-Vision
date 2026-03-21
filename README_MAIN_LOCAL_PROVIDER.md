# Main Branch Local Provider Recreation

This branch keeps the original SWE-Vision flow from `main` and recreates the experiment by pointing the existing OpenAI-compatible client at a local vLLM server.

## Local Setup

Start a local OpenAI-compatible server with Qwen:

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --limit-mm-per-prompt '{"image":4}'
```

Build the Docker image for the notebook runtime:

```bash
docker build -t swe-vision:latest -f ./env/Dockerfile ./env
```

## Recreate The Experiment

```bash
OPENAI_API_KEY=EMPTY \
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_MODEL=Qwen/Qwen2.5-VL-3B-Instruct \
python -m swe_vision.cli \
  --no-reasoning \
  --image assets/test_image.png \
  "What is the gap between GPT5.2 and 6-year-olds from the chart?"
```

Or use the convenience script:

```bash
bash run.sh
```

## Notes

- This branch changes the provider path, not the overall agent design.
- The local model server is OpenAI-compatible, so the existing `OpenAI` client path is reused.
- `--no-reasoning` is the safer default for the local vLLM/Qwen path because provider-specific reasoning fields were not reliably honored.
