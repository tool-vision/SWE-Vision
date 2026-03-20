#!/usr/bin/env python3
"""Batch runner for local vLLM/Qwen SWE-Vision experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import error, parse, request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
ALT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MAX_ITERATIONS = 100


@dataclass
class ManifestRow:
    id: str
    query: str
    image_paths: List[str]
    metadata: Dict[str, Any]
    expected_answer: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a JSONL manifest through SWE-Vision using local vLLM/Qwen.",
    )
    parser.add_argument("--manifest", required=True, help="Path to the input JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for predictions and trajectories.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Local vLLM-served model to use (default: {DEFAULT_MODEL}; lighter option: {ALT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible vLLM base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help=f"API key sent to the OpenAI-compatible client (default: {DEFAULT_API_KEY}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Max agent iterations per example (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable reasoning mode (default: False for the local vLLM/Qwen baseline).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N manifest rows.")
    parser.add_argument("--resume", action="store_true", help="Skip IDs already recorded as successful.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed example.")
    return parser.parse_args()


def read_manifest(manifest_path: Path) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    seen_ids = set()

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Manifest line {line_no}: invalid JSON: {exc}") from exc

            if not isinstance(data, dict):
                raise ValueError(f"Manifest line {line_no}: expected an object.")

            row_id = data.get("id")
            if not isinstance(row_id, str) or not row_id.strip():
                raise ValueError(f"Manifest line {line_no}: missing non-empty string 'id'.")
            if row_id in seen_ids:
                raise ValueError(f"Manifest line {line_no}: duplicate id '{row_id}'.")
            seen_ids.add(row_id)

            query = data.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Manifest line {line_no}: missing non-empty string 'query'.")

            image_paths = data.get("image_paths", [])
            if image_paths is None:
                image_paths = []
            if not isinstance(image_paths, list) or any(not isinstance(item, str) for item in image_paths):
                raise ValueError(f"Manifest line {line_no}: 'image_paths' must be a list of strings.")

            metadata = data.get("metadata", {})
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError(f"Manifest line {line_no}: 'metadata' must be an object when provided.")

            resolved_images = []
            for image_path in image_paths:
                resolved = (manifest_path.parent / image_path).resolve()
                if not resolved.exists():
                    raise ValueError(
                        f"Manifest line {line_no}: image path does not exist: {image_path}"
                    )
                resolved_images.append(str(resolved))

            rows.append(
                ManifestRow(
                    id=row_id,
                    query=query,
                    image_paths=resolved_images,
                    metadata=metadata,
                    expected_answer=data.get("expected_answer"),
                )
            )

    return rows


def ensure_output_dirs(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    return runs_dir


def predictions_path(output_dir: Path) -> Path:
    return output_dir / "predictions.jsonl"


def summary_path(output_dir: Path) -> Path:
    return output_dir / "summary.json"


def load_latest_statuses(path: Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return latest

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Existing predictions line {line_no}: invalid JSON: {exc}") from exc
            row_id = record.get("id")
            if isinstance(row_id, str):
                latest[row_id] = record
    return latest


def append_prediction(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sanitize_id(raw_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id).strip("._")
    return sanitized or "example"


def find_latest_run_dir(runs_dir: Path, row_id: str) -> Optional[Path]:
    prefix = f"{sanitize_id(row_id)}_"
    matches = sorted(
        path for path in runs_dir.glob(f"{prefix}*")
        if path.is_dir()
    )
    return matches[-1] if matches else None


def format_timestamp(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


def vllm_models_url(base_url: str) -> str:
    parsed = parse.urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        models_path = f"{path}/models"
    elif path:
        models_path = f"{path}/v1/models"
    else:
        models_path = "/v1/models"
    rebuilt = parsed._replace(path=models_path, params="", query="", fragment="")
    return parse.urlunparse(rebuilt)


def recommended_vllm_serve_command(model: str) -> str:
    return (
        f"vllm serve {model} "
        "--max-model-len 8192 "
        "--gpu-memory-utilization 0.85 "
        "--enable-auto-tool-choice "
        "--tool-call-parser hermes "
        "--limit-mm-per-prompt '{\"image\":4}'"
    )


def fetch_json(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def preflight_or_die(base_url: str, model: str) -> None:
    models_url = vllm_models_url(base_url)
    serve_cmd = recommended_vllm_serve_command(model)
    try:
        payload = fetch_json(models_url)
    except error.URLError as exc:
        install_hint = ""
        if shutil.which("vllm") is None:
            install_hint = "\nInstall vLLM first, for example: pip install vllm"
        raise SystemExit(
            "Preflight failed: vLLM server is unreachable at "
            f"{base_url}.{install_hint}\n"
            "Start a local OpenAI-compatible server with:\n"
            f"  {serve_cmd}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Preflight failed: expected JSON from {models_url}, got invalid data: {exc}"
        ) from exc

    models = payload.get("data", [])
    available = {item.get("id") for item in models if isinstance(item, dict)}
    if model not in available:
        available_list = ", ".join(sorted(name for name in available if name)) or "(none found)"
        raise SystemExit(
            "Preflight failed: requested model is not being served by vLLM.\n"
            f"Requested: {model}\nAvailable: {available_list}\n"
            "Recover by starting vLLM with:\n"
            f"  {serve_cmd}"
        )


async def run_example(
    row: ManifestRow,
    args: argparse.Namespace,
    runs_dir: Path,
) -> Dict[str, Any]:
    from swe_vision.agent import VLMToolCallAgent

    started_at = datetime.now()
    trajectory_prefix = runs_dir / sanitize_id(row.id)
    agent = VLMToolCallAgent(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_iterations=args.max_iterations,
        verbose=False,
        save_trajectory=str(trajectory_prefix),
        reasoning=args.reasoning,
    )

    status = "success"
    answer = None
    error_text = None

    try:
        answer = await agent.run(row.query, row.image_paths or None)
    except Exception as exc:
        status = "error"
        error_text = str(exc)
    finally:
        await agent.cleanup()

    finished_at = datetime.now()
    trajectory_dir = find_latest_run_dir(runs_dir, row.id)

    return {
        "id": row.id,
        "status": status,
        "query": row.query,
        "image_paths": row.image_paths,
        "metadata": row.metadata,
        "expected_answer": row.expected_answer,
        "model": args.model,
        "base_url": args.base_url,
        "reasoning": args.reasoning,
        "max_iterations": args.max_iterations,
        "trajectory_dir": str(trajectory_dir) if trajectory_dir else None,
        "answer": answer,
        "error": error_text,
        "started_at": format_timestamp(started_at),
        "finished_at": format_timestamp(finished_at),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
    }


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    output_dir: Path,
    total_manifest_rows: int,
    selected_rows: int,
    records: Iterable[Dict[str, Any]],
    started_at: datetime,
    finished_at: datetime,
) -> None:
    counts = Counter(record["status"] for record in records)
    summary = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "model": args.model,
        "base_url": args.base_url,
        "reasoning": args.reasoning,
        "max_iterations": args.max_iterations,
        "limit": args.limit,
        "resume": args.resume,
        "fail_fast": args.fail_fast,
        "total_manifest_rows": total_manifest_rows,
        "selected_rows": selected_rows,
        "attempted": counts.get("success", 0) + counts.get("error", 0),
        "succeeded": counts.get("success", 0),
        "failed": counts.get("error", 0),
        "skipped": counts.get("skipped", 0),
        "started_at": format_timestamp(started_at),
        "finished_at": format_timestamp(finished_at),
        "predictions_path": str(predictions_path(output_dir)),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


async def async_main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    runs_dir = ensure_output_dirs(output_dir)

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer.")

    all_rows = read_manifest(manifest_path)
    rows = all_rows
    if args.limit is not None:
        rows = rows[: args.limit]

    preflight_or_die(args.base_url, args.model)

    existing = load_latest_statuses(predictions_path(output_dir)) if args.resume else {}
    run_started_at = datetime.now()
    current_records: List[Dict[str, Any]] = []

    print(f"Loaded {len(rows)} manifest row(s) from {manifest_path}")
    print(f"Writing outputs to {output_dir}")

    for index, row in enumerate(rows, start=1):
        previous = existing.get(row.id)
        if previous and previous.get("status") == "success":
            skipped_record = {
                "id": row.id,
                "status": "skipped",
                "query": row.query,
                "image_paths": row.image_paths,
                "metadata": row.metadata,
                "expected_answer": row.expected_answer,
                "model": args.model,
                "base_url": args.base_url,
                "reasoning": args.reasoning,
                "max_iterations": args.max_iterations,
                "trajectory_dir": previous.get("trajectory_dir"),
                "answer": previous.get("answer"),
                "error": None,
                "started_at": format_timestamp(datetime.now()),
                "finished_at": format_timestamp(datetime.now()),
                "duration_seconds": 0.0,
            }
            append_prediction(predictions_path(output_dir), skipped_record)
            current_records.append(skipped_record)
            print(f"[{index}/{len(rows)}] Skipping {row.id} (already successful)")
            continue

        print(f"[{index}/{len(rows)}] Running {row.id}")
        record = await run_example(row, args, runs_dir)
        if isinstance(record["answer"], str) and record["answer"].startswith("[Error]"):
            record["status"] = "error"
            record["error"] = record["answer"]
        append_prediction(predictions_path(output_dir), record)
        current_records.append(record)

        if record["status"] == "success":
            print(f"  Success: trajectory={record['trajectory_dir']}")
        else:
            print(f"  Error: {record['error']}")
            if args.fail_fast:
                print("Stopping early because --fail-fast is set.")
                break

    run_finished_at = datetime.now()
    write_summary(
        summary_path(output_dir),
        args=args,
        manifest_path=manifest_path,
        output_dir=output_dir,
        total_manifest_rows=len(all_rows),
        selected_rows=len(rows),
        records=current_records,
        started_at=run_started_at,
        finished_at=run_finished_at,
    )
    print(f"Summary written to {summary_path(output_dir)}")
    return 0


def main() -> int:
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
