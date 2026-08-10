#!/usr/bin/env python3
"""Replay Ascend compile commands with --mlir-timing --mlir-output-format=json
and produce per-kernel pass timing NDJSON.

Walks $TRITON_DUMP_DIR, finds every external compile command, re-runs it with
MLIR timing flags, parses the pass timing JSON from stderr, and emits one
NDJSON line per kernel with the breakdown:
  total_us, npuir_self_us, hivmc_self_us, bisheng_self_us, lld_self_us
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dump_dir",
        nargs="?",
        type=Path,
        help="Dump directory to scan. Default: $TRITON_DUMP_DIR.",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="-",
        type=Path,
        help="Output NDJSON path. Use '-' for stdout. Default: stdout.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds per compile replay. Default: 600.",
    )
    return parser.parse_args()


def find_external_commands(dump_dir: Path) -> list[tuple[Path, Path, dict[str, Any]]]:
    results: list[tuple[Path, Path, dict[str, Any]]] = []
    for subdir in sorted(dump_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for cmd_file in sorted(subdir.glob("compile_command.*.json")):
            try:
                command = json.loads(cmd_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(command, dict) or command.get("kind") != "external":
                continue
            results.append((subdir, cmd_file, command))
    return results


def extract_json_arrays(text: str) -> list[str]:
    arrays: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                arrays.append(text[start : i + 1])
                start = -1
    return arrays


def parse_timing_arrays(text: str) -> list[list[dict[str, Any]]]:
    arrays: list[list[dict[str, Any]]] = []
    for raw in extract_json_arrays(text):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and len(parsed) > 0:
            arrays.append(parsed)
    return arrays


def top_level_names(arr: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for entry in arr:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.add(entry["name"])
    return names


def classify_arrays(
    arrays: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    npuir_arr: list[dict[str, Any]] | None = None
    hivmc_arr: list[dict[str, Any]] | None = None
    for arr in arrays:
        names = top_level_names(arr)
        if "hivmc-a5" in names:
            npuir_arr = arr
        if "bisheng" in names and "ld.lld" in names:
            hivmc_arr = arr
    return npuir_arr, hivmc_arr


def find_entry(
    arr: list[dict[str, Any]], name: str
) -> dict[str, Any] | None:
    for entry in arr:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    return None


def wall_duration_s(entry: dict[str, Any] | None) -> float | None:
    if entry is None:
        return None
    wall = entry.get("wall")
    if not isinstance(wall, dict):
        return None
    value = wall.get("duration")
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def to_us(duration_s: float | None) -> int | None:
    if duration_s is None:
        return None
    return int(round(duration_s * 1_000_000))


def compute_metrics(
    npuir_arr: list[dict[str, Any]],
    hivmc_arr: list[dict[str, Any]] | None,
) -> dict[str, int | None] | None:
    total_s = wall_duration_s(find_entry(npuir_arr, "Total"))
    if total_s is None:
        return None
    total_us = to_us(total_s)
    if total_us is None:
        return None

    hivmc_a5_s = wall_duration_s(find_entry(npuir_arr, "hivmc-a5"))
    hivmc_a5_us = to_us(hivmc_a5_s)
    npuir_self_us = total_us - (hivmc_a5_us or 0)

    hivmc_launch_us: int | None = None
    hivmc_self_us: int | None = None
    bisheng_us: int | None = None
    lld_us: int | None = None

    if hivmc_arr is not None:
        hivmc_total_s = wall_duration_s(find_entry(hivmc_arr, "Total"))
        hivmc_total_us = to_us(hivmc_total_s)
        bisheng_us = to_us(wall_duration_s(find_entry(hivmc_arr, "bisheng")))
        lld_us = to_us(wall_duration_s(find_entry(hivmc_arr, "ld.lld")))

        if hivmc_a5_us is not None and hivmc_total_us is not None:
            hivmc_launch_us = hivmc_a5_us - hivmc_total_us

        if hivmc_total_us is not None:
            hivmc_self_us = hivmc_total_us - (bisheng_us or 0) - (lld_us or 0)

    return {
        "total_us": total_us,
        "npuir_self_us": npuir_self_us,
        "hivmc_launch_us": hivmc_launch_us,
        "hivmc_self_us": hivmc_self_us,
        "bisheng_self_us": bisheng_us,
        "lld_self_us": lld_us,
    }


def get_kernel_signature(subdir: Path) -> str | None:
    ndjson_path = subdir / "compile_times.ndjson"
    if not ndjson_path.is_file():
        return None
    try:
        for line in reversed(ndjson_path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sig = record.get("kernel_signature")
            if isinstance(sig, str) and sig:
                return sig
    except (json.JSONDecodeError, OSError):
        return None
    return None


def main() -> int:
    args = parse_args()

    if args.dump_dir is not None:
        dump_dir = args.dump_dir
    else:
        value = os.environ.get("TRITON_DUMP_DIR")
        if not value:
            raise SystemExit("dump_dir is required when TRITON_DUMP_DIR is not set")
        dump_dir = Path(value)

    if not dump_dir.is_dir():
        raise SystemExit(f"dump directory does not exist: {dump_dir}")

    commands = find_external_commands(dump_dir)
    if not commands:
        raise SystemExit(f"no external compile commands found in {dump_dir}")

    results: list[dict[str, Any]] = []
    total = len(commands)

    for idx, (subdir, cmd_file, command) in enumerate(commands, 1):
        kernel_signature = get_kernel_signature(subdir)
        label = kernel_signature or subdir.name
        sys.stderr.write(f"[{idx}/{total}] {label} ... ")
        sys.stderr.flush()

        replay_argv = command.get("replay_argv")
        replay_cwd = command.get("replay_cwd")

        if not isinstance(replay_argv, list) or not replay_argv:
            sys.stderr.write("FAIL (no replay_argv)\n")
            raise SystemExit(f"no replay_argv in {cmd_file}")

        cwd = replay_cwd if isinstance(replay_cwd, str) and replay_cwd else None
        timed_argv = list(replay_argv) + ["--mlir-timing", "--mlir-output-format=json"]

        try:
            proc = subprocess.run(
                timed_argv,
                cwd=cwd,
                capture_output=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"FAIL (timeout {args.timeout}s)\n")
            raise SystemExit(f"compile timed out for {label}")
        except OSError as exc:
            sys.stderr.write(f"FAIL ({exc})\n")
            raise SystemExit(f"failed to run compile for {label}: {exc}")

        if proc.returncode != 0:
            stderr_text = proc.stderr.decode("utf-8", errors="replace")
            tail = stderr_text[-800:] if len(stderr_text) > 800 else stderr_text
            sys.stderr.write(f"FAIL (exit {proc.returncode})\n")
            raise SystemExit(
                f"compile failed for {label} (exit {proc.returncode}):\n{tail}"
            )

        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        arrays = parse_timing_arrays(stderr_text)

        if not arrays:
            stdout_text = proc.stdout.decode("utf-8", errors="replace")
            arrays = parse_timing_arrays(stdout_text)

        if not arrays:
            sys.stderr.write("FAIL (no timing JSON)\n")
            raise SystemExit(f"no timing JSON arrays found for {label}")

        npuir_arr, hivmc_arr = classify_arrays(arrays)

        if npuir_arr is None:
            sys.stderr.write("FAIL (no npuir timing)\n")
            raise SystemExit(f"no npuir timing array (with hivmc-a5) found for {label}")

        metrics = compute_metrics(npuir_arr, hivmc_arr)
        if metrics is None:
            sys.stderr.write("FAIL (no Total entry)\n")
            raise SystemExit(f"no Total timing entry found for {label}")

        record: dict[str, Any] = {
            "kernel_signature": kernel_signature or subdir.name,
            "source": subdir.name,
        }
        record.update(metrics)
        results.append(record)
        sys.stderr.write("ok\n")

    if str(args.out) == "-":
        out_handle = sys.stdout
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_handle = args.out.open("w", encoding="utf-8")

    try:
        for record in results:
            out_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    finally:
        if out_handle is not sys.stdout:
            out_handle.close()

    print(f"wrote {len(results)} record(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
