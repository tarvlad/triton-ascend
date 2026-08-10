#!/usr/bin/env python3
"""Collect Triton compile-time NDJSON records from a dump directory.

By default the input directory is TRITON_DUMP_DIR and the script scans all
*.ndjson files below it. JSON object lines with a stages_us list must carry a
name(args...) kernel_signature captured at launch time. Unrelated NDJSON lines
are ignored. Each output record contains that human-readable kernel_signature
plus ordered stage timings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dump_dir",
        nargs="?",
        type=Path,
        help="Directory to scan. Default: $TRITON_DUMP_DIR.",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="-",
        type=Path,
        help="Combined NDJSON output path. Use '-' for stdout. Default: stdout.",
    )
    parser.add_argument(
        "--pattern",
        default="*.ndjson",
        help="Recursive filename pattern to scan. Default: *.ndjson.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed JSON in matched files instead of skipping those lines.",
    )
    return parser.parse_args()


def dump_dir_from_args(args: argparse.Namespace) -> Path:
    if args.dump_dir is not None:
        return args.dump_dir
    value = os.environ.get("TRITON_DUMP_DIR")
    if not value:
        raise SystemExit("dump_dir is required when TRITON_DUMP_DIR is not set")
    return Path(value)


def load_compile_time_record(line: str, *, path: Path, line_no: int, strict: bool) -> dict[str, Any] | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        if strict:
            raise SystemExit(f"bad JSON {path}:{line_no}: {exc}") from exc
        return None
    if not isinstance(record, dict) or not isinstance(record.get("stages_us"), list):
        return None
    return record


def input_paths(dump_dir: Path, pattern: str, out_path: Path | None) -> list[Path]:
    paths = []
    out_resolved = out_path.resolve() if out_path is not None else None
    for path in sorted(dump_dir.rglob(pattern)):
        if not path.is_file():
            continue
        if out_resolved is not None and path.resolve() == out_resolved:
            continue
        paths.append(path)
    return paths


def signature_from_structured(record: dict[str, Any]) -> str | None:
    signature = record.get("signature")
    if isinstance(signature, str) and signature:
        return signature
    if not isinstance(signature, dict):
        return None

    name = signature.get("name") or record.get("kernel_name")
    params = signature.get("parameters")
    if not isinstance(name, str) or not isinstance(params, list):
        return None

    rendered_params = []
    for param in params:
        if not isinstance(param, dict):
            continue
        text = str(param.get("name", "arg"))
        if "type" in param:
            text += f":{param['type']}"
        launch = param.get("launch")
        shape = param.get("shape")
        stride = param.get("stride")
        if not isinstance(shape, list) and isinstance(launch, dict):
            shape = launch.get("shape")
            stride = launch.get("stride")
        if isinstance(shape, list):
            text += "[" + ("x".join(str(dim) for dim in shape) if shape else "scalar") + "]"
            if isinstance(stride, list):
                text += " stride=(" + ",".join(str(item) for item in stride) + ")"
        if "value" in param:
            text += f"={param['value']!r}"
        rendered_params.append(text)
    return f"{name}({', '.join(rendered_params)})"


def is_full_signature(value: str) -> bool:
    return "(" in value and value.rstrip().endswith(")")


def kernel_signature(record: dict[str, Any], record_path: Path) -> str | None:
    value = record.get("kernel_signature")
    if isinstance(value, str) and is_full_signature(value):
        return value
    structured = signature_from_structured(record)
    if structured and is_full_signature(structured):
        return structured
    return None


def normalized_stages(record: dict[str, Any]) -> list[dict[str, int | str]]:
    stages = []
    for stage in record.get("stages_us", []):
        if not isinstance(stage, dict):
            continue
        name = stage.get("name")
        if not isinstance(name, str):
            continue
        try:
            time_us = int(stage.get("time_us", 0))
        except (TypeError, ValueError):
            time_us = 0
        stages.append({"name": name, "time_us": time_us})
    return stages


def normalize_record(
    record: dict[str, Any],
    record_path: Path,
    dump_dir: Path,
    line_no: int,
) -> dict[str, Any]:
    signature = kernel_signature(record, record_path)
    if signature is None:
        raise SystemExit(f"missing full kernel signature for compile timing record at {record_path}:{line_no}")
    stages = normalized_stages(record)
    output = {
        "kernel_signature": signature,
        "stages_us": stages,
    }
    try:
        output["source"] = str(record_path.relative_to(dump_dir))
    except ValueError:
        output["source"] = str(record_path)
    return output


def concat_records(paths: list[Path], out_handle, *, dump_dir: Path, strict: bool) -> tuple[int, int]:
    file_count = 0
    record_count = 0
    for path in paths:
        matched_in_file = 0
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                record = load_compile_time_record(line, path=path, line_no=line_no, strict=strict)
                if record is None:
                    continue
                normalized = normalize_record(record, path, dump_dir, line_no)
                out_handle.write(json.dumps(normalized, separators=(",", ":")) + "\n")
                matched_in_file += 1
        if matched_in_file:
            file_count += 1
            record_count += matched_in_file
    return file_count, record_count


def main() -> int:
    args = parse_args()
    dump_dir = dump_dir_from_args(args)
    if not dump_dir.is_dir():
        raise SystemExit(f"dump directory does not exist: {dump_dir}")

    out_path = None if str(args.out) == "-" else args.out
    paths = input_paths(dump_dir, args.pattern, out_path)

    if out_path is None:
        file_count, record_count = concat_records(paths, sys.stdout, dump_dir=dump_dir, strict=args.strict)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as out_handle:
            file_count, record_count = concat_records(
                paths,
                out_handle,
                dump_dir=dump_dir,
                strict=args.strict,
            )

    print(
        f"wrote {record_count} compile-time record(s) from {file_count} file(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
