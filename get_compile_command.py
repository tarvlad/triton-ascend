#!/usr/bin/env python3
"""Print Ascend compile commands for a compilation timing record.

The input is the normalized NDJSON produced by concat_compile_times.py. Select a
compilation by NDJSON line number or kernel signature, then this script resolves
the record's artifact directory and prints the saved compile command(s).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys

from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timings", type=Path, help="Normalized compile timing NDJSON.")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="TRITON_DUMP_DIR used to create the timing log. Default: $TRITON_DUMP_DIR, or inferred.",
    )

    select = parser.add_mutually_exclusive_group()
    select.add_argument("--line", type=int, help="1-based NDJSON line number to select.")
    select.add_argument("--signature", help="Exact kernel_signature to select.")
    select.add_argument("--contains", help="Substring to match inside kernel_signature.")
    select.add_argument("--list", action="store_true", help="List timing records and exit.")

    parser.add_argument("--stage", help="Only print one stage, e.g. ttadapter_to_npubin.")
    parser.add_argument("--actual", action="store_true", help="Print the original tempdir command instead of replay command.")
    parser.add_argument("--json", action="store_true", help="Print selected command JSON instead of shell commands.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"bad JSON {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"invalid record at {path}:{line_no}: expected object")
            signature = record.get("kernel_signature")
            source = record.get("source")
            if not isinstance(signature, str) or not signature:
                raise SystemExit(f"missing kernel_signature at {path}:{line_no}")
            if not isinstance(source, str) or not source:
                raise SystemExit(f"missing source at {path}:{line_no}")
            records.append({"line": line_no, "record": record})
    if not records:
        raise SystemExit(f"no timing records found: {path}")
    return records


def stage_names(record: dict[str, Any]) -> list[str]:
    names = []
    for stage in record.get("stages_us", []):
        if isinstance(stage, dict) and isinstance(stage.get("name"), str):
            names.append(stage["name"])
    return names


def list_records(records: list[dict[str, Any]]) -> None:
    for item in records:
        record = item["record"]
        stages = ", ".join(stage_names(record))
        print(f"{item['line']}: {record['kernel_signature']} [{stages}]")


def select_record(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    if args.line is not None:
        for item in records:
            if item["line"] == args.line:
                return item
        raise SystemExit(f"line not found in {args.timings}: {args.line}")

    if args.signature is not None:
        matches = [item for item in records if item["record"]["kernel_signature"] == args.signature]
    elif args.contains is not None:
        matches = [item for item in records if args.contains in item["record"]["kernel_signature"]]
    else:
        matches = records

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit("no timing records matched")

    print("matched multiple timing records; select one with --line:", file=sys.stderr)
    list_records(matches)
    raise SystemExit(2)


def infer_dump_dir(args: argparse.Namespace, source: str) -> Path | None:
    if args.dump_dir is not None:
        return args.dump_dir
    env_dump_dir = os.environ.get("TRITON_DUMP_DIR")
    if env_dump_dir:
        return Path(env_dump_dir)

    source_path = Path(source)
    if source_path.is_absolute():
        return None

    candidate = args.timings.parent / source_path
    if candidate.exists() or candidate.parent.exists():
        return args.timings.parent
    return Path.cwd()


def timing_source_path(args: argparse.Namespace, record: dict[str, Any]) -> Path:
    source = str(record["source"])
    source_path = Path(source)
    if source_path.is_absolute():
        return source_path
    dump_dir = infer_dump_dir(args, source)
    if dump_dir is None:
        raise SystemExit(f"cannot resolve relative source path without dump dir: {source}")
    return dump_dir / source_path


def safe_stage_name(stage_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_name)


def load_command(path: Path) -> dict[str, Any]:
    try:
        command = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"bad command JSON {path}: {exc}") from exc
    if not isinstance(command, dict):
        raise SystemExit(f"invalid command JSON {path}: expected object")
    command["_command_file"] = str(path)
    return command


def command_files(artifact_dir: Path, stage: str | None) -> list[Path]:
    if stage is not None:
        path = artifact_dir / f"compile_command.{safe_stage_name(stage)}.json"
        if not path.is_file():
            raise SystemExit(f"command file not found for stage {stage}: {path}")
        return [path]

    paths = sorted(artifact_dir.glob("compile_command.*.json"))
    if not paths:
        raise SystemExit(f"no compile_command.*.json files found in {artifact_dir}")
    return paths


def ordered_commands(paths: list[Path], selected_stages: list[str]) -> list[dict[str, Any]]:
    commands = [load_command(path) for path in paths]
    order = {stage: index for index, stage in enumerate(selected_stages)}
    return sorted(commands, key=lambda command: (order.get(command.get("stage"), len(order)), command["_command_file"]))


def shell_for_command(command: dict[str, Any], *, actual: bool) -> str:
    shell_key = "shell" if actual else "replay_shell"
    argv_key = "argv" if actual else "replay_argv"
    cwd_key = "cwd" if actual else "replay_cwd"

    shell = command.get(shell_key)
    if not isinstance(shell, str):
        argv = command.get(argv_key)
        if not isinstance(argv, list):
            raise SystemExit(f"command has no {'actual' if actual else 'replay'} shell/argv: {command['_command_file']}")
        shell = shlex.join(str(arg) for arg in argv)

    cwd = command.get(cwd_key)
    if isinstance(cwd, str) and cwd:
        return f"cd {shlex.quote(cwd)} && {shell}"
    return shell


def print_commands(commands: list[dict[str, Any]], *, actual: bool, as_json: bool) -> None:
    for command in commands:
        if as_json:
            print(json.dumps(command, separators=(",", ":")))
            continue
        stage = command.get("stage", Path(command["_command_file"]).stem)
        print(f"# {stage}")
        print(shell_for_command(command, actual=actual))


def main() -> int:
    args = parse_args()
    records = load_records(args.timings)
    if args.list:
        list_records(records)
        return 0

    selected = select_record(args, records)
    record = selected["record"]
    artifact_dir = timing_source_path(args, record).parent
    paths = command_files(artifact_dir, args.stage)
    commands = ordered_commands(paths, stage_names(record))
    print_commands(commands, actual=args.actual, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
