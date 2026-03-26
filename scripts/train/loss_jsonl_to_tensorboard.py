#!/usr/bin/env python3
"""Stream DreamZero loss_log.jsonl into TensorBoard event files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to loss_log.jsonl")
    parser.add_argument("--logdir", required=True, help="TensorBoard output directory")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval")
    parser.add_argument("--tag-prefix", default="train", help="TensorBoard tag prefix")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process currently available lines once and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))

    offset = 0
    while True:
        if not input_path.exists():
            if args.once:
                break
            time.sleep(args.poll_seconds)
            continue

        with input_path.open("r", encoding="utf-8") as fh:
            fh.seek(offset)
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                step = payload.get("step")
                if step is None:
                    continue
                for key, value in payload.items():
                    if key == "step" or not isinstance(value, (int, float)):
                        continue
                    writer.add_scalar(f"{args.tag_prefix}/{key}", value, step)
            offset = fh.tell()
        writer.flush()

        if args.once:
            break
        time.sleep(args.poll_seconds)

    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
