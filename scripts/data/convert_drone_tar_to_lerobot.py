#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def choose_extract_program() -> list[str] | None:
    # Prefer multithreaded gzip decompression when available.
    if shutil.which("pigz"):
        return ["pigz", "-d"]
    return None


def probe_video(video_path: Path) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return int(stream["width"]), int(stream["height"]), fps


def trim_video(input_path: Path, output_path: Path, start_frame: int, num_frames: int, fps: float, ffmpeg_threads: int) -> None:
    start_time = start_frame / fps
    duration = max(num_frames / fps, 1.0 / fps)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-threads",
        str(ffmpeg_threads),
        "-ss",
        f"{start_time:.6f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        str(output_path),
    ]
    run(cmd)


def copy_metadata_files(output_root: Path, file_names: list[str]) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in file_names:
        shutil.copy2(output_root / name, meta_dir / name)


def compute_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = values.astype(np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def build_task_text(segment: np.ndarray) -> str:
    disp = segment[-1, :3] - segment[0, :3]
    axes = ["x", "y", "z"]
    dominant = int(np.argmax(np.abs(disp)))
    direction = "positive" if disp[dominant] >= 0 else "negative"
    distance = float(np.linalg.norm(disp))
    speed = float(np.mean(np.linalg.norm(np.diff(segment[:, :3], axis=0), axis=1))) if len(segment) > 1 else 0.0
    if distance > 2.0:
        scale = "large"
    elif distance > 0.5:
        scale = "medium"
    else:
        scale = "small"
    if speed > 0.08:
        motion = "fast"
    elif speed > 0.03:
        motion = "steady"
    else:
        motion = "slow"
    return f"follow a {scale} {motion} drone trajectory moving in {direction} {axes[dominant]} direction"


def discover_trajectory_dirs(root: Path) -> list[Path]:
    traj_dirs = []
    for candidate in sorted(root.rglob("trajs/*")):
        if candidate.is_dir() and (candidate / "rgb.mp4").exists() and (candidate / "states.npy").exists():
            traj_dirs.append(candidate)
    return traj_dirs


def resolve_source_root(input_path: Path) -> Path:
    """Resolve flexible user inputs to a directory containing trajs/ or nested test outputs."""
    if input_path.is_file():
        return input_path

    candidates = [
        input_path,
        input_path / "trajs",
        input_path / "outputs" / "test",
        input_path / "source" / "outputs" / "test",
    ]

    for candidate in candidates:
        if candidate.is_dir() and discover_trajectory_dirs(candidate):
            return candidate

    if input_path.name == "trajs" and input_path.is_dir():
        return input_path.parent

    return input_path


def iter_segments(length: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if length <= window_size:
        return [(0, length)]
    segments = []
    start = 0
    while start + window_size < length:
        segments.append((start, window_size))
        start += stride
    tail_start = max(length - window_size, 0)
    if not segments or segments[-1][0] != tail_start:
        segments.append((tail_start, min(window_size, length - tail_start)))
    return segments


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert drone trajectory data to DreamZero-ready LeRobot format.")
    parser.add_argument("--input", required=True, help="Path to output.tar.gz, a test run dir, a trajs dir, or an extracted directory.")
    parser.add_argument("--output", required=True, help="Output dataset root.")
    parser.add_argument("--window-size", type=int, default=160, help="Frames per generated episode window.")
    parser.add_argument("--stride", type=int, default=120, help="Stride between generated windows.")
    parser.add_argument("--min-frames", type=int, default=64, help="Skip segments shorter than this many frames.")
    parser.add_argument("--task-variants", type=int, default=2, help="Duplicate each segment with paraphrased task text.")
    parser.add_argument("--ffmpeg-threads", type=int, default=2, help="Threads passed to ffmpeg for each trim job.")
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="Directory for temporary extraction when --input is a tarball. Defaults to the output parent directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_root / "videos" / "chunk-000" / "observation.images.rgb").mkdir(parents=True, exist_ok=True)

    extract_parent = (
        Path(args.extract_dir).expanduser().resolve()
        if args.extract_dir is not None
        else output_root.parent.resolve()
    )
    extract_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dreamzero_drone_", dir=str(extract_parent)) as tmpdir:
        tmp_root = Path(tmpdir)
        source_root = input_path
        if input_path.is_file():
            extract_program = choose_extract_program()
            if extract_program is not None:
                run(["tar", "--use-compress-program", " ".join(extract_program), "-xf", str(input_path), "-C", str(tmp_root)])
            else:
                run(["tar", "-xzf", str(input_path), "-C", str(tmp_root)])
            source_root = tmp_root
        else:
            source_root = resolve_source_root(input_path)

        trajectory_dirs = discover_trajectory_dirs(source_root)
        if not trajectory_dirs:
            raise FileNotFoundError(f"No trajectories found under {source_root}")

        first_video = trajectory_dirs[0] / "rgb.mp4"
        width, height, fps = probe_video(first_video)

        tasks: dict[str, int] = {}
        episodes = []
        state_rows = []
        action_rows = []
        total_frames = 0
        global_index = 0
        episode_index = 0

        for traj_dir in trajectory_dirs:
            states = np.load(traj_dir / "states.npy").astype(np.float32)
            proprio = np.load(traj_dir / "obs.npy").astype(np.float32)
            coefs_npz = np.load(traj_dir / "traj_coefs.npz")
            coef_matrices = coefs_npz["coef_matrices"].astype(np.float32).reshape(-1, 18)
            valid_indices = coefs_npz["valid_indices"].astype(np.int64)
            timestamps = coefs_npz["timestamps"].astype(np.float64)
            if states.ndim != 2 or states.shape[1] != 16:
                raise ValueError(f"Expected states.npy with shape [T,16], got {states.shape} in {traj_dir}")
            if proprio.ndim != 2 or proprio.shape[0] != states.shape[0]:
                raise ValueError(f"obs.npy shape mismatch in {traj_dir}: {proprio.shape} vs states {states.shape}")
            if coef_matrices.shape[0] != len(valid_indices) or coef_matrices.shape[0] != len(timestamps):
                raise ValueError(f"traj_coefs.npz shape mismatch in {traj_dir}")
            if not np.all(np.diff(valid_indices) == 1):
                raise ValueError(f"valid_indices must be contiguous in current converter: {traj_dir}")

            usable_len = len(valid_indices)
            states = states[valid_indices]
            proprio = proprio[valid_indices]
            segments = iter_segments(usable_len, args.window_size, args.stride)
            for start, seg_len in segments:
                if seg_len < args.min_frames:
                    continue
                state_segment = states[start:start + seg_len]
                proprio_segment = proprio[start:start + seg_len]
                action_segment = coef_matrices[start:start + seg_len]
                timestamp_segment = timestamps[start:start + seg_len]
                task_text = build_task_text(state_segment)
                task_texts = [task_text]
                if args.task_variants >= 2:
                    task_texts.append(task_text.replace("follow", "track"))
                if args.task_variants >= 3:
                    task_texts.append(task_text.replace("trajectory", "motion pattern"))

                canonical_video_path = None
                for variant_id, variant_text in enumerate(task_texts[:args.task_variants]):
                    task_index = tasks.setdefault(variant_text, len(tasks))
                    parquet_path = output_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
                    video_path = output_root / "videos" / "chunk-000" / "observation.images.rgb" / f"episode_{episode_index:06d}.mp4"

                    df = pd.DataFrame(
                        {
                            "observation.state.proprio": list(proprio_segment),
                            "observation.state.dynamics": list(state_segment),
                            "action": list(action_segment),
                            "timestamp": timestamp_segment,
                            "task_index": np.full(seg_len, task_index, dtype=np.int64),
                            "episode_index": np.full(seg_len, episode_index, dtype=np.int64),
                            "frame_index": np.arange(seg_len, dtype=np.int64),
                            "index": np.arange(global_index, global_index + seg_len, dtype=np.int64),
                            "next.reward": np.zeros(seg_len, dtype=np.float64),
                            "next.done": np.array([False] * (seg_len - 1) + [True], dtype=bool),
                            "is_terminal": np.array([False] * (seg_len - 1) + [True], dtype=bool),
                            "is_first": np.array([True] + [False] * (seg_len - 1), dtype=bool),
                            "discount": np.ones(seg_len, dtype=np.float64),
                            "annotation.task": [variant_text] * seg_len,
                        }
                    )
                    df.to_parquet(parquet_path, index=False)
                    if canonical_video_path is None:
                        trim_video(
                            traj_dir / "rgb.mp4",
                            video_path,
                            int(valid_indices[start]),
                            seg_len,
                            fps,
                            args.ffmpeg_threads,
                        )
                        canonical_video_path = video_path
                    else:
                        link_or_copy(canonical_video_path, video_path)

                    state_rows.append(np.concatenate([proprio_segment, state_segment], axis=1))
                    action_rows.append(action_segment)
                    episodes.append(
                        {
                            "episode_index": episode_index,
                            "tasks": [task_index],
                            "length": seg_len,
                            "source_traj": traj_dir.name,
                            "window_start": start,
                            "task_variant": variant_id,
                        }
                    )
                    total_frames += seg_len
                    global_index += seg_len
                    episode_index += 1

        all_states = np.concatenate(state_rows, axis=0)
        all_actions = np.concatenate(action_rows, axis=0)
        tasks_jsonl = [{"task_index": idx, "task": text} for text, idx in sorted(tasks.items(), key=lambda x: x[1])]

        info = {
            "codebase_version": "v2.0",
            "robot_type": "drone",
            "total_episodes": episode_index,
            "total_frames": int(total_frames),
            "total_tasks": len(tasks_jsonl),
            "total_videos": 1,
            "total_chunks": 1,
            # This converter currently writes every episode into chunk-000.
            # Keep chunks_size aligned with the on-disk layout so the sharded
            # dataset loader does not look for non-existent chunk-XYZ folders.
            "chunks_size": max(episode_index, 1),
            "fps": fps,
            "splits": {"train": "0:100"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.rgb": {
                    "dtype": "video",
                    "shape": [height, width, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                },
                "observation.state.proprio": {"dtype": "float32", "shape": [12], "names": [f"proprio_{i}" for i in range(12)]},
                "observation.state.dynamics": {"dtype": "float32", "shape": [16], "names": [f"state_{i}" for i in range(16)]},
                "action": {"dtype": "float32", "shape": [18], "names": [f"coef_{axis}_{order}" for axis in ["x", "y", "z"] for order in range(6)]},
                "timestamp": {"dtype": "float64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "next.reward": {"dtype": "float64", "shape": [1]},
                "next.done": {"dtype": "bool", "shape": [1]},
                "is_terminal": {"dtype": "bool", "shape": [1]},
                "is_first": {"dtype": "bool", "shape": [1]},
                "discount": {"dtype": "float64", "shape": [1]},
                "annotation.task": {"dtype": "str", "shape": [1]},
            },
        }
        modality = {
            "state": {
                "proprio": {
                    "original_key": "observation.state.proprio",
                    "start": 0,
                    "end": 12,
                    "rotation_type": None,
                    "absolute": True,
                    "dtype": "float32",
                    "range": None,
                },
                "dynamics": {
                    "original_key": "observation.state.dynamics",
                    "start": 0,
                    "end": 16,
                    "rotation_type": None,
                    "absolute": True,
                    "dtype": "float32",
                    "range": None,
                }
            },
            "action": {
                "trajectory_coeffs": {
                    "original_key": "action",
                    "start": 0,
                    "end": 18,
                    "rotation_type": None,
                    "absolute": True,
                    "dtype": "float32",
                    "range": None,
                }
            },
            "video": {"rgb": {"original_key": "observation.images.rgb"}},
            "annotation": {"task": {"original_key": "annotation.task"}},
        }
        stats = {
            "observation.state.proprio": compute_stats(all_states[:, :12]),
            "observation.state.dynamics": compute_stats(all_states[:, 12:]),
            "action": compute_stats(all_actions),
            "timestamp": compute_stats(np.concatenate([np.load(traj_dir / "traj_coefs.npz")["timestamps"].reshape(-1, 1) for traj_dir in trajectory_dirs], axis=0)),
        }
        embodiment = {"embodiment_tag": "drone"}

        (output_root / "info.json").write_text(json.dumps(info, indent=2))
        (output_root / "modality.json").write_text(json.dumps(modality, indent=2))
        (output_root / "stats.json").write_text(json.dumps(stats, indent=2))
        (output_root / "relative_stats_dreamzero.json").write_text(json.dumps({}, indent=2))
        (output_root / "embodiment.json").write_text(json.dumps(embodiment, indent=2))
        with open(output_root / "tasks.jsonl", "w") as f:
            for row in tasks_jsonl:
                f.write(json.dumps(row) + "\n")
        with open(output_root / "episodes.jsonl", "w") as f:
            for row in episodes:
                f.write(json.dumps(row) + "\n")

        copy_metadata_files(
            output_root,
            [
                "info.json",
                "modality.json",
                "stats.json",
                "relative_stats_dreamzero.json",
                "embodiment.json",
                "tasks.jsonl",
                "episodes.jsonl",
            ],
        )

        summary = {
            "episodes": episode_index,
            "tasks": len(tasks_jsonl),
            "frames": int(total_frames),
            "fps": fps,
            "video_resolution": [width, height],
            "source_trajectories": len(trajectory_dirs),
        }
        (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
