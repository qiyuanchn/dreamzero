#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def choose_executor(executor_name: str, video_mode: str):
    if executor_name == "thread":
        return ThreadPoolExecutor
    if executor_name == "process":
        return ProcessPoolExecutor
    return ThreadPoolExecutor if video_mode == "source-link" else ProcessPoolExecutor


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


def load_npy_float32(path: Path, *, mmap: bool) -> np.ndarray:
    array = np.load(path, mmap_mode="r" if mmap else None)
    if array.dtype == np.float32:
        return array
    return array.astype(np.float32, copy=False)


def load_npz_key_float32(path: Path, key: str) -> np.ndarray:
    with np.load(path) as payload:
        array = payload[key]
    if array.dtype == np.float32:
        return array
    return array.astype(np.float32, copy=False)


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


def link_source_video(input_path: Path, output_path: Path, allow_copy: bool) -> None:
    """Reuse the original trajectory video without trimming or re-encoding.

    DreamZero reads frames by the parquet timestamp column, so a hard link to the
    full source video preserves the correct temporal coordinates and avoids the
    dominant conversion cost: one ffmpeg encode per generated episode.
    """
    if output_path.exists():
        output_path.unlink()
    try:
        os.link(input_path, output_path)
    except OSError as exc:
        if not allow_copy:
            raise OSError(
                f"Failed to hard-link {input_path} -> {output_path}. Keep --input and --output on the same filesystem, "
                "or pass --allow-copy-source-video if you intentionally want full video copies."
            ) from exc
        shutil.copy2(input_path, output_path)


def copy_metadata_files(output_root: Path, file_names: list[str]) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in file_names:
        shutil.copy2(output_root / name, meta_dir / name)


def empty_stats_acc(dim: int) -> dict:
    return {
        "count": 0,
        "sum": np.zeros(dim, dtype=np.float64),
        "sumsq": np.zeros(dim, dtype=np.float64),
        "min": np.full(dim, np.inf, dtype=np.float64),
        "max": np.full(dim, -np.inf, dtype=np.float64),
        "q01_sum": np.zeros(dim, dtype=np.float64),
        "q99_sum": np.zeros(dim, dtype=np.float64),
        "q_weight": 0,
    }


def summarize_stats(values: np.ndarray) -> dict:
    values = values.astype(np.float64, copy=False)
    count = values.shape[0]
    if count == 0:
        return empty_stats_acc(values.shape[1])
    return {
        "count": count,
        "sum": values.sum(axis=0),
        "sumsq": np.square(values).sum(axis=0),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "q01_sum": np.quantile(values, 0.01, axis=0) * count,
        "q99_sum": np.quantile(values, 0.99, axis=0) * count,
        "q_weight": count,
    }


def merge_stats_acc(target: dict, source: dict) -> None:
    target["count"] += int(source["count"])
    target["sum"] += source["sum"]
    target["sumsq"] += source["sumsq"]
    target["min"] = np.minimum(target["min"], source["min"])
    target["max"] = np.maximum(target["max"], source["max"])
    target["q01_sum"] += source["q01_sum"]
    target["q99_sum"] += source["q99_sum"]
    target["q_weight"] += int(source["q_weight"])


def finalize_stats(acc: dict) -> dict[str, list[float]]:
    if acc["count"] == 0:
        raise ValueError("Cannot finalize empty stats")
    mean = acc["sum"] / acc["count"]
    var = np.maximum(acc["sumsq"] / acc["count"] - np.square(mean), 0.0)
    q_weight = max(int(acc["q_weight"]), 1)
    return {
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
        "min": acc["min"].tolist(),
        "max": acc["max"].tolist(),
        "q01": (acc["q01_sum"] / q_weight).tolist(),
        "q99": (acc["q99_sum"] / q_weight).tolist(),
    }


def fixed_size_list_array(values: np.ndarray, value_type: pa.DataType) -> pa.FixedSizeListArray:
    contiguous = np.ascontiguousarray(values)
    return pa.FixedSizeListArray.from_arrays(pa.array(contiguous.reshape(-1), type=value_type), contiguous.shape[1])


def write_episode_parquet(
    parquet_path: Path,
    obs_segment: np.ndarray,
    action_segment: np.ndarray,
    timestamp_segment: np.ndarray,
    task_index: int,
    episode_index: int,
    global_start: int,
    task_text: str,
    compression: str | None,
) -> None:
    seg_len = len(timestamp_segment)
    table = pa.table(
        {
            "observation.state.obs": fixed_size_list_array(obs_segment, pa.float32()),
            "action": fixed_size_list_array(action_segment, pa.float32()),
            "timestamp": pa.array(timestamp_segment, type=pa.float64()),
            "task_index": pa.array(np.full(seg_len, task_index, dtype=np.int64)),
            "episode_index": pa.array(np.full(seg_len, episode_index, dtype=np.int64)),
            "frame_index": pa.array(np.arange(seg_len, dtype=np.int64)),
            "index": pa.array(np.arange(global_start, global_start + seg_len, dtype=np.int64)),
            "next.reward": pa.array(np.zeros(seg_len, dtype=np.float64)),
            "next.done": pa.array([False] * (seg_len - 1) + [True], type=pa.bool_()),
            "is_terminal": pa.array([False] * (seg_len - 1) + [True], type=pa.bool_()),
            "is_first": pa.array([True] + [False] * (seg_len - 1), type=pa.bool_()),
            "discount": pa.array(np.ones(seg_len, dtype=np.float64)),
            "annotation.task": pa.array([task_text] * seg_len, type=pa.string()),
        }
    )
    pq.write_table(table, parquet_path, compression=compression, use_dictionary=False, write_statistics=False)


def load_task_text(traj_dir: Path, obs_segment: np.ndarray) -> str:
    task_path = traj_dir / "task.txt"
    if task_path.exists():
        task_text = task_path.read_text(encoding="utf-8", errors="replace").strip()
        if task_text:
            return task_text
    return build_task_text(obs_segment)


def load_rpyva_actions(rpyva: np.ndarray) -> np.ndarray:
    rpyva = rpyva.astype(np.float32)
    if rpyva.ndim != 3 or rpyva.shape[1:] != (8, 9):
        raise ValueError(f"Unsupported rpyva.npy shape: {rpyva.shape}")
    return rpyva.reshape(rpyva.shape[0], 72)


def detect_traj_format(traj_dir: Path) -> str:
    if (traj_dir / "traj_coefs.npz").exists():
        coefs_npz = np.load(traj_dir / "traj_coefs.npz")
        if "rpyva" in coefs_npz:
            return "collector_rpyva_npz"
    if (traj_dir / "rpyva.npy").exists():
        return "tracking_rpyva_npy"
    raise FileNotFoundError(f"Unsupported trajectory format under {traj_dir}")


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
    traj_dirs: set[Path] = set()
    if not root.is_dir():
        return []

    for video_path in sorted(root.rglob("rgb.mp4")):
        candidate = video_path.parent
        if not candidate.is_dir():
            continue
        if (candidate / "obs.npy").exists() and (candidate / "traj_coefs.npz").exists():
            traj_dirs.add(candidate)
            continue
        if (candidate / "obs.npy").exists() and (candidate / "rpyva.npy").exists():
            traj_dirs.add(candidate)

    return sorted(traj_dirs)


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


def episode_chunk(episode_index: int, chunk_size: int) -> int:
    return episode_index // chunk_size


def scan_trajectory_job(args: tuple[str, int, int, int]) -> dict:
    traj_dir = Path(args[0])
    window_size, stride, min_frames = args[1:]
    traj_format = detect_traj_format(traj_dir)
    obs = load_npy_float32(traj_dir / "obs.npy", mmap=True)
    if obs.ndim != 2 or obs.shape[1] != 12:
        raise ValueError(f"Expected obs.npy with shape [T,12], got {obs.shape} in {traj_dir}")
    if traj_format == "collector_rpyva_npz":
        rpyva = load_npz_key_float32(traj_dir / "traj_coefs.npz", "rpyva")
    elif traj_format == "tracking_rpyva_npy":
        rpyva = load_npy_float32(traj_dir / "rpyva.npy", mmap=True)
    else:
        raise ValueError(f"Unsupported trajectory format {traj_format} in {traj_dir}")
    if rpyva.ndim != 3 or rpyva.shape[1:] != (8, 9):
        raise ValueError(f"Expected rpyva with shape [T,8,9], got {rpyva.shape} in {traj_dir}")

    obs_offset = obs.shape[0] - rpyva.shape[0]
    if obs_offset < 0:
        raise ValueError(f"obs.npy shorter than rpyva in {traj_dir}: {obs.shape} vs {rpyva.shape}")
    usable_len = rpyva.shape[0]

    segments = [(start, seg_len) for start, seg_len in iter_segments(usable_len, window_size, stride) if seg_len >= min_frames]
    task_path = traj_dir / "task.txt"
    task_text = task_path.read_text(encoding="utf-8", errors="replace").strip() if task_path.exists() else ""
    if not task_text and segments:
        obs_aligned = np.asarray(obs[obs_offset:obs_offset + usable_len], dtype=np.float32)
        start, seg_len = segments[0]
        task_text = build_task_text(obs_aligned[start:start + seg_len])

    return {
        "traj_dir": str(traj_dir),
        "traj_format": traj_format,
        "segments": segments,
        "frames": int(sum(seg_len for _, seg_len in segments)),
        "task_text": task_text,
        "obs_offset": int(obs_offset),
    }


def convert_trajectory_job(job: dict) -> dict:
    traj_dir = Path(job["traj_dir"])
    traj_format = str(job["traj_format"])
    output_root = Path(job["output_root"])
    chunk_size = int(job["chunk_size"])
    episode_start = int(job["episode_start"])
    global_start = int(job["global_start"])
    task_index = int(job["task_index"])
    task_text = str(job["task_text"])
    fps = float(job["fps"])
    video_mode = str(job["video_mode"])
    ffmpeg_threads = int(job["ffmpeg_threads"])
    allow_copy_source_video = bool(job["allow_copy_source_video"])
    parquet_compression = job["parquet_compression"]
    obs_offset = int(job["obs_offset"])
    segments = job["segments"]

    obs = load_npy_float32(traj_dir / "obs.npy", mmap=True)
    if obs.ndim != 2 or obs.shape[1] != 12:
        raise ValueError(f"Expected obs.npy with shape [T,12], got {obs.shape} in {traj_dir}")
    if traj_format == "collector_rpyva_npz":
        rpyva = load_npz_key_float32(traj_dir / "traj_coefs.npz", "rpyva")
    elif traj_format == "tracking_rpyva_npy":
        rpyva = load_npy_float32(traj_dir / "rpyva.npy", mmap=True)
    else:
        raise ValueError(f"Unsupported trajectory format {traj_format} in {traj_dir}")
    if rpyva.ndim != 3 or rpyva.shape[1:] != (8, 9):
        raise ValueError(f"Expected rpyva with shape [T,8,9], got {rpyva.shape} in {traj_dir}")
    if obs_offset < 0 or obs_offset + rpyva.shape[0] > obs.shape[0]:
        raise ValueError(f"Invalid obs/rpyva alignment in {traj_dir}: obs_offset={obs_offset}, obs={obs.shape}, rpyva={rpyva.shape}")

    obs_aligned = np.asarray(obs[obs_offset:obs_offset + rpyva.shape[0]], dtype=np.float32)
    action_values = load_rpyva_actions(rpyva)
    timestamps = np.arange(obs_offset, obs_offset + rpyva.shape[0], dtype=np.float64) / fps
    video_frame_offset = obs_offset

    episodes = []
    obs_stats = empty_stats_acc(12)
    action_stats = empty_stats_acc(72)
    timestamp_stats = empty_stats_acc(1)
    local_global_index = global_start

    for offset, (start, seg_len) in enumerate(segments):
        episode_index = episode_start + offset
        obs_segment = obs_aligned[start:start + seg_len]
        action_segment = action_values[start:start + seg_len]
        timestamp_segment = timestamps[start:start + seg_len]
        chunk_index = episode_chunk(episode_index, chunk_size)
        data_chunk_dir = output_root / "data" / f"chunk-{chunk_index:03d}"
        video_chunk_dir = output_root / "videos" / f"chunk-{chunk_index:03d}" / "observation.images.rgb"
        parquet_path = data_chunk_dir / f"episode_{episode_index:06d}.parquet"
        video_path = video_chunk_dir / f"episode_{episode_index:06d}.mp4"

        write_episode_parquet(
            parquet_path,
            obs_segment,
            action_segment,
            timestamp_segment,
            task_index,
            episode_index,
            local_global_index,
            task_text,
            parquet_compression,
        )
        if video_mode == "source-link":
            link_source_video(traj_dir / "rgb.mp4", video_path, allow_copy_source_video)
        else:
            trim_video(traj_dir / "rgb.mp4", video_path, int(video_frame_offset + start), seg_len, fps, ffmpeg_threads)

        merge_stats_acc(obs_stats, summarize_stats(obs_segment))
        merge_stats_acc(action_stats, summarize_stats(action_segment))
        merge_stats_acc(timestamp_stats, summarize_stats(timestamp_segment.reshape(-1, 1)))
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [task_index],
                "length": seg_len,
                "source_traj": traj_dir.name,
                "window_start": start,
            }
        )
        local_global_index += seg_len

    return {
        "traj_dir": str(traj_dir),
        "episodes": episodes,
        "obs_stats": obs_stats,
        "action_stats": action_stats,
        "timestamp_stats": timestamp_stats,
        "frames": int(sum(seg_len for _, seg_len in segments)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert drone trajectory data to DreamZero-ready LeRobot format.")
    parser.add_argument("--input", required=True, help="Path to output.tar.gz, a test run dir, a trajs dir, or an extracted directory.")
    parser.add_argument("--output", required=True, help="Output dataset root.")
    parser.add_argument("--window-size", type=int, default=160, help="Frames per generated episode window.")
    parser.add_argument("--stride", type=int, default=120, help="Stride between generated windows.")
    parser.add_argument("--min-frames", type=int, default=64, help="Skip segments shorter than this many frames.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of episodes per LeRobot chunk directory.")
    parser.add_argument(
        "--task-variants",
        type=int,
        default=1,
        help="Number of text variants to emit per segment. Keep at 1 to avoid duplicating identical trajectories.",
    )
    parser.add_argument("--ffmpeg-threads", type=int, default=2, help="Threads passed to ffmpeg for each trim job.")
    parser.add_argument(
        "--video-mode",
        choices=["source-link", "trim-reencode"],
        default="source-link",
        help=(
            "source-link hard-links each episode video to the original rgb.mp4 and keeps original timestamps; "
            "trim-reencode creates physically clipped videos with ffmpeg."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel ffmpeg jobs for --video-mode trim-reencode. Ignored by source-link.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Parallel trajectory conversion workers. Increase this on large CPU machines for source-link mode.",
    )
    parser.add_argument(
        "--executor",
        choices=["auto", "thread", "process"],
        default="auto",
        help="Parallel executor backend. auto uses threads for source-link and processes for trim-reencode.",
    )
    parser.add_argument(
        "--parquet-compression",
        choices=["none", "snappy", "zstd"],
        default="none",
        help="Parquet compression. none is fastest and uses more space; snappy/zstd are smaller but slower.",
    )
    parser.add_argument(
        "--allow-copy-source-video",
        action="store_true",
        help="Allow source-link mode to copy full videos if hard links fail. Avoid this unless input/output are on different filesystems.",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="Directory for temporary extraction when --input is a tarball. Defaults to the output parent directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.task_variants <= 0:
        raise ValueError("--task-variants must be positive")
    if args.task_variants != 1:
        print("WARNING: --task-variants is kept for CLI compatibility but drone conversion now emits one episode per segment.")
    if args.video_mode == "trim-reencode" and args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if output_root.exists():
        shutil.rmtree(output_root)

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

        scan_args = [(str(traj_dir), args.window_size, args.stride, args.min_frames) for traj_dir in trajectory_dirs]
        print(f"Found {len(trajectory_dirs)} source trajectories under {source_root}", flush=True)
        executor_cls = choose_executor(args.executor, args.video_mode)
        if args.workers == 1:
            scan_results = [
                scan_trajectory_job(scan_arg)
                for scan_arg in tqdm(scan_args, desc="Scanning trajectories", unit="traj")
            ]
        else:
            with executor_cls(max_workers=args.workers) as executor:
                futures = [executor.submit(scan_trajectory_job, scan_arg) for scan_arg in scan_args]
                scan_results = [
                    future.result()
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning trajectories", unit="traj")
                ]
        scan_results = [result for result in scan_results if result["segments"]]
        if not scan_results:
            raise ValueError("No usable trajectory segments found after applying --window-size/--stride/--min-frames")

        tasks: dict[str, int] = {}
        for result in scan_results:
            tasks.setdefault(result["task_text"], len(tasks))

        jobs = []
        episode_index = 0
        global_index = 0
        total_frames = 0
        for result in scan_results:
            segments = result["segments"]
            frames = int(result["frames"])
            jobs.append(
                {
                    "traj_dir": result["traj_dir"],
                    "traj_format": result["traj_format"],
                    "output_root": str(output_root),
                    "chunk_size": args.chunk_size,
                    "episode_start": episode_index,
                    "global_start": global_index,
                    "task_index": tasks[result["task_text"]],
                    "task_text": result["task_text"],
                    "segments": segments,
                    "fps": fps,
                    "video_mode": args.video_mode,
                    "ffmpeg_threads": args.ffmpeg_threads,
                    "allow_copy_source_video": args.allow_copy_source_video,
                    "parquet_compression": None if args.parquet_compression == "none" else args.parquet_compression,
                    "obs_offset": result["obs_offset"],
                }
            )
            episode_index += len(segments)
            global_index += frames
            total_frames += frames

        total_chunk_dirs = int(math.ceil(episode_index / args.chunk_size)) if episode_index else 0
        for chunk_index in range(total_chunk_dirs):
            (output_root / "data" / f"chunk-{chunk_index:03d}").mkdir(parents=True, exist_ok=True)
            (output_root / "videos" / f"chunk-{chunk_index:03d}" / "observation.images.rgb").mkdir(parents=True, exist_ok=True)

        episodes = []
        obs_stats = empty_stats_acc(12)
        action_stats = empty_stats_acc(72)
        timestamp_stats = empty_stats_acc(1)
        completed = 0
        completed_episodes = 0
        completed_frames = 0
        print(
            f"Converting {len(jobs)} trajectories -> {episode_index} episodes / {total_frames} frames "
            f"with workers={args.workers}, video_mode={args.video_mode}, parquet_compression={args.parquet_compression}",
            flush=True,
        )
        if args.workers == 1:
            results = [
                convert_trajectory_job(job)
                for job in tqdm(jobs, desc="Converting trajectories", unit="traj")
            ]
        else:
            results = []
            with executor_cls(max_workers=args.workers) as executor:
                future_to_job = {executor.submit(convert_trajectory_job, job): job for job in jobs}
                progress = tqdm(as_completed(future_to_job), total=len(future_to_job), desc="Converting trajectories", unit="traj")
                for future in progress:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    completed_episodes += len(result["episodes"])
                    completed_frames += int(result["frames"])
                    progress.set_postfix(
                        episodes=f"{completed_episodes}/{episode_index}",
                        frames=f"{completed_frames}/{total_frames}",
                    )

        results.sort(key=lambda result: result["episodes"][0]["episode_index"] if result["episodes"] else -1)
        for result in results:
            episodes.extend(result["episodes"])
            merge_stats_acc(obs_stats, result["obs_stats"])
            merge_stats_acc(action_stats, result["action_stats"])
            merge_stats_acc(timestamp_stats, result["timestamp_stats"])
        tasks_jsonl = [{"task_index": idx, "task": text} for text, idx in sorted(tasks.items(), key=lambda x: x[1])]
        source_formats = sorted({result["traj_format"] for result in scan_results})

        info = {
            "codebase_version": "v2.0",
            "robot_type": "drone",
            "total_episodes": episode_index,
            "total_frames": int(total_frames),
            "total_tasks": len(tasks_jsonl),
            "total_videos": episode_index,
            "total_chunks": int(math.ceil(episode_index / args.chunk_size)) if episode_index else 0,
            "chunks_size": args.chunk_size,
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
                "observation.state.obs": {"dtype": "float32", "shape": [12], "names": [f"obs_{i}" for i in range(12)]},
                "action": {
                    "dtype": "float32",
                    "shape": [72],
                    "names": [f"rpyva_wp{wp}_{name}" for wp in range(8) for name in ["roll", "pitch", "yaw", "vx", "vy", "vz", "ax", "ay", "az"]],
                },
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
                "obs": {
                    "original_key": "observation.state.obs",
                    "start": 0,
                    "end": 12,
                    "rotation_type": None,
                    "absolute": True,
                    "dtype": "float32",
                    "range": None,
                }
            },
            "action": {
                "rpyva": {
                    "original_key": "action",
                    "start": 0,
                    "end": 72,
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
            "observation.state.obs": finalize_stats(obs_stats),
            "action": finalize_stats(action_stats),
            "timestamp": finalize_stats(timestamp_stats),
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
            "source_formats": source_formats,
            "source_root": str(source_root),
            "observation_source": (
                "aligned obs.npy rows after dropping the leading offset obs_len - rpyva_len; stored as observation.state.obs (12D)"
            ),
            "action_source": (
                "collector_rpyva_npz: traj_coefs.npz['rpyva'] flattened from [T,8,9] to 72D; "
                "tracking_rpyva_npy: rpyva.npy flattened from [T,8,9] to 72D"
            ),
            "timestamp_source": (
                "inferred from rgb.mp4 fps and the alignment offset obs_len - rpyva_len"
            ),
            "language_source": "task.txt when present, otherwise generated from displacement",
            "video_mode": args.video_mode,
            "video_note": "source-link stores hard links to full source rgb.mp4 files and relies on parquet timestamps" if args.video_mode == "source-link" else "trim-reencode stores clipped per-episode mp4 files",
            "jobs": args.jobs if args.video_mode == "trim-reencode" else 1,
            "workers": args.workers,
            "parquet_compression": args.parquet_compression,
            "stats_note": "mean/std/min/max are exact; q01/q99 are weighted per-trajectory approximations for faster conversion",
        }
        (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
