#!/usr/bin/env python3
"""Offline open-loop evaluation for DreamZero on drone data.

Loads a DreamZero checkpoint, reads the converted drone LeRobot dataset,
runs one or more offline inference samples, compares predicted rpyva actions
against the stored ground truth, and optionally decodes the generated latent
video for inspection.
"""

import torch._dynamo
torch._dynamo.config.disable = True

import argparse
import bisect
import glob
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "dreamzero_mpl"))

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


VIDEO_KEY = "video.rgb"
VIDEO_FOLDER = "observation.images.rgb"
ACTIONS_PER_WAYPOINT = 9
NUM_WAYPOINTS = 8
ACTION_KEY = "action.rpyva"
PARQUET_STATE_KEYS = {
    "state.obs": "observation.state.obs",
}


class DroneDataset:
    """Reads converted drone data in LeRobot chunked parquet + MP4 format."""

    def __init__(self, dataset_path: str):
        self.root = Path(dataset_path)

        parquet_files = sorted(glob.glob(str(self.root / "data" / "**" / "episode_*.parquet"), recursive=True))
        if not parquet_files:
            raise FileNotFoundError(f"No episode_*.parquet found under {self.root / 'data'}")

        self.parquet_files = [Path(p) for p in parquet_files]
        self.episode_lengths = []
        self.cum_lengths = [0]
        for parquet_path in self.parquet_files:
            num_rows = pq.ParquetFile(parquet_path).metadata.num_rows
            self.episode_lengths.append(num_rows)
            self.cum_lengths.append(self.cum_lengths[-1] + num_rows)
        self.total_rows = self.cum_lengths[-1]
        self.table_cache: OrderedDict[int, object] = OrderedDict()
        self.max_cached_tables = 4

        self.video_files = [self._video_path_for_parquet(p) for p in self.parquet_files]
        missing_videos = [str(p) for p in self.video_files if not p.exists()]
        if missing_videos:
            raise FileNotFoundError(f"Missing video files for converted drone dataset, first missing path: {missing_videos[0]}")

    def __len__(self) -> int:
        return self.total_rows

    def _video_path_for_parquet(self, parquet_path: Path) -> Path:
        dataset_root = parquet_path.parents[2]
        chunk_name = parquet_path.parent.name
        return dataset_root / "videos" / chunk_name / VIDEO_FOLDER / f"{parquet_path.stem}.mp4"

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(f"Index {idx} out of range ({self.total_rows})")
        ep = bisect.bisect_right(self.cum_lengths, idx) - 1
        return ep, idx - self.cum_lengths[ep]

    def _get_table(self, ep: int):
        if ep in self.table_cache:
            table = self.table_cache.pop(ep)
            self.table_cache[ep] = table
            return table

        table = pq.read_table(self.parquet_files[ep])
        self.table_cache[ep] = table
        while len(self.table_cache) > self.max_cached_tables:
            self.table_cache.popitem(last=False)
        return table

    def get_task(self, idx: int) -> str:
        ep, row = self._locate(idx)
        return str(self._get_table(ep).column("annotation.task")[row].as_py())

    def get_state_dict(self, idx: int) -> dict[str, np.ndarray]:
        ep, row = self._locate(idx)
        table = self._get_table(ep)
        return {
            key: np.array(table.column(parquet_key)[row].as_py(), dtype=np.float64).reshape(1, -1)
            for key, parquet_key in PARQUET_STATE_KEYS.items()
        }

    def get_action(self, idx: int) -> np.ndarray:
        ep, row = self._locate(idx)
        return np.array(self._get_table(ep).column("action")[row].as_py(), dtype=np.float64)

    def get_frame(self, idx: int) -> np.ndarray:
        ep, row = self._locate(idx)
        mp4 = self.video_files[ep]
        cap = cv2.VideoCapture(str(mp4))
        cap.set(cv2.CAP_PROP_POS_FRAMES, row)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {row} from {mp4}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_obs(dataset: DroneDataset, idx: int, prompt: str) -> dict:
    obs = {
        VIDEO_KEY: dataset.get_frame(idx)[np.newaxis, ...].astype(np.uint8),
        "annotation.task": prompt,
    }
    obs.update(dataset.get_state_dict(idx))
    return obs


def decode_video_latents(policy: GrootSimPolicy, video_pred: torch.Tensor | None) -> np.ndarray | None:
    if video_pred is None:
        return None

    latents = video_pred
    if not torch.is_tensor(latents):
        latents = torch.as_tensor(latents)
    latents = latents.to(device=policy.device)
    if latents.dtype != torch.bfloat16:
        latents = latents.to(dtype=torch.bfloat16)

    with torch.inference_mode():
        decoded = policy.trained_model.action_head.vae.decode(latents, tiled=False)

    decoded = decoded.detach().float().cpu()
    decoded = decoded.clamp(-1, 1)
    decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
    decoded = decoded.permute(0, 2, 3, 4, 1).numpy()
    return decoded[0]


def collapse_tiled_drone_frames(frames: np.ndarray) -> np.ndarray:
    # Older drone checkpoints used a 2x2 duplicated canvas for WAN compatibility.
    # Detect that legacy layout heuristically and collapse it for visualization.
    if frames.ndim != 4:
        return frames
    _, h, w, _ = frames.shape
    if h % 2 != 0 or w % 2 != 0:
        return frames
    top_left = frames[:, : h // 2, : w // 2, :].astype(np.float32)
    top_right = frames[:, : h // 2, w // 2 :, :].astype(np.float32)
    bottom_left = frames[:, h // 2 :, : w // 2, :].astype(np.float32)
    bottom_right = frames[:, h // 2 :, w // 2 :, :].astype(np.float32)
    diffs = [
        np.mean(np.abs(top_left - top_right)),
        np.mean(np.abs(top_left - bottom_left)),
        np.mean(np.abs(top_left - bottom_right)),
    ]
    if max(diffs) < 2.0:
        return frames[:, : h // 2, : w // 2, :]
    return frames


def save_rgb_video(frames: np.ndarray, path: Path, fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def save_action_plot(pred: np.ndarray, gt: np.ndarray, path: Path) -> dict[str, float]:
    mse = float(np.mean((pred - gt) ** 2))
    mae = float(np.mean(np.abs(pred - gt)))

    plt.figure(figsize=(14, 5))
    dims = min(pred.shape[-1], ACTIONS_PER_WAYPOINT * 2)
    for d in range(dims):
        plt.plot(gt[:, d], "--", alpha=0.45, linewidth=0.9)
        plt.plot(pred[:, d], alpha=0.8, linewidth=1.0)
    plt.title(f"RPYVA action: pred (solid) vs gt (dashed) | MSE={mse:.6f} MAE={mae:.6f}")
    plt.xlabel("horizon step")
    plt.ylabel("value")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return {"mse": mse, "mae": mae}


def parse_indices(indices_arg: str | None, start_index: int, num_chunks: int, chunk_stride: int) -> list[int]:
    if indices_arg:
        return [int(x.strip()) for x in indices_arg.split(",") if x.strip()]
    return [start_index + i * chunk_stride for i in range(num_chunks)]


def run_single_inference(
    policy: GrootSimPolicy,
    dataset: DroneDataset,
    idx: int,
    prompt_override: str | None,
    output_dir: Path,
    skip_video_decode: bool,
) -> dict[str, Any]:
    prompt = prompt_override or dataset.get_task(idx)
    obs = build_obs(dataset, idx, prompt)
    gt_action = dataset.get_action(idx)

    t0 = time.perf_counter()
    with torch.inference_mode():
        result, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
    elapsed = time.perf_counter() - t0

    pred_action = result.act[ACTION_KEY]
    if isinstance(pred_action, torch.Tensor):
        pred_action = pred_action.detach().cpu().numpy()
    pred_action = np.asarray(pred_action, dtype=np.float32)
    if pred_action.ndim == 3:
        pred_action = pred_action[0]
    elif pred_action.ndim == 1:
        pred_action = pred_action.reshape(1, -1)

    gt_action_seq = np.repeat(gt_action.reshape(1, -1), pred_action.shape[0], axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)

    frame0 = dataset.get_frame(idx)
    imageio.imwrite(output_dir / "input_frame.png", frame0)
    metrics = save_action_plot(pred_action, gt_action_seq, output_dir / "rpyva_action.png")

    decoded_frames = None if skip_video_decode else decode_video_latents(policy, video_pred)
    if decoded_frames is not None:
        decoded_frames = collapse_tiled_drone_frames(decoded_frames)
        save_rgb_video(decoded_frames, output_dir / "pred_video.mp4", fps=8)
        imageio.imwrite(output_dir / "pred_video_first_frame.png", decoded_frames[0])

    summary = {
        "index": idx,
        "prompt": prompt,
        "inference_seconds": elapsed,
        "pred_shape": list(pred_action.shape),
        "gt_shape": list(gt_action_seq.shape),
        "metrics": metrics,
        "pred_first_step_first_8": pred_action[0, :8].tolist(),
        "gt_first_step_first_8": gt_action_seq[0, :8].tolist(),
        "decoded_video_frames": 0 if decoded_frames is None else int(decoded_frames.shape[0]),
        "decoded_video_resolution": None if decoded_frames is None else [int(decoded_frames.shape[2]), int(decoded_frames.shape[1])],
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "summary": summary,
        "decoded_frames": decoded_frames,
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index", type=int, default=0, help="Global row index inside the dataset")
    parser.add_argument("--indices", default=None, help="Comma-separated global row indices to evaluate. Overrides --index/--num_chunks/--chunk_stride.")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunked samples to evaluate starting from --index.")
    parser.add_argument("--chunk_stride", type=int, default=24, help="Stride between chunk start indices when generating multiple chunks.")
    parser.add_argument("--output_dir", default="outputs/inference/drone_open_loop")
    parser.add_argument("--prompt", default=None, help="Override dataset task text")
    parser.add_argument("--skip_video_decode", action="store_true", help="Skip decoding/saving generated video for faster action-only evaluation.")
    parser.add_argument("--concat_video", action="store_true", help="Concatenate decoded videos from all chunks into one longer MP4 when video decoding is enabled.")
    args = parser.parse_args()

    dataset = DroneDataset(args.dataset_path)
    indices = parse_indices(args.indices, args.index, args.num_chunks, args.chunk_stride)

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.DRONE,
        model_path=args.model_path,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_frames = []
    chunk_summaries = []
    for chunk_id, idx in enumerate(indices):
        chunk_output_dir = output_dir / f"chunk_{chunk_id:02d}_idx_{idx}"
        result = run_single_inference(
            policy=policy,
            dataset=dataset,
            idx=idx,
            prompt_override=args.prompt,
            output_dir=chunk_output_dir,
            skip_video_decode=args.skip_video_decode,
        )
        chunk_summary = result["summary"]
        chunk_summary["chunk_id"] = chunk_id
        chunk_summaries.append(chunk_summary)
        if result["decoded_frames"] is not None:
            combined_frames.append(result["decoded_frames"])

    aggregate = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "device": args.device,
        "indices": indices,
        "num_chunks": len(indices),
        "avg_inference_seconds": float(np.mean([item["inference_seconds"] for item in chunk_summaries])),
        "avg_mse": float(np.mean([item["metrics"]["mse"] for item in chunk_summaries])),
        "avg_mae": float(np.mean([item["metrics"]["mae"] for item in chunk_summaries])),
        "chunks": chunk_summaries,
    }

    if args.concat_video and combined_frames:
        stitched = np.concatenate(combined_frames, axis=0)
        save_rgb_video(stitched, output_dir / "pred_video_concat.mp4", fps=8)
        aggregate["concat_video_frames"] = int(stitched.shape[0])
        aggregate["concat_video_seconds"] = float(stitched.shape[0] / 8.0)
    else:
        aggregate["concat_video_frames"] = 0
        aggregate["concat_video_seconds"] = 0.0

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
