#!/usr/bin/env python3
"""Test client for AR_droid policy server using roboarena interface.

Sends real video frames from debug_image/ directory instead of zero dummy images.

Frame schedule (matching debug_inference.py):
  - Step 0 (initial): send frame [0]             (1 frame, H W 3)
  - Step 1: send frames [0, 7, 15, 23]           (4 frames, 4 H W 3)
  - Step 2: send frames [24, 31, 39, 47]         (4 frames)
  - Step 3: send frames [48, 55, 63, 71]         (4 frames)
  - ...

Expected server configuration:
    - image_resolution: (180, 320)
    - n_external_cameras: 2
    - needs_wrist_camera: True
    - action_space: "joint_position"

Usage:
    # Start server with roboarena interface:
    torchrun --nproc_per_node=8 socket_test_optimized_AR.py --port 8000

    # Run this test:
    python test_client_AR.py --host <server_host> --port 8000

    # Use zero images instead of real video (old behavior):
    python test_client_AR.py --host <server_host> --port 8000 --use-zero-images
"""

import argparse
import datetime
import json
import logging
import os
import time
import uuid

import cv2
import numpy as np

import eval_utils.policy_server as policy_server
from eval_utils.policy_client import WebsocketClientPolicy

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "debug_image")

# roboarena key -> video filename
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}

# Frame schedule constants (matching debug_inference.py)
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24


def _repo_root() -> str:
    return os.path.dirname(__file__)


def _default_output_root() -> str:
    return os.getenv("DREAMZERO_OUTPUT_ROOT", os.path.join(_repo_root(), "outputs"))


def _default_report_base() -> str:
    report_dir = os.path.join(_default_output_root(), "reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(report_dir, f"{timestamp}_test_client_ar_latency")


def load_all_frames(video_path: str) -> np.ndarray:
    """Load all frames from a video file. Returns (N, H, W, 3) uint8 array (RGB)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def load_camera_frames() -> dict[str, np.ndarray]:
    """Load all video frames for each camera from the debug_image/ directory.

    Returns:
        Dict mapping roboarena camera keys to (N, H, W, 3) uint8 arrays.
    """
    camera_frames: dict[str, np.ndarray] = {}
    for cam_key, fname in CAMERA_FILES.items():
        path = os.path.join(VIDEO_DIR, fname)
        camera_frames[cam_key] = load_all_frames(path)
        logging.info(f"Loaded {cam_key}: {camera_frames[cam_key].shape}")
    return camera_frames


def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    """Build the frame index schedule for multi-frame chunks.

    Returns a list of frame-index lists. Each inner list has 4 indices.
    """
    chunks: list[list[int]] = []
    current_frame = 23  # first anchor frame
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            logging.info(
                f"Frame {indices[-1]} >= {total_frames}, stopping at {len(chunks)} chunks"
            )
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def _make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    prompt: str,
    session_id: str,
) -> dict:
    """Build an observation dict from real video frames.

    For 1 frame: each image key is (H, W, 3).
    For 4 frames: each image key is (4, H, W, 3).
    """
    obs: dict = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]  # (T, H, W, 3)
        if len(frame_indices) == 1:
            selected = selected[0]  # (H, W, 3)
        obs[cam_key] = selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def _make_zero_observation(
    server_config: policy_server.PolicyServerConfig,
    prompt: str = "pick up the object",
    session_id: str | None = None,
) -> dict:
    """Create a dummy observation matching AR_droid expectations.
    
    AR_droid expects:
        - 2 external cameras (exterior_image_0_left, exterior_image_1_left)
        - 1 wrist camera (wrist_image_left)
        - Image resolution: 180x320 (H x W)
        - joint_position: 7 DoF
        - gripper_position: 1 DoF
    """
    obs = {}
    
    # Determine image resolution
    if server_config.image_resolution is not None:
        h, w = server_config.image_resolution
    else:
        # Default for AR_droid
        h, w = 180, 320
    
    # External cameras (0-indexed in roboarena)
    for i in range(server_config.n_external_cameras):
        obs[f"observation/exterior_image_{i}_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs[f"observation/exterior_image_{i}_right"] = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Wrist camera
    if server_config.needs_wrist_camera:
        obs["observation/wrist_image_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs["observation/wrist_image_right"] = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Session ID - should be passed in to ensure consistency within a session
    if server_config.needs_session_id:
        import uuid
        # Generate unique session ID if not provided
        obs["session_id"] = session_id if session_id else str(uuid.uuid4())
    
    # State observations (AR_droid: 7 DoF arm + 1 gripper)
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    
    # Language prompt
    obs["prompt"] = prompt
    
    return obs


def _summarize_actions(actions: np.ndarray) -> dict:
    return {
        "shape": list(actions.shape),
        "min": float(actions.min()),
        "max": float(actions.max()),
    }


def _write_latency_report(report_base: str, report: dict) -> tuple[str, str]:
    markdown_path = f"{report_base}.md"
    json_path = f"{report_base}.json"

    lines = [
        "# DreamZero Test Client Latency Report",
        "",
        f"Date: {report['date']}",
        "",
        "## Run Info",
        "",
        f"- host: `{report['host']}`",
        f"- port: `{report['port']}`",
        f"- mode: `{report['mode']}`",
        f"- session_id: `{report['session_id']}`",
        f"- prompt: `{report['prompt']}`",
        f"- total_frames_available: `{report['total_frames_available']}`",
        f"- reset_status: `{report['reset_status']}`",
        "",
        "## Latency",
        "",
        "| Step | Frames | Time | Action Shape | Action Range |",
        "| --- | --- | ---: | --- | --- |",
    ]

    for item in report["steps"]:
        lines.append(
            "| "
            f"{item['name']} | "
            f"`{item['frame_indices']}` | "
            f"{item['latency_seconds']:.2f}s | "
            f"`{tuple(item['action']['shape'])}` | "
            f"[{item['action']['min']:.4f}, {item['action']['max']:.4f}] |"
        )

    summary = report["summary"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- total_infer_calls: `{summary['total_infer_calls']}`",
            f"- total_infer_time: `{summary['total_infer_time_seconds']:.2f}s`",
            f"- avg_infer_time: `{summary['avg_infer_time_seconds']:.2f}s`",
            f"- hottest_step: `{summary['slowest_step_name']}` = `{summary['slowest_step_seconds']:.2f}s`",
        ]
    )

    if summary["avg_hot_chunk_time_seconds"] is not None:
        lines.append(
            f"- avg_hot_chunk_time: `{summary['avg_hot_chunk_time_seconds']:.2f}s`"
        )

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return markdown_path, json_path


def test_ar_droid_policy_server(
    host: str = "localhost",
    port: int = 8000,
    num_chunks: int = 15,
    prompt: str = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
    use_zero_images: bool = False,
    report_base: str | None = None,
):
    """Test the AR_droid policy server with roboarena interface.

    When use_zero_images is False (default), loads real video frames from
    debug_image/ and follows the frame schedule from debug_inference.py.
    """
    logging.info(f"Connecting to AR_droid server at {host}:{port}...")
    
    client = WebsocketClientPolicy(host=host, port=port)
    
    # Validate server metadata
    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    assert isinstance(metadata, dict), "Metadata should be a dict"
    
    try:
        server_config = policy_server.PolicyServerConfig(**metadata)
    except Exception as e:
        logging.error(f"Error parsing metadata: {e}")
        raise e
    
    # Validate expected AR_droid configuration
    logging.info(f"Server config: {server_config}")
    assert server_config.n_external_cameras == 2, f"Expected 2 external cameras, got {server_config.n_external_cameras}"
    assert server_config.needs_wrist_camera, "Expected wrist camera to be enabled"
    assert server_config.action_space == "joint_position", f"Expected joint_position action space, got {server_config.action_space}"
    
    logging.info("Server configuration validated for AR_droid")
    
    # Generate unique session ID for this test run
    import uuid
    session_id = str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}")

    report_steps: list[dict] = []
    total_frames_available: int | None = None
    reset_status = "not_sent"

    # ── Zero-image fallback mode ──────────────────────────────────────
    if use_zero_images:
        logging.info("Using ZERO dummy images (legacy mode)")
        for i in range(num_chunks):
            obs = _make_zero_observation(server_config, prompt=prompt, session_id=session_id)
            logging.info(f"Inference {i + 1}/{num_chunks}: prompt='{prompt}'")
            t0 = time.time()
            actions = client.infer(obs)
            dt = time.time() - t0
            _log_action(actions, dt)
            report_steps.append(
                {
                    "name": f"Chunk {i}",
                    "frame_indices": [],
                    "latency_seconds": dt,
                    "action": _summarize_actions(actions),
                }
            )

        logging.info("Sending reset...")
        client.reset({})
        reset_status = "success"
        logging.info("Done (zero-image mode).")
        return _finalize_report(
            report_base=report_base,
            host=host,
            port=port,
            prompt=prompt,
            mode="zero-image",
            session_id=session_id,
            total_frames_available=total_frames_available,
            reset_status=reset_status,
            report_steps=report_steps,
        )

    # ── Real video frame mode ─────────────────────────────────────────
    logging.info("Loading real video frames from debug_image/ directory")
    camera_frames = load_camera_frames()

    total_frames = min(v.shape[0] for v in camera_frames.values())
    total_frames_available = total_frames
    logging.info(f"Total frames available: {total_frames}")

    # Build frame schedule
    chunks = build_frame_schedule(total_frames, num_chunks)

    logging.info("Frame schedule:")
    logging.info("  Initial: [0]")
    for i, indices in enumerate(chunks):
        logging.info(f"  Chunk {i}: {indices}")

    # Step 0: initial single frame
    logging.info("=== Initial: frame [0] ===")
    obs = _make_obs_from_video(camera_frames, [0], prompt, session_id)
    t0 = time.time()
    actions = client.infer(obs)
    dt = time.time() - t0
    _log_action(actions, dt)
    report_steps.append(
        {
            "name": "Initial",
            "frame_indices": [0],
            "latency_seconds": dt,
            "action": _summarize_actions(actions),
        }
    )

    # Subsequent chunks: send 4 frames at a time
    for chunk_idx, frame_indices in enumerate(chunks):
        logging.info(f"=== Chunk {chunk_idx}: frames {frame_indices} ===")
        obs = _make_obs_from_video(camera_frames, frame_indices, prompt, session_id)
        t0 = time.time()
        actions = client.infer(obs)
        dt = time.time() - t0
        _log_action(actions, dt)
        report_steps.append(
            {
                "name": f"Chunk {chunk_idx}",
                "frame_indices": frame_indices,
                "latency_seconds": dt,
                "action": _summarize_actions(actions),
            }
        )

    # Reset triggers video save on the server
    logging.info("Sending reset to save video...")
    client.reset({})
    reset_status = "success"

    logging.info("Done.")
    return _finalize_report(
        report_base=report_base,
        host=host,
        port=port,
        prompt=prompt,
        mode="real-video",
        session_id=session_id,
        total_frames_available=total_frames_available,
        reset_status=reset_status,
        report_steps=report_steps,
    )


def _finalize_report(
    report_base: str | None,
    host: str,
    port: int,
    prompt: str,
    mode: str,
    session_id: str,
    total_frames_available: int | None,
    reset_status: str,
    report_steps: list[dict],
) -> tuple[str, str] | None:
    if not report_base:
        return None

    total_infer_time = sum(step["latency_seconds"] for step in report_steps)
    avg_infer_time = total_infer_time / len(report_steps) if report_steps else 0.0
    slowest_step = (
        max(report_steps, key=lambda step: step["latency_seconds"])
        if report_steps
        else {"name": "N/A", "latency_seconds": 0.0}
    )
    hot_chunks = [
        step["latency_seconds"]
        for step in report_steps
        if step["name"].startswith("Chunk ") and step["name"] != "Chunk 0"
    ]
    avg_hot_chunk_time = (
        sum(hot_chunks) / len(hot_chunks) if hot_chunks else None
    )

    report = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host": host,
        "port": port,
        "mode": mode,
        "session_id": session_id,
        "prompt": prompt,
        "total_frames_available": total_frames_available,
        "reset_status": reset_status,
        "steps": report_steps,
        "summary": {
            "total_infer_calls": len(report_steps),
            "total_infer_time_seconds": total_infer_time,
            "avg_infer_time_seconds": avg_infer_time,
            "slowest_step_name": slowest_step["name"],
            "slowest_step_seconds": slowest_step["latency_seconds"],
            "avg_hot_chunk_time_seconds": avg_hot_chunk_time,
        },
    }
    markdown_path, json_path = _write_latency_report(report_base, report)
    logging.info("Latency report written to %s", markdown_path)
    logging.info("Latency report JSON written to %s", json_path)
    return markdown_path, json_path


def _log_action(actions: np.ndarray, dt: float) -> None:
    """Pretty-print action shape, range, and timing."""
    assert isinstance(actions, np.ndarray), f"Expected numpy array, got {type(actions)}"
    assert actions.ndim == 2, f"Expected 2D array, got shape {actions.shape}"
    assert actions.shape[-1] == 8, (
        f"Expected 8 action dims (7 joints + 1 gripper), got {actions.shape[-1]}"
    )
    logging.info(
        f"  Action shape: {actions.shape}, "
        f"range: [{actions.min():.4f}, {actions.max():.4f}], "
        f"time: {dt:.2f}s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test AR_droid policy server with real video frames from debug_image/"
    )
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=15,
        help="Number of 4-frame chunks to send after the initial frame (default: 15)",
    )
    parser.add_argument(
        "--prompt",
        default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
        help="Language prompt for the policy",
    )
    parser.add_argument(
        "--use-zero-images",
        action="store_true",
        help="Use zero dummy images instead of real video frames (legacy mode)",
    )
    parser.add_argument(
        "--report-base",
        default=_default_report_base(),
        help=(
            "Base path for latency report outputs without extension. "
            "Defaults to outputs/reports/<timestamp>_test_client_ar_latency"
        ),
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable writing latency reports to outputs/reports",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    test_ar_droid_policy_server(
        host=args.host,
        port=args.port,
        num_chunks=args.num_chunks,
        prompt=args.prompt,
        use_zero_images=args.use_zero_images,
        report_base=None if args.no_report else args.report_base,
    )


if __name__ == "__main__":
    main()
