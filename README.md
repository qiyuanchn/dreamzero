# DreamZero Fork: 1.3B Training, Drone Embodiment, and Inference Workflows

Fork of NVIDIA DreamZero with a practical local workflow centered on:

- Wan2.1 1.3B DreamZero training
- DROID LoRA and full fine-tuning
- Drone embodiment conversion, training, and offline evaluation
- Distributed inference service validation on single-GPU and dual-GPU setups

This README describes the current state of this repository as it exists now, including the fork-specific pipelines and the main results already verified locally.

## Upstream DreamZero

DreamZero is a World Action Model that jointly predicts actions and videos for zero-shot robot policies.

- Upstream project page: <https://dreamzero0.github.io/>
- Upstream paper: <https://arxiv.org/abs/2602.15922>
- Upstream repository lineage: NVIDIA GEAR DreamZero

This fork keeps the upstream codebase but adds a more operational workflow for local training and deployment.

## What This Fork Adds

The main fork-specific capabilities are:

- DreamZero 1.3B training path based on `Wan2.1-Fun-V1.1-1.3B-InP`
- Stable DROID training scripts for LoRA and full fine-tuning
- Unified output layout under `outputs/train`, `outputs/inference`, `outputs/logs`, and `outputs/reports`
- TensorBoard-first monitoring and `loss_log.jsonl` backfill support
- Inference service scripts for start/stop/TensorRT build workflows
- Drone embodiment support:
  - converter from raw tar/source trajectories to LeRobot-style data
  - single-view drone data config
  - offline open-loop evaluation
  - training entrypoint for large drone corpora
- Large-scale sharded dataset memory fixes for multi-million-episode drone training

## Current Status

### Verified Training

- DROID LoRA training is the most mature local training path
- DROID full fine-tune startup and checkpoint emission are verified
- Drone sharded training loader has been refactored to avoid host RAM blowups on very large datasets

### Verified Inference

- Single-GPU inference works
- Dual-GPU inference works
- Multi-chunk real-video client runs work
- `reset`-triggered video saving works
- New LoRA checkpoints have been validated through the inference path

### Current Limitations

- 4-GPU inference is not considered stable in this fork
- The primary validated inference path is single-GPU or dual-GPU
- Full fine-tune still needs longer-duration local runs before making strong quality claims

## Key Achievements in This Fork

- Converted the drone embodiment to a clean `12D observation + 72D flattened RPYVA action` format
- Updated the model/data path to treat drone as a true single-view embodiment instead of duplicating one image into a tiled canvas
- Added drone offline evaluation over converted LeRobot episodes with optional multi-chunk video stitching
- Reworked sharded dataset loading so million-episode corpora do not pre-expand `all_steps`, `step_filter`, or Python-heavy shard schedules into hundreds of GB of RAM
- Validated that the new sharded drone dataset initialization path can start with about `1.1 GB` RSS on the real large dataset instead of exploding host memory

## Repository Layout

Important entrypoints:

- Training:
  - `scripts/train/droid_training_lora.sh`
  - `scripts/train/droid_training_full_finetune.sh`
  - `scripts/train/drone_training.sh`
- Data conversion:
  - `scripts/data/convert_droid.py`
  - `scripts/data/convert_drone_tar_to_lerobot.py`
- Inference:
  - `scripts/inference/start_dreamzero_service.sh`
  - `scripts/inference/stop_dreamzero_service.sh`
  - `scripts/inference/build_trt_engine.sh`
  - `test_client_AR.py`
- Offline evaluation:
  - `scripts/open_loop_drone.py`
- Docs:
  - `docs/TRAINING_GUIDE.md`
  - `docs/INFERENCE_SERVICE_GUIDE.md`
  - `docs/DATASET_TO_GEAR_AND_TRAIN.md`
  - `docs/DROID_CONVERSION.md`

## Environment Setup

### Python

- Recommended: Python `3.11`

### Conda

```bash
conda create -n dreamzero python=3.11
conda activate dreamzero
```

### Install

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
MAX_JOBS=8 pip install --no-build-isolation flash-attn
```

Optional components:

- Transformer Engine: useful on GB200 workflows
- TensorRT: only needed for the TensorRT-backed inference route

### Local Shell Setup

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
export PYTHONPATH=/home/zqy/ws/dreamzero
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
```

## Checkpoints

### Wan 1.3B Base Model

This fork primarily trains from:

- `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP`
- tokenizer: `google/umt5-xxl`

Download them with:

```bash
pip install "huggingface_hub[cli]"

hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP \
  --local-dir ./checkpoints/Wan2.1-Fun-V1.1-1.3B-InP

hf download google/umt5-xxl \
  --local-dir ./checkpoints/umt5-xxl
```

### Upstream DreamZero Checkpoints

Upstream released checkpoints remain useful references:

- DreamZero-DROID: <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>
- DreamZero-AgiBot: <https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot>

## Output Layout

This fork standardizes outputs under:

- `outputs/train`
- `outputs/inference`
- `outputs/logs`
- `outputs/reports`
- `outputs/tensorboard`

Training scripts maintain convenient symlinks such as:

- `outputs/train/latest`
- `outputs/train/latest_droid_lora`
- `outputs/train/latest_droid_full`

## DROID Workflow

### Dataset

Recommended main training dataset:

- Hugging Face preprocessed DreamZero DROID data
- dataset repo: `GEAR-Dreams/DreamZero-DROID-Data`

Download:

```bash
huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data \
  --repo-type dataset \
  --local-dir ./data/droid_lerobot
```

### LoRA Training

```bash
export DROID_DATA_ROOT=./data/droid_lerobot
export WAN_CKPT_DIR=./checkpoints/Wan2.1-Fun-V1.1-1.3B-InP
export TOKENIZER_DIR=./checkpoints/umt5-xxl
export NUM_GPUS=4

bash scripts/train/droid_training_lora.sh
```

### Full Fine-Tuning

```bash
export DROID_DATA_ROOT=./data/droid_lerobot
export WAN_CKPT_DIR=./checkpoints/Wan2.1-Fun-V1.1-1.3B-InP
export TOKENIZER_DIR=./checkpoints/umt5-xxl
export NUM_GPUS=4

bash scripts/train/droid_training_full_finetune.sh
```

### Local DROID Training Notes

What has already been verified locally:

- LoRA startup is stable
- Hydra logging directory overrides are fixed
- checkpoints are emitted correctly
- TensorBoard logging is enabled by default
- latest local LoRA checkpoints have been exercised through inference

## Drone Workflow

### Drone Data Format

The current drone embodiment in this fork uses:

- video: `video.rgb`
- state: `observation.state.obs` with `12` dimensions
- action: flattened `rpyva` with `72` dimensions
- language: `annotation.task`

### Drone Conversion

Use the converter to build a LeRobot-style dataset from raw drone trajectories:

```bash
python scripts/data/convert_drone_tar_to_lerobot.py \
  --input <raw-drone-root-or-tar> \
  --output <converted-output-dir>
```

Important converter properties in this fork:

- emits chunked LeRobot layout
- supports large parallel conversion
- supports `source-link` video mode to reuse original `rgb.mp4`
- writes richer conversion metadata to `conversion_summary.json`

### Drone Training

Main entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
DRONE_DATA_ROOT=/path/to/drone_lerobot \
OUTPUT_DIR=/path/to/output \
bash scripts/train/drone_training.sh
```

Useful knobs:

- `NUM_STEPS_PER_SHARD`
- `NUM_WORKERS`
- `MAX_STEPS`
- `SAVE_STEPS`
- `LOGGING_STEPS`

### Large Drone Dataset Memory Fixes

This fork includes a substantial sharded loader rewrite for very large drone corpora.

Fixed high-memory behaviors:

- no precomputed full `all_steps` for sharded datasets
- lazy default `step_filter`
- streamed `episodes.jsonl` parsing
- no eager `all_video_paths` / `all_parquet_paths` for the drone sharded loader
- shard bookkeeping rewritten to compact numpy/range-based structures
- shard schedule no longer stored as huge Python tuple lists

Practical result already measured on the real large drone dataset:

- `num_steps_per_shard=256`
- dataset init peak RSS about `1.1 GB`
- shard count about `1,047,444`

For very large runs, increasing `NUM_STEPS_PER_SHARD` to `1024` or `2048` is the first tuning lever if host RAM is still under pressure.

### Drone Offline Evaluation

```bash
python scripts/open_loop_drone.py \
  --model_path <checkpoint-or-run-dir> \
  --dataset_path <converted-drone-dataset> \
  --device cuda:0 \
  --index 0 \
  --output_dir outputs/inference/drone_open_loop
```

The current evaluator supports:

- action metric plotting
- multi-chunk evaluation
- optional concatenated decoded video
- old tiled-drone-frame visualization fallback

## Inference Service

### Start the Service

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PORT=6000 \
MODEL_PATH=<checkpoint-or-model-dir> \
bash scripts/inference/start_dreamzero_service.sh
```

### Test Client

```bash
python test_client_AR.py --host 127.0.0.1 --port 6000 --num-chunks 3
```

### What Is Verified

- single-GPU server path
- dual-GPU server path
- multi-chunk client path
- real video client path
- reset-triggered MP4 saving

### Current Inference Caveats

- 4-GPU inference is not recommended
- the service start script reporting success does not guarantee the socket is ready yet
- a LoRA `checkpoint-*` cannot be used directly unless `experiment_cfg/conf.yaml` is present

If needed, add a symlink inside a LoRA checkpoint:

```bash
cd <run_dir>/checkpoint-2000
ln -sfn ../experiment_cfg experiment_cfg
```

## Current Recommended Workflows

### Best-Validated Training Path

1. Start with DROID LoRA
2. Monitor with TensorBoard
3. Validate checkpoints through dual-GPU inference
4. Move to full fine-tuning only after LoRA is stable

### Best-Validated Inference Path

1. Dual GPU: `CUDA_VISIBLE_DEVICES=0,1`
2. Use `test_client_AR.py` for smoke tests and real video multi-chunk runs
3. Check port readiness with `ss -ltnp`
4. Tail the generated service log if startup looks suspicious

## Local Results Snapshot

This fork has already produced local verification runs, including:

- DROID LoRA checkpoints through at least `7000` steps
- best observed local LoRA loss around `0.0922`
- full fine-tune smoke tests with successful save behavior
- dual-GPU real-video inference validation on trained LoRA checkpoints

These are engineering validation milestones, not official benchmark claims.

## Additional Documentation

- [Training guide](docs/TRAINING_GUIDE.md)
- [Inference service guide](docs/INFERENCE_SERVICE_GUIDE.md)
- [Adding a new embodiment](docs/DATASET_TO_GEAR_AND_TRAIN.md)
- [DROID conversion notes](docs/DROID_CONVERSION.md)
- [Wan2.2 backbone notes](docs/WAN22_BACKBONE.md)

## Known Good Practical Notes

- `TOKENIZERS_PARALLELISM=false` is a good default
- `NUM_WORKERS=0` remains the safest starting point when debugging data issues
- LoRA checkpoints often need the `experiment_cfg` symlink for inference entrypoints
- dual-GPU inference is the sweet spot for this fork right now

## License

This repository inherits the upstream DreamZero licensing and repository structure. See [LICENSE](LICENSE).
