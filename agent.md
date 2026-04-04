# Agent Notes

## 使用约定

- 每次开始新的对话或任务前，先阅读本文件，再继续交流或执行操作。
- 每次完成会影响项目状态的工作后，都要把关键信息同步更新到本文件。
- 本文件记录项目当前的“操作性事实”，优先写环境、路径、实验结论、限制条件、已知坑和可复用命令。

## 当前环境

- 仓库路径: `/home/zqy/ws/dreamzero`
- 当前开发分支: `dev`
- 常用环境激活:

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
export PYTHONPATH=/home/zqy/ws/dreamzero
```

- 当前主要工作路线:
  - 训练: Wan2.1 1.3B DreamZero-DROID
  - 推理: 双卡 `0,1` 为主
- 推理当前只验证了单卡和双卡路径，不建议继续尝试 4 卡推理。

## 输出与目录约定

- 所有统一输出根目录: `/home/zqy/ws/dreamzero/outputs`
- 当前实际目录约定:
  - 训练: `/home/zqy/ws/dreamzero/outputs/train`
  - 推理: `/home/zqy/ws/dreamzero/outputs/inference`
  - 日志: `/home/zqy/ws/dreamzero/outputs/logs`
  - 报告: `/home/zqy/ws/dreamzero/outputs/reports`
  - TensorBoard 补录目录: `/home/zqy/ws/dreamzero/outputs/tensorboard`

- 训练脚本会自动维护这些软链接:
  - `/home/zqy/ws/dreamzero/outputs/train/latest`
  - `/home/zqy/ws/dreamzero/outputs/train/latest_droid_lora`
  - `/home/zqy/ws/dreamzero/outputs/train/latest_droid_full`

## 当前训练状态

- DreamZero 1.3B 训练脚本已经整理完成:
  - LoRA: `/home/zqy/ws/dreamzero/scripts/train/droid_training_lora.sh`
  - Full: `/home/zqy/ws/dreamzero/scripts/train/droid_training_full_finetune.sh`
- 两个脚本都已修复 Hydra 覆盖问题:
  - 必须使用 `+training_args.logging_dir=...`
  - 不能写成 `training_args.logging_dir=...`
- 两个脚本都默认:
  - 自动识别 Hugging Face cache/snapshot 数据目录
  - 输出到 `outputs/train/<日期>/HH-MM-SS-droid-{lora|full}`
  - `report_to=tensorboard`

### 数据集

- 当前主训练数据:
  - `/data2/zqy/datasets--GEAR-Dreams--DreamZero-DROID-Data`
- 这是 DreamZero 官方发布的预处理 DROID 数据，不是原始 RLDS 本体。
- 小数据集路径:
  - `/home/zqy/ws/dreamzero/data/droid_lerobot`
- 小数据集不能直接替代正式训练集，因为缺少完整 `meta` 元数据。

### 当前 1.3B 训练策略

- 当前最合理的初始化方案不是拿 14B DreamZero checkpoint 硬加载到 1.3B。
- 当前主线是:
  - 基座: `Wan2.1-Fun-V1.1-1.3B-InP`
  - tokenizer: `umt5-xxl`
  - DreamZero 新增模块随机初始化
  - 先 LoRA，再 full fine-tune

### 当前训练脚本关键配置

- LoRA:
  - `train_architecture=lora`
  - `save_lora_only=true`
  - 默认 `MAX_STEPS=10000`
  - 默认 `SAVE_STEPS=1000`
  - 默认 `NUM_GPUS=4`
- Full:
  - `train_architecture=full`
  - `save_lora_only=false`
  - 默认 `MAX_STEPS=100000`
  - 默认 `SAVE_STEPS=2000`
  - 默认 `training_args.deepspeed=zero2_offload`

### 最近已验证的训练产物

- 最新可用 LoRA 训练 run:
  - `/home/zqy/ws/dreamzero/outputs/train/2026-03-26/23-21-38-droid-lora`
- 最新可用于推理的 LoRA checkpoint:
  - `/home/zqy/ws/dreamzero/outputs/train/2026-03-26/23-21-38-droid-lora/checkpoint-2000`
- 最新 full run:
  - `/home/zqy/ws/dreamzero/outputs/train/2026-03-26/21-39-35-droid-full`
- 注意:
  - 当时 full run 目录里没有可直接拿来推理的 `checkpoint-*`
  - 当前实际可用的“最新训练权重”是 LoRA `checkpoint-2000`

### 训练监控

- 每个 run 会写:
  - `loss_log.jsonl`
  - `tensorboard/`
- 常看指标:
  - `loss`
  - `dynamics_loss_avg`
  - `action_loss_avg`
  - `learning_rate`
- 当前代码里 LoRA 和 full 的 loss 形式基本相同，区别主要在可训练参数范围，不在 loss 定义。

## 当前推理状态

- 当前 1.3B 推理路线:
  - backbone: `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP`
  - tokenizer: `umt5-xxl`
  - server: `/home/zqy/ws/dreamzero/socket_test_optimized_AR.py`
  - client: `/home/zqy/ws/dreamzero/test_client_AR.py`

- 已确认跑通:
  - 单卡推理
  - 双卡推理
  - 真实视频多 chunk 推理
  - `reset` 后视频保存
  - 新训练 LoRA checkpoint 推理
  - `test_client_AR.py` 自动写客户端耗时报告到 `outputs/reports`

- 仍不建议 4 卡推理:
  - 代码主实现没有稳定支持
  - `parallelize()` 里只稳定支持 `ip_size=1` 或 `2`

## 推理输入流程结论

- DreamZero 推理不是“只输入第 0 帧和 prompt，然后后续全靠模型自由生成”。
- 实际流程是:
  - client 反复发送观测
  - 每次观测包含 `prompt + 图像 + robot state`
  - 模型每次输出一段动作 `(24, 8)`，同时返回 `video_pred`
  - 服务端把多次 `video_pred` 累积起来
  - 在 `reset` 或连接关闭时用 VAE 解码并保存 mp4

- `test_client_AR.py` 当前真实视频验证用的是:
  - prompt:
    - `Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan`
  - 输入视频:
    - `/home/zqy/ws/dreamzero/debug_image/exterior_image_1_left.mp4`
    - `/home/zqy/ws/dreamzero/debug_image/exterior_image_2_left.mp4`
    - `/home/zqy/ws/dreamzero/debug_image/wrist_image_left.mp4`

- 当前默认 chunk schedule:
  - Initial: `[0]`
  - Chunk 0: `[0, 7, 15, 23]`
  - Chunk 1: `[24, 31, 39, 47]`
  - Chunk 2: `[48, 55, 63, 71]`

## 推理已知坑

### 1. LoRA checkpoint 不能直接裸当 `model_path`

- 推理入口要求 `model_path/experiment_cfg/conf.yaml` 存在。
- 训练保存出来的 LoRA `checkpoint-*` 默认只有:
  - `config.json`
  - `model.safetensors`
- 因此如果直接把 `checkpoint-2000` 当 `--model-path`，会报:
  - `FileNotFoundError: ... checkpoint-2000/experiment_cfg/conf.yaml`

- 解决办法:
  - 在 checkpoint 目录里补一个软链接:

```bash
cd /home/zqy/ws/dreamzero/outputs/train/2026-03-26/23-21-38-droid-lora/checkpoint-2000
ln -sfn ../experiment_cfg experiment_cfg
```

### 2. `start_dreamzero_service.sh` 的“启动成功”不代表服务一定已准备好

- 之前遇到过:
  - `screen` 会话创建了
  - 但实际推理窗口没有正常存活或没有真正 ready
- 所以启动后要额外检查:
  - `ss -ltnp | rg ':<port>'`
  - `tail -f <run_dir>/dreamzero_service_port_<port>.log`

### 3. 后台服务如果没有收到请求，不会自己产出动作结果文件

- 它是 websocket 服务。
- 动作结果直接返回给 client，不会自动保存成 `json/npy`。
- 如果打开 `DREAMZERO_SAVE_RESET_VIDEO=1`，则在 `reset` 后会保存视频。

### 4. 后台服务可能在一次 client 连接结束后退出

- 之前出现过:
  - 服务启动正常
  - 跑完一次 client 后，后续再次连接失败
- 因此如果需要稳定复用服务，先检查:
  - 端口是否还在监听
  - 对应 `socket_test_optimized_AR.py` 进程是否还在

## 最近已确认的推理实验

### 旧的基准实验

- 单卡:
  - GPU: `6`
  - 端口: `6011`
  - 实验目录: `/home/zqy/ws/dreamzero/outputs/experiments/20260326_single_gpu6_multichunk_mem`
- 双卡:
  - GPU: `3,6`
  - 端口: `6012`
  - 实验目录: `/home/zqy/ws/dreamzero/outputs/experiments/20260326_dual_gpu3_6_multichunk_mem`
- 详细总结:
  - `/home/zqy/ws/dreamzero/outputs/reports/20260326_multichunk_inference_report.md`

### 新模型 LoRA checkpoint 推理

- 使用权重:
  - `/home/zqy/ws/dreamzero/outputs/train/2026-03-26/23-21-38-droid-lora/checkpoint-2000`
- 验证方式:
  - 双卡 `0,1`
  - 真实视频 client

## 2026-04-04 本地验证补充

- 这次改动的快速验证统一限制在后四张卡:
  - 仅使用 `GPU 7`
  - 后续如果继续做 smoke test，也优先只用 `4,5,6,7`
- 当前本机可直接用于推理加载、且带 `experiment_cfg/conf.yaml` 的 LoRA checkpoint:
  - `/home/zqy/ws/dreamzero/outputs/train/2026-04-04/10-11-15-droid-lora-continue/checkpoint-2000`
- 当前仓库里默认脚本写的 `MODEL_PATH=/home/zqy/ws/dreamzero/checkpoints/DreamZero-DROID` 在本机不存在。
  - 如果直接用 `scripts/inference/start_dreamzero_service.sh`，建议显式传入 `MODEL_PATH`

### 这次已做的验证

- 语法检查通过:
  - `python -m py_compile` 已覆盖这次改动涉及的核心 Python 文件
  - `bash -n` 已覆盖 `start_dreamzero_service.sh`、`build_trt_engine.sh`、`stop_dreamzero_service.sh`
- `test_client_AR.py` 新增报告逻辑已做本地函数级验证:
  - 能成功生成 `.md` 和 `.json`
  - summary 里的 `total_infer_calls`、`avg_hot_chunk_time_seconds` 正常
- `_build_dit_step_mask()` 已做快速回归:
  - 兼容原来的 `(16, 5/6/7/8)`
  - 对新的非 16 step 组合例如 `(10, 3)` 也能返回合法 mask
- `create_wan_test_inputs(..., model_type=\"ar_1.3B_droid\")` 已验证 shape 正常:
  - `kv_cache_packed` shape 为 `(30, 2, 1, 7920, 12, 128)`

### 这次未拿到完整通过证据的项

- 在 `GPU 7` 上前台启动:
  - `python -m torch.distributed.run --standalone --nproc_per_node=1 socket_test_optimized_AR.py --port 6104 --model-path /home/zqy/ws/dreamzero/outputs/train/2026-04-04/10-11-15-droid-lora-continue/checkpoint-2000`
- 已确认:
  - 分布式初始化正常
  - LoRA 权重加载正常
  - 模型构建持续进行，没有立刻报错退出
- 但在本轮观察窗口内还没有等到端口真正进入监听态，因此:
  - 当前可以认为“启动加载阶段未见明显异常”
  - 还不能把它记成“完整 ready + client 推理已通过”
- 已确认结果:
  - Initial: `Action shape (24, 8)`
  - Chunk 0: `Action shape (24, 8)`

### 新模型 `chunk3` 验证

- 临时运行目录:
  - `/home/zqy/ws/dreamzero/outputs/inference/2026-03-27/chunk3_run_13-20-27`
- 服务保存视频目录:
  - `/home/zqy/ws/dreamzero/outputs/inference/2026-03-27/13-22-28-checkpoint-2000`
- 结果:
  - Initial `[0]`: `14.70s`
  - Chunk 0 `[0, 7, 15, 23]`: `21.77s`
  - Chunk 1 `[24, 31, 39, 47]`: `1.19s`
  - Chunk 2 `[48, 55, 63, 71]`: `0.92s`
  - 所有阶段都成功返回 `Action shape: (24, 8)`
- 视频产物:
  - `/home/zqy/ws/dreamzero/outputs/inference/2026-03-27/13-22-28-checkpoint-2000/000000_03_27_13_23_10_n4.mp4`

### 2026-04-02 `GPU 6,7` 双卡 checkpoint-2000 验证

- GPU:
  - `6,7`
- 端口:
  - `6007`
- 模型:
  - `/home/zqy/ws/dreamzero/outputs/train/2026-03-26/23-21-38-droid-lora/checkpoint-2000`
- 关键环境:
  - `--enable-dit-cache`
  - `DREAMZERO_SAVE_RESET_VIDEO=0`
- 服务输出目录:
  - `/home/zqy/ws/dreamzero/outputs/inference/2026-04-02/11-36-30-checkpoint-2000`
- 客户端耗时报告:
  - `/home/zqy/ws/dreamzero/outputs/reports/20260402_gpu6_7_checkpoint2000_latency.md`
  - `/home/zqy/ws/dreamzero/outputs/reports/20260402_gpu6_7_checkpoint2000_latency.json`
- 客户端结果:
  - Initial `[0]`: `6.31s`
  - Chunk 0 `[0, 7, 15, 23]`: `17.67s`
  - Chunk 1 `[24, 31, 39, 47]`: `0.91s`
  - Chunk 2 `[48, 55, 63, 71]`: `0.97s`
  - hot chunk 平均: `0.94s`
  - `reset`: `success`
- 服务端关键拆分:
  - Initial:
    - Total `6.04s`
    - Text `0.24s`
    - Image `0.94s`
    - KV cache `0.07s`
    - Diffusion `0.61s`
    - Scheduler `4.19s`
  - Chunk 0:
    - Total `17.63s`
    - Text `0.05s`
    - VAE `0.14s`
    - KV cache `0.07s`
    - Diffusion `0.56s`
    - Scheduler `16.81s`
  - Chunk 1:
    - Total `0.87s`
    - Text `0.05s`
    - VAE `0.14s`
    - KV cache `0.07s`
    - Diffusion `0.60s`
    - Scheduler `0.00s`
  - Chunk 2:
    - Total `0.93s`
    - Text `0.05s`
    - VAE `0.14s`
    - KV cache `0.08s`
    - Diffusion `0.66s`
    - Scheduler `0.00s`
- 结论:
  - `6,7` 双卡这轮明显符合“冷态慢、热态快”的预期
  - 相比 2026-03-27 的 `0,1` 双卡结果，这轮 `Initial` 和 `Chunk 0` 也更快

## 当前推理提速建议

- 必开:
  - `--enable-dit-cache`
- 如果当前目标是测动作返回延迟，而不是留视频:
  - 保持 `DREAMZERO_SAVE_RESET_VIDEO=0`
- 如果当前目标不是排查输入内容:
  - 优先避免额外的输入图片落盘，因为长跑时会带来不必要 I/O
- 做性能对比时必须把冷态和热态分开:
  - `Initial` 和 `Chunk 0` 主要受 warmup / scheduler 影响
  - 热态 chunk 才更接近可复用吞吐
- 快速验证建议顺序:
  - 先 `--use-zero-images --num-chunks 1`
  - 再真实视频 `--num-chunks 3`
- TensorRT 相关:
  - 仅设置 `ENABLE_TENSORRT=true` 不等于已经加载 TRT engine
  - 真正走 TRT 还需要 `LOAD_TRT_ENGINE=<...>.trt`
  - 当前启动脚本已支持自动发现:
    - `MODEL_PATH/tensorrt/wan/WanModel_<precision>.trt`
  - 当前环境确认:
    - `dreamzero` conda 环境里有 `tensorrt`、`modelopt`、`onnx`
    - 当前机器上仍然没有 `trtexec`
    - 但仓库现在已经支持 fallback 到 TensorRT Python API builder
  - 当前 DreamZero-DROID 这条 `wan13 / 1.3B / action_horizon=24` 路线:
    - TRT model type 应该用 `ar_1.3B_droid`
- 新增推理提速开关:
  - `NUM_INFERENCE_STEPS=<N>`:
    - 控制总 scheduler step
    - 默认 `16`
  - `NUM_DIT_STEPS=<N>`:
    - 控制真正执行 DiT forward 的步数
    - 默认 `8`
    - 当总步数不是 `16` 时，当前实现会自动生成均匀分布的 step mask
  - `PREWARM_ON_START=1`:
    - 启动脚本会在服务起来后自动打一轮本地 client 预热
  - `PREWARM_USE_ZERO_IMAGES=1`:
    - 预热时用零图像，避免真实视频加载
  - `PREWARM_NUM_CHUNKS=1`:
    - 预热请求 chunk 数
- 当前不建议继续花时间尝试 4 卡推理:
  - 当前实现只稳定支持 1 卡和 2 卡

## 2026-04-02 TRT 结果

- TensorRT engine 已成功构建:
  - `/home/zqy/ws/dreamzero/outputs/train/2026-03-26/23-21-38-droid-lora/checkpoint-2000/tensorrt/wan/WanModel_fp16.trt`
- 构建配置:
  - GPU: `6`
  - precision: `fp16`
  - model type: `ar_1.3B_droid`
  - engine build time: `154.424s`
- TRT 双卡推理实测:
  - GPUs: `6,7`
  - port: `6007`
  - `--enable-dit-cache`
  - `DREAMZERO_SAVE_RESET_VIDEO=0`
  - client report:
    - `/home/zqy/ws/dreamzero/outputs/reports/20260402_gpu6_7_checkpoint2000_trt_fp16_latency.md`
    - `/home/zqy/ws/dreamzero/outputs/reports/20260402_gpu6_7_checkpoint2000_trt_fp16_latency.json`
  - client latency:
    - `Initial 6.35s`
    - `Chunk 0 15.31s`
    - `Chunk 1 0.76s`
    - `Chunk 2 0.81s`
    - hot avg `0.79s`
  - server timing split:
    - `Initial`: Diffusion `0.80s`, Scheduler `4.04s`
    - `Chunk 0`: Diffusion `0.75s`, Scheduler `14.23s`
    - `Chunk 1`: Diffusion `0.45s`, Scheduler `0.01s`
    - `Chunk 2`: Diffusion `0.49s`, Scheduler `0.01s`

## 2026-04-02 Step Sweep

- 目标:
  - 在 TRT `fp16` 基础上，比较 `NUM_INFERENCE_STEPS / NUM_DIT_STEPS`
  - 每组先做一次零图预热，再做真实视频 `num-chunks 4`
- 对比报告:
  - `/home/zqy/ws/dreamzero/outputs/reports/20260402_stepsweep_trt_fp16_chunks4_summary.md`
- 结果:
  - `12/6`
    - Initial `13.79s`
    - Chunk 0 `0.61s`
    - Chunk 1 `0.64s`
    - Chunk 2 `0.69s`
    - Chunk 3 `1.15s`
    - hot avg `0.83s`
  - `8/5`
    - Initial `11.04s`
    - Chunk 0 `0.56s`
    - Chunk 1 `0.58s`
    - Chunk 2 `0.62s`
    - Chunk 3 `1.10s`
    - hot avg `0.77s`
  - `8/4`
    - Initial `9.81s`
    - Chunk 0 `0.51s`
    - Chunk 1 `0.52s`
    - Chunk 2 `0.55s`
    - Chunk 3 `1.06s`
    - hot avg `0.71s`
- 当前速度优先推荐:
  - `NUM_INFERENCE_STEPS=8`
  - `NUM_DIT_STEPS=4`
- 额外观察:
  - 第 4 个 chunk 一直偏慢
  - 原因是 `current_start_frame >= local_attn_size` 后触发重置，导致该步重新走 image encoder

## 2026-04-02 额外提速修复

- 已新增:
  - `socket_test_optimized_AR.py --max-chunk-size <N>`
    - 现在会真正覆盖推理时的 `local_attn_size`
  - prompt embedding cache
    - 语言不变时，不再重复跑 text encoder
- 关键代码:
  - `/home/zqy/ws/dreamzero/socket_test_optimized_AR.py`
  - `/home/zqy/ws/dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- 实测:
  - 配置:
    - TRT `fp16`
    - `NUM_INFERENCE_STEPS=8`
    - `NUM_DIT_STEPS=4`
    - `--max-chunk-size 8`
  - 报告:
    - `/home/zqy/ws/dreamzero/outputs/reports/20260402_gpu6_7_checkpoint2000_trt_fp16_steps8_dit4_chunks4_maxchunk8.md`
    - `/home/zqy/ws/dreamzero/outputs/reports/20260402_gpu6_7_checkpoint2000_trt_fp16_steps8_dit4_chunks4_maxchunk8.json`
  - 结果:
    - Initial `4.31s`
    - Chunk 0 `9.15s`
    - Chunk 1 `0.48s`
    - Chunk 2 `0.50s`
    - Chunk 3 `0.52s`
  - 服务端拆分:
    - Initial: Text `0.24s`, Image `0.90s`, Diffusion `0.38s`, Scheduler `2.35s`
    - Chunk 0: Text `0.00s`, Image `0.00s`, Diffusion `0.24s`, Scheduler `8.66s`
    - Chunk 1: Text `0.00s`, Image `0.00s`, Diffusion `0.23s`
    - Chunk 2: Text `0.00s`, Image `0.00s`, Diffusion `0.25s`
    - Chunk 3: Text `0.00s`, Image `0.00s`, Diffusion `0.26s`
- 结论:
  - 第 4 个 chunk 的重置问题已经通过更大的 `local_attn_size` 避开
  - prompt cache 让热态 text encoder 基本降到 `0.00s`
  - 如果当前目标是速度优先，推荐:
    - `NUM_INFERENCE_STEPS=8`
    - `NUM_DIT_STEPS=4`
    - `--max-chunk-size 8`

## 2026-04-02 更激进步数实验

- 新增报告:
  - `/home/zqy/ws/dreamzero/outputs/reports/20260402_speed_quality_tradeoff.md`
- 速度结果:
  - `8/4`
    - Initial `4.31s`
    - Chunk 0 `9.15s`
    - Chunk 1 `0.48s`
    - Chunk 2 `0.50s`
    - Chunk 3 `0.52s`
  - `6/3`
    - Initial `9.96s`
    - Chunk 0 `5.93s`
    - Chunk 1 `0.42s`
    - Chunk 2 `0.44s`
    - Chunk 3 `0.45s`
  - `4/2`
    - Initial `8.31s`
    - Chunk 0 `4.98s`
    - Chunk 1 `0.38s`
    - Chunk 2 `0.38s`
    - Chunk 3 `0.39s`
- 质量快速判断:
  - 对比了 `8/4` 和 `4/2` 的保存视频
  - `4/2` 没有出现明显崩坏或结构错误
  - 但观感上比 `8/4` 略软一些
- 当前建议:
  - 极限速度优先:
    - `NUM_INFERENCE_STEPS=4`
    - `NUM_DIT_STEPS=2`
    - `--max-chunk-size 8`
  - 速度/质量更稳妥:
    - `NUM_INFERENCE_STEPS=6`
    - `NUM_DIT_STEPS=3`
    - `--max-chunk-size 8`

## 当前推理耗时记录方式

- `test_client_AR.py` 现在默认会输出两份客户端耗时报告:
  - `outputs/reports/<timestamp>_test_client_ar_latency.md`
  - `outputs/reports/<timestamp>_test_client_ar_latency.json`
- 报告内容包含:
  - `Initial`
  - 每个 `Chunk`
  - 总推理时间
  - 平均推理时间
  - 最慢 step
  - 热态 chunk 平均时间
- 本轮新增优化报告:
  - `/home/zqy/ws/dreamzero/outputs/reports/20260402_inference_acceleration_notes.md`

## 重要文档

- 训练说明:
  - `/home/zqy/ws/dreamzero/docs/TRAINING_GUIDE.md`
- 推理说明:
  - `/home/zqy/ws/dreamzero/docs/INFERENCE_SERVICE_GUIDE.md`
- DROID 转换:
  - `/home/zqy/ws/dreamzero/docs/DROID_CONVERSION.md`
- 新数据集接入:
  - `/home/zqy/ws/dreamzero/docs/DATASET_TO_GEAR_AND_TRAIN.md`

## 当前协作偏好

- 用户希望优先把流程跑通，再做更系统的整理和优化。
- 训练相关优先记录:
  - GPU 选择
  - 数据集路径
  - 输出目录
  - 首个可用 checkpoint
  - loss 趋势
- 推理相关优先记录:
  - GPU 和端口
  - 输入 prompt 和图像来源
  - chunk schedule
  - 每个 chunk 的耗时
  - 是否保存视频
  - 日志与产物路径

## 更新规则

- 新增或修改脚本后，同步补充这里的“输出与目录约定”“已知坑”或“当前状态”。
- 新跑完实验后，补充:
  - 时间
  - GPU 配置
  - 端口
  - 模型路径
  - 关键耗时
  - 产物路径
- 如果某个方向被证明不成立，也要记在这里，避免重复排查。
