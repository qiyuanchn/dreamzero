# DreamZero 推理服务说明

这份文档整理了当前仓库里 DreamZero 推理服务的实际用法，包含：
- 单卡 / 双卡启动
- 端口选择
- `screen` 后台托管
- `zero-image` 与真实视频验证
- 常见报错和建议做法

适用仓库路径：`/home/zqy/ws/dreamzero`

## 1. 基本结论

当前这套 1.3B 推理路线：
- backbone: `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP`
- tokenizer: `umt5-xxl`
- server 入口: `socket_test_optimized_AR.py`
- client 入口: `test_client_AR.py`

当前验证状态：
- 单卡可以稳定跑 `zero-image` 和真实视频的首帧推理
- 单卡在真实视频 `4` 帧 chunk 上容易 OOM
- 双卡可以跑通真实视频首帧和后续 `4` 帧 chunk
- `reset` 阶段保存视频仍然可能 OOM
- 服务端返回的动作是 `(24, 8)`，由：
  - `action.joint_position: (24, 7)`
  - `action.gripper_position: (24,)`
  拼接得到

## 2. 环境准备

进入环境：

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
export PYTHONPATH=/home/zqy/ws/dreamzero
```

常用模型路径：

```bash
MODEL_PATH=/home/zqy/ws/dreamzero/checkpoints/DreamZero-DROID
WAN_CKPT_DIR=/home/zqy/ws/dreamzero/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP
TOKENIZER_DIR=/home/zqy/ws/dreamzero/checkpoints/umt5-xxl
```

## 3. 直接命令启动服务

### 3.1 单卡启动

推荐端口：`5999`

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
PYTHONPATH=/home/zqy/ws/dreamzero \
CUDA_VISIBLE_DEVICES=0 \
DREAMZERO_SAVE_RESET_VIDEO=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run --standalone --nproc_per_node=1 \
socket_test_optimized_AR.py \
  --port 5999 \
  --timeout-seconds 50000 \
  --model-path /home/zqy/ws/dreamzero/checkpoints/DreamZero-DROID
```

适合：
- `zero-image` 验证
- 服务是否能加载
- 真实视频首帧验证

不适合：
- 真实视频多 chunk 长时间推理

### 3.2 双卡启动

推荐端口：`6000`

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
PYTHONPATH=/home/zqy/ws/dreamzero \
CUDA_VISIBLE_DEVICES=0,1 \
DREAMZERO_SAVE_RESET_VIDEO=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
socket_test_optimized_AR.py \
  --port 6000 \
  --timeout-seconds 50000 \
  --model-path /home/zqy/ws/dreamzero/checkpoints/DreamZero-DROID
```

适合：
- 真实视频多 chunk 推理
- 长时间推理验证

当前实际验证结果：
- 单卡：真实视频首帧成功，进入 `4` 帧 chunk 后 OOM
- 双卡：真实视频首帧、`Chunk 0`、`Chunk 1` 都成功

## 4. 用脚本启动服务

脚本路径：
- 启动：`scripts/inference/start_dreamzero_service.sh`
- 停止：`scripts/inference/stop_dreamzero_service.sh`

### 4.1 单卡脚本启动

```bash
CUDA_VISIBLE_DEVICES=0 \
PORT=5999 \
MODEL_PATH=/home/zqy/ws/dreamzero/checkpoints/DreamZero-DROID \
bash /home/zqy/ws/dreamzero/scripts/inference/start_dreamzero_service.sh
```

### 4.2 双卡脚本启动

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PORT=6000 \
MODEL_PATH=/home/zqy/ws/dreamzero/checkpoints/DreamZero-DROID \
bash /home/zqy/ws/dreamzero/scripts/inference/start_dreamzero_service.sh
```

默认行为：
- 后台用 `screen` 托管服务
- 默认关闭 `reset` 时视频保存：`DREAMZERO_SAVE_RESET_VIDEO=0`
- 日志写到：`logs/dreamzero_service_port_<PORT>.log`

## 5. screen 怎么看输出

启动脚本会把服务放进一个 `screen` 会话里。

默认：
- session: `dreamzero_service`
- window: `port_<PORT>`

例如端口 `6000`：
- session: `dreamzero_service`
- window: `port_6000`

查看输出：

```bash
screen -r dreamzero_service
```

进入后常用操作：
- `Ctrl-a "`：看窗口列表
- `Ctrl-a n`：下一个窗口
- `Ctrl-a p`：上一个窗口
- `Ctrl-a d`：脱离 screen，不停止服务

如果你只想看日志，也可以：

```bash
tail -f /home/zqy/ws/dreamzero/logs/dreamzero_service_port_6000.log
```

## 6. 停止服务

### 6.1 用脚本停止

```bash
PORT=6000 bash /home/zqy/ws/dreamzero/scripts/inference/stop_dreamzero_service.sh
```

### 6.2 手动停止 screen 中的服务

进到 screen：

```bash
screen -r dreamzero_service
```

切到对应窗口后：
- `Ctrl-c` 终止当前服务
- `Ctrl-a d` 脱离 screen

## 7. 客户端验证

### 7.1 zero-image 最小验证

这一步只验证：
- 服务能否启动
- WebSocket 是否正常
- 模型能否返回 action

单卡示例：

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
PYTHONPATH=/home/zqy/ws/dreamzero \
python test_client_AR.py --host 127.0.0.1 --port 5999 --use-zero-images --num-chunks 1
```

双卡示例：

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
PYTHONPATH=/home/zqy/ws/dreamzero \
python test_client_AR.py --host 127.0.0.1 --port 6000 --use-zero-images --num-chunks 1
```

当前实际结果：
- 能返回 `Action shape: (24, 8)`
- 耗时约 `9s~10s`

### 7.2 真实视频验证

这一步会读取：
- `debug_image/exterior_image_1_left.mp4`
- `debug_image/exterior_image_2_left.mp4`
- `debug_image/wrist_image_left.mp4`

并按以下 schedule 发送：
- 初始：`[0]`
- `Chunk 0`: `[0, 7, 15, 23]`
- `Chunk 1`: `[24, 31, 39, 47]`

双卡推荐命令：

```bash
source /home/zqy/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /home/zqy/ws/dreamzero
PYTHONPATH=/home/zqy/ws/dreamzero \
python test_client_AR.py --host 127.0.0.1 --port 6000 --num-chunks 2
```

当前实际结果：
- 单卡：
  - 首帧成功
  - 第一个 `4` 帧 chunk OOM
- 双卡：
  - 首帧成功
  - `Chunk 0` 成功
  - `Chunk 1` 成功

双卡本次真实视频验证耗时参考：
- 初始单帧：约 `9.96s`
- `Chunk 0`: 约 `40.63s`
- `Chunk 1`: 约 `1.34s`

## 8. 输入 / 输出格式

### 8.1 服务端先发什么

服务端启动后会先发 metadata，客户端会据此构造 observation。主要字段有：
- `image_resolution: [180, 320]`
- `needs_wrist_camera: True`
- `n_external_cameras: 2`
- `needs_session_id: True`
- `action_space: joint_position`

### 8.2 客户端发什么

`zero-image` 模式下，客户端会发：
- `observation/exterior_image_0_left`
- `observation/exterior_image_1_left`
- `observation/wrist_image_left`
- `observation/joint_position`
- `observation/cartesian_position`
- `observation/gripper_position`
- `prompt`
- `session_id`

真实视频模式下，图像不是全 0，而是从 `debug_image/*.mp4` 里抽帧。

### 8.3 服务端返回什么

当前服务端返回：
- `action.joint_position: (24, 7)`
- `action.gripper_position: (24,)`

然后服务端适配层会拼成客户端看到的：
- `Action shape: (24, 8)`

这不是“补出来的假第 8 维”，而是：
- 7 维 joint
- 1 维 gripper

## 9. 为什么推理结束后显存还在

这是服务模式的正常现象，不是单独的 bug。

原因包括：
- 模型本体常驻 GPU
- `torch.compile` / CUDA Graph 占用 private pools
- KV cache / cross-attn cache 还在
- PyTorch allocator 会保留 reserved memory，方便下次推理复用

想继续当服务用：
- 显存就应该保留一部分

想真正释放显存：
- 最有效的方法是停掉服务进程

我已经在服务端 reset 清理后加了：
- `torch.cuda.empty_cache()`

但这只能帮一点，不能替代停服务。

## 10. 为什么 reset 还会 OOM

`reset` 默认会做一件额外的事：
- 把累计的 `video_across_time` 用 VAE 解码成帧
- 再保存成 mp4

这一步额外吃显存，所以容易在推理结束后报：
- `Failed to save video on reset: CUDA out of memory`

现在已经加了开关：

```bash
DREAMZERO_SAVE_RESET_VIDEO=0
```

建议默认关掉，除非你真的需要自动保存视频。

## 11. 常见问题

### 11.1 启动脚本执行后立刻返回，是不是没启动？

不是。

现在脚本是用 `screen` 在后台托管服务：
- 脚本本身会立即返回
- 真正的服务在 `screen` 窗口里继续跑

请看：
- `screen -r dreamzero_service`
- 或者 `tail -f logs/dreamzero_service_port_<PORT>.log`

### 11.2 为什么 zero-image 能跑，真实视频 chunk 跑不动？

因为真实视频 `4` 帧 chunk 显存压力明显更大：
- 输入帧数更多
- latent 更大
- 中间激活更大

当前验证表明：
- 单卡只适合最小验证和真实首帧
- 双卡更适合真实视频 chunk

### 11.3 推荐端口怎么选？

建议：
- 单卡：`5999`
- 双卡：`6000`

这样你自己区分服务方便，也不容易和之前的测试混在一起。

## 12. 推荐工作流

### 12.1 只想确认服务能不能跑

```bash
CUDA_VISIBLE_DEVICES=0 PORT=5999 bash scripts/inference/start_dreamzero_service.sh
PYTHONPATH=/home/zqy/ws/dreamzero python test_client_AR.py --host 127.0.0.1 --port 5999 --use-zero-images --num-chunks 1
```

### 12.2 想做真实视频验证

```bash
CUDA_VISIBLE_DEVICES=0,1 PORT=6000 bash scripts/inference/start_dreamzero_service.sh
PYTHONPATH=/home/zqy/ws/dreamzero python test_client_AR.py --host 127.0.0.1 --port 6000 --num-chunks 2
```

### 12.3 验证完停服务

```bash
PORT=6000 bash scripts/inference/stop_dreamzero_service.sh
```

## 13. 你现在最该记住的结论

- 单卡：适合最小验证，不适合真实多帧 chunk
- 双卡：适合真实视频推理
- 默认建议关闭 `reset save video`
- 服务端输出是 `(24, 8)`，来自 `7维 joint + 1维 gripper`
- 看后台输出优先用 `screen -r dreamzero_service`
