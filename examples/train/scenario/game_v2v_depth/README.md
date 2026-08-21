# Game V2V + depth ControlNet (Wan)

Pipeline: **BD → AR → DMD**. Source game clip fills Wan's I2V conditioning slot;
pixel-aligned depth drives a zero-init ControlNet. Optional wide-FOV depth is
enabled in stage 3.

## Models

| Track | Checkpoint | Notes |
| --- | --- | --- |
| **1.3B (debug)** | `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers` | Real I2V 1.3B. Do **not** use `Wan2.1-T2V-1.3B` — it has no condition channels. |
| **A14B (prod)** | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | MoE; configs pick `expert: high`. |

## 0. Data (once)

Manifest JSONL (one clip per line):

```json
{"target": "3a/0001.mp4", "source": "simple/0001.mp4", "depth": "depth/0001.mp4",
 "depth_wide": "depth_wide/0001.mp4", "prompt": "...", "depth_range": [0.1, 500.0]}
```

```bash
# Encode narrow depth + RGB + text (use the same resolution/frames as the YAML)
python scripts/v2v_depth/prepare_data/encode_v2v_depth_samples.py \
  --manifest data/game_v2v_depth/manifest.jsonl \
  --root data/game_v2v_depth/raw \
  --output data/game_v2v_depth/train \
  --model-path weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers \
  --num-frames 81 --height 480 --width 832

# Optional: backfill wide-FOV depth before stage 3
python scripts/v2v_depth/prepare_data/add_wide_depth_latent.py \
  --cache data/game_v2v_depth/train \
  --wide-depth-dir data/game_v2v_depth/raw/depth_wide \
  --model-path weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers
```

For A14B later, re-encode with `--model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers`
(VAE/text encoder must match the train checkpoint).

## 1. Node setup

```bash
bash scripts/bootstrap_node.sh   # clone / pull / editable install
source scripts/node_env.sh
```

## 2. Train — 1.3B (recommended first)

Single node, 8 GPUs:

```bash
# Stage 1 BD (frozen backbone, train ControlNet only)
NUM_GPUS=8 bash examples/train/run.sh \
  examples/train/scenario/game_v2v_depth/stage1_bd_finetune_1p3b.yaml

# Stage 2 AR (teacher-forcing causal)
NUM_GPUS=8 bash examples/train/run.sh \
  examples/train/scenario/game_v2v_depth/stage2_ar_tfsft_1p3b.yaml

# Stage 3 DMD (self-forcing; needs wide depth if enable_wide_fov=true)
NUM_GPUS=8 bash examples/train/run.sh \
  examples/train/scenario/game_v2v_depth/stage3_dmd_self_forcing_1p3b.yaml
```

Dry-run (config + model construct, no training loop):

```bash
NUM_GPUS=1 bash examples/train/run.sh \
  examples/train/scenario/game_v2v_depth/stage1_bd_finetune_1p3b.yaml \
  --dry-run \
  --training.distributed.num_gpus 1 \
  --training.distributed.hsdp_shard_dim 1 \
  --training.distributed.sp_size 1
```

Override data / steps without editing YAML:

```bash
NUM_GPUS=8 bash examples/train/run.sh \
  examples/train/scenario/game_v2v_depth/stage1_bd_finetune_1p3b.yaml \
  --training.data.data_path data/game_v2v_depth/train \
  --training.loop.max_train_steps 100
```

## 3. Train — A14B (2×8 H200)

Configs assume `num_gpus: 16`, `sp_size: 4`. On each node:

```bash
# node 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> NUM_GPUS=8 \
  bash examples/train/run.sh examples/train/scenario/game_v2v_depth/stage1_bd_finetune.yaml

# node 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> NUM_GPUS=8 \
  bash examples/train/run.sh examples/train/scenario/game_v2v_depth/stage1_bd_finetune.yaml
```

Then `stage2_ar_tfsft.yaml` → `stage3_dmd_self_forcing.yaml`.

## Evaluation

Validation runs inside the training loop. `ValidationCallback` injects the live
training transformer into `WanV2VDepthPipeline`, so the ControlNet weights being
trained are the ones sampled — no checkpoint round-trip.

The pipeline deliberately does **not** reuse `WanVideoToVideoPipeline`: that one
builds the Wan-Fun-Control layout `[noise | source | zeros]`, whereas training
builds `[noise | mask | source]`. `WanV2VDepthConditioningStage` reproduces the
training layout, and both it and `encode_v2v_depth_samples.py` share
`fastvideo/dataset/v2v_depth_preprocess.py`, so crop, resize and depth encoding
cannot drift between the cache and validation.

Build the validation set from the same pack the cache came from:

```bash
python scripts/v2v_depth/prepare_data/kof_jsonl_to_validation_json.py \
  --root /data/raw/kof_1k_pre_q_060_rgb \
  --split val --limit 8 \
  --num-frames 81 \
  --crop-top 0.175 --crop-bottom 0.105 \
  --out /data/raw/kof_1k_pre_q_060_rgb/validation_val8.json
```

Each record carries:

| Field | Meaning |
| --- | --- |
| `caption` | prompt |
| `control_video_path` | low-poly source clip — the V2V condition |
| `ref_video` | teacher target, used as the metric reference |
| `crop_*` | **must match the `--crop-*` used to encode the cache** |
| `depth_video_path` | added by `--depth-subdir` once depth renders exist |

`video_path` is intentionally absent: that key makes the callback treat the
clip's first frame as an I2V prompt. Keep `use_validation_media_conditioning:
false` in the YAML for the same reason.

Wire it up per stage (see `stage1_bd_finetune_1p3b_kof_rgb.yaml`):

```yaml
callbacks:
  validation:
    _target_: fastvideo.train.callbacks.validation.ValidationCallback
    pipeline_target: fastvideo.pipelines.basic.wan.wan_v2v_depth_pipeline.WanV2VDepthPipeline
    dataset_file: /data/raw/kof_1k_pre_q_060_rgb/validation_val8.json
    every_steps: 100
    num_frames: 81
    use_validation_media_conditioning: false
    offload_training_state: true
    unload_pipeline_after_validation: true
```

Keep `--limit` small: eight clips at 30 sampling steps already costs real time
on every validation event, and the callback holds a full pipeline in memory
alongside the training state.

## Checkpoint handoff

| From → To | How |
| --- | --- |
| Stage1 → Stage2 | `controlnet_checkpoint` (stage1 only saved `depth_controlnet.*`) |
| Stage2 → Stage3 student | `init_checkpoint` (full transformer) |
| Stage1 → Stage3 teacher/critic | `controlnet_checkpoint` layered on `init_from` |

Update the paths in each YAML if you change `max_train_steps` / save interval.
