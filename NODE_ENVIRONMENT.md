# The A3-Ultra H200 cluster this repo is being run on

Written against `bingghhe123051-0145-20260904-123051-node-0-*`: two Google Cloud A3-Ultra nodes,
eight H200s each, 141 GB per GPU. Nothing here is specific to one experiment — the filesystem
layout, the NCCL setup and the launch recipe are the same for any run on these nodes.

This file is not published to the docs site (`mkdocs` builds `docs/` only).

A 16-GPU smoke (`max_train_steps=2`) completed on 2026-09-07: both nodes reached
`Training completed`, DCP wrote `checkpoint-2`, W&B run
`https://wandb.ai/hbin/fastvideo_h3_proxy/runs/pjagwm6a` (`finetune_loss ≈ 0.16`,
`step_time_sec ≈ 30`). That is the known-good baseline this page describes.

## Filesystem: what survives the node and what does not

| Path | Lifetime | What belongs there |
| --- | --- | --- |
| `/data` | persistent disk | caches, checkpoints, manifests, model snapshots |
| `/workspace` | dies with the node | the git checkout, and nothing else |
| `/opt/venv` | image | the Python 3.12 environment (`python` is already this one) |
| `/tmp` | dies with the node | `tee` logs only. Do not put checkpoints here. |

The rule that follows: **every output path in a config must be absolute and under `/data`.** A
relative `output_dir` resolves against the launch directory, which is the checkout under
`/workspace`, so a relative path silently puts the checkpoints *and* the validation mp4s on the
disk that disappears.

`tee` is the exception. A long run that `tee`s onto `/data` has already hit
`Stale file handle` (gcsfuse / FUSE releasing the fd after a quiet period). Log to
`/tmp/train16.log` and keep artifacts under `/data`.

Known-good locations:

- `/data/models/MiniMax-H3` — the 134 GB Ref2VA snapshot. Verify before spending an encode on it:
  `python scripts/h3_proxy/prepare_models/verify_h3_snapshot.py --path /data/models/MiniMax-H3 --profile ref2va`
- `/data/binghe/datasets/` — raw datasets
- `/data/binghe/h3_proxy/` — manifests, `cache/`, `runs/`

The earlier `gcsfuse` faults that forced a staged write through `/workspace` (`Errno 107`, `SIGBUS`
under sustained throughput) are fixed. Write straight to `/data`.

The checkout is at `/workspace/FastVideo` and it is the one that imports —
`python -c "import fastvideo; print(fastvideo.__file__)"` gives
`/workspace/FastVideo/fastvideo/__init__.py`. Worth re-checking after any environment change,
because a stale `/FastVideo` copy exists on the image and a run importing that one ignores every
local edit.

`/data` is shared enough for 16-rank training: both nodes read the same H3 snapshot and the same
`.pt` cache, and the smoke run wrote `checkpoint-2` from a two-node job. Still worth a glance after
a container restart (`ls /data/binghe/h3_proxy/cache/abot_train | wc -l` on node-1, expect ~9865).

## Interconnect

- Intra-node: NVLink, used by NCCL's P2P transport. Logs look like
  `Channel N : 0[0] -> 1[1] via P2P/CUMEM`. That is local, not gIB.
- Inter-node: 8 × Mellanox RoCE NICs, one per GPU (`mlx5_0`..`mlx5_7` on
  `192.168.{4,8,...,80}.x`), plus `eth0` (`10.x`) for management and torchrun rendezvous.
  NCCL data across nodes goes through Google's gIB plugin. Use the `10.x` / hostname for
  `MASTER_ADDR`, never a `192.168.x` RDMA address.

Present and correct on this image: `/dev/infiniband/uverbs0..7`, `/sys/class/infiniband/mlx5_0..7`,
`/usr/local/gib/lib64/libnccl-net.so`, `/usr/local/gib/configs/`.

## Every new container: libibverbs

**`gIB` dlopens `libibverbs.so.1` at runtime, and it is not installed.** Both nodes, after every
container restart:

```bash
apt-get update && apt-get install -y libibverbs1 ibverbs-providers ibverbs-utils && ldconfig
ibv_devinfo | grep -E "hca_id|state"      # expect mlx5_0..7, PORT_ACTIVE
```

`ibverbs-providers` is not optional — it supplies `libmlx5`, and without it the library opens but
enumerates zero devices, which fails in exactly the same way. `ibverbs-utils` is only for
`ibv_devinfo`.

Google's A3-Ultra guidance requires `rdma-core` in the base image; getting it added there removes
this step.

### Why the failure looks like something else

`/usr/local/gib/scripts/set_nccl_env.sh` sets `NCCL_CONF_FILE` to a file whose first line is
`NCCL_NET=gIB`. A forced network with no usable devices has no fallback, on one node as readily
as on two:

```
NCCL INFO NET/Plugin: Loaded net plugin gIB (v11)          <- the plugin loads fine
NCCL INFO Successfully loaded external network plugin ...   <- so does its config
NCCL INFO Failed to open libibverbs.so[.1]                  <- the actual cause, at INFO level
NCCL WARN NCCLCHECK failed with 3: ncclIbInitDevices(...)
NCCL WARN Failed to initialize any NET plugin
RuntimeError: NCCL error: invalid usage       (or: internal error)
```

The cause prints at `INFO`, three lines *above* the first `WARN`. `NCCL_DEBUG=WARN` shows only the
consequence. Use `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` and `grep -B8`. `ldd libnccl-net.so`
is clean because the dependency is `dlopen`ed, not linked.

### Three plausible-sounding fixes that are wrong

All three were tried here. Each one looks right from the error message alone.

- **`NCCL_P2P_DISABLE=1`** (copied from older launchers under `examples/`). Disables NVLink and
  pushes every collective onto the network plugin.
- **`unset NCCL_CONF_FILE; export NCCL_NET_PLUGIN=none`** on single-node runs. Works by falling
  back to Socket and hides the missing library until the first multi-node job.
- **Pinning NCCL to the system 2.23.4** via `LD_PRELOAD` or `FASTVIDEO_NCCL_SO_PATH`. Two
  libraries exist — PyTorch 2.29.3 at `/opt/venv/.../nvidia/nccl/lib/libnccl.so.2` and 2.23.4 at
  `/usr/lib/x86_64-linux-gnu/` — but the plugin exports `ncclNetPlugin_v8` and `ncclNetPlugin_v11`
  and loads into 2.29.3. There is no version problem. `LD_LIBRARY_PATH` cannot choose between them
  (PyTorch uses an RPATH). `nccl-gib-plugins` ships plugins only; there is no `libnccl.so.2` under
  `/usr/local/gib/lib64`.

## The launch environment

Same for one node and for two, once libibverbs is installed:

```bash
cd /workspace/FastVideo
source /usr/local/gib/scripts/set_nccl_env.sh
export LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH}
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
# H200 + checkpoint_wrapper otherwise compiles a Triton permute that the driver rejects.
# Must be in the shell *before* torchrun, not only inside the Python process.
export TORCHINDUCTOR_DISABLE=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_NET_PLUGIN FASTVIDEO_NCCL_SO_PATH LD_PRELOAD
```

Do not add anything else. A stale `NCCL_NET_PLUGIN=none` sends a 16-rank job over TCP sockets at a
fraction of the bandwidth, with no error.

`TORCH_NCCL_ENABLE_MONITORING=0` keeps the watchdog from killing a rank during the long first
compile and the text-encoder load. `TOKENIZERS_PARALLELISM=false` silences a warning after the
first fork.

`/dev/shm` is 256 GB, which is ample.

## Rendezvous: what the platform actually injects

`torchrun` does **not** read `PET_*` as its own flags. Empty `PET_NNODES` plus a bare
`torchrun -m ...` is a **one-node, eight-process** job, which then dies with

```
mesh should not be bigger than the default world size 8, but get 16
```

if the YAML/CLI asked for 16. `init_device_mesh` looks at `WORLD_SIZE`, not at `num_gpus` in the
config.

Observed on this cluster (both nodes; names stay the same across shells):

| Variable | node-0-0 | node-0-1 |
| --- | --- | --- |
| `PET_MASTER_ADDR` | `bingghhe123051-0145-20260904-123051-node-0-0.bingghhe123051-0145-20260904-123051` | same |
| `PET_MASTER_PORT` | `29500` | `29500` |
| `PET_NODE_RANK` | `0` | `1` |
| `PET_NNODES` | empty | empty |
| `PET_NPROC_PER_NODE` | `auto` | `auto` |
| `MASTER_ADDR` / `MASTER_PORT` | empty | empty |

`getent hosts` of that master hostname resolves to `10.33.34.12` (management `eth0`). Use
`$PET_MASTER_ADDR` as `--master_addr`; do not invent a `192.168.x` RDMA IP.

Always pass `--nnodes` / `--nproc_per_node` / `--node_rank` / `--master_addr` / `--master_port`
explicitly for a 16-GPU job. Both nodes run the **same command**; only `$PET_NODE_RANK` differs.

For a single-node run, add `--standalone` and do **not** pass `--nnodes 2`. `PET_NODE_RANK` being
set is not enough to wait for a peer, but `--nnodes 2` without a second process will hang at
rendezvous.

Before any launch: `nvidia-smi` on both nodes. A leftover encode shard (~64 GB text encoder per
GPU) surfaces as an NCCL init error, not an OOM.

## Sizing the mesh

`nnodes * nproc_per_node == num_gpus == hsdp_replicate_dim * hsdp_shard_dim`.
`init_device_mesh` rejects a mesh larger than the world size; a smaller mesh leaves GPUs idle
without complaint.

Global batch is `(num_gpus / sp_size) * train_batch_size * gradient_accumulation_steps`. Doubling
nodes at fixed global batch means **halving `gradient_accumulation_steps`**, not touching the
learning rate. `max_train_steps` stays 3702.

| | 1 node | 2 nodes |
| --- | --- | --- |
| `num_gpus` | 8 | 16 |
| `hsdp_shard_dim` | 8 | 16 |
| `sp_size` | 8 | 8 |
| `gradient_accumulation_steps` | 8 | 4 |
| global batch | 8 | 8 |

## Page cache: warm the snapshot before a launch

Sixteen ranks opening the same safetensors on `/data` at once make the first shards look frozen
(`0/14`) and later shards drop from ~2 s to ~50 s. The kernel page cache is **per node**. Warm
both, **before** `torchrun`, not during a load (that competes for the same bandwidth).

```bash
# both nodes. Sequential read into the page cache.
for d in \
    /data/models/MiniMax-H3/text_encoder \
    /data/models/MiniMax-H3/transformer_ref \
    /data/models/MiniMax-H3/vae
do
    echo "warming $d"
    find "$d" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.json' \) \
        -print -exec dd if={} of=/dev/null bs=16M status=none \;
done
echo done
```

If `vmtouch` is available:

```bash
apt-get install -y vmtouch
vmtouch -t -m 200G /data/models/MiniMax-H3/text_encoder /data/models/MiniMax-H3/transformer_ref
vmtouch -v /data/models/MiniMax-H3/text_encoder | tail
```

`text_encoder` (~64 GB, 14 shards) is what step-0 validation loads. `transformer_ref` is the
training DiT. `/data` via gcsfuse still benefits: warming pulls object blocks into the local FUSE
cache. Host RAM on these nodes is enough for both trees under a 200 GB cap.

Do not `echo 3 > /proc/sys/vm/drop_caches` afterwards.

## Bring-up checklist (every new experiment)

On **both** nodes, in order:

1. `apt-get install -y libibverbs1 ibverbs-providers ibverbs-utils && ldconfig`
2. `ibv_devinfo | grep -E "hca_id|state"` — `mlx5_0..7`, `PORT_ACTIVE`
3. `python -c "import fastvideo; print(fastvideo.__file__)"` — `/workspace/FastVideo/...`
4. Source the launch environment (section above) and `unset` the stale NCCL overrides
5. Confirm rendezvous: `echo $PET_MASTER_ADDR $PET_MASTER_PORT $PET_NODE_RANK`
6. Warm the snapshot (section above)
7. `nvidia-smi` — empty
8. Launch. `tee` to `/tmp`, not `/data`

## Launch recipes

All of these assume the environment block above is already in the shell.

### Two-GPU dry-run (NCCL + config only)

`--dry-run` parses the config, builds the model / dataloader / process groups, then exits. After
NCCL prints `P2P/CUMEM` it will sit silent for many minutes while H3 loads; wait for
`Dry-run: config parsed and build_from_config succeeded.` then `Training completed`. It exits
on its own. Do not Ctrl+C.

```bash
unset PET_NNODES
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET torchrun --standalone --nproc_per_node 2 \
    -m fastvideo.train.entrypoint.train \
    --config examples/train/scenario/h3_proxy/proxy_bd_finetune_abot.yaml \
    --training.distributed.num_gpus 2 \
    --training.distributed.sp_size 2 \
    --training.distributed.hsdp_shard_dim 2 \
    --dry-run 2>&1 | tee /tmp/dryrun2.log
```

Must contain `Using network gIB` and no `Failed to open libibverbs`. `Using network Socket` is a
silent fallback — the job will run, slowly.

### Eight-GPU smoke, this node only

Do not pass `num_gpus 16`. YAML defaults are already 8 / 8 / 8.

```bash
torchrun --standalone --nproc_per_node 8 \
    -m fastvideo.train.entrypoint.train \
    --config examples/train/scenario/h3_proxy/proxy_bd_finetune_abot.yaml \
    --training.loop.max_train_steps 2 \
    --training.checkpoint.output_dir /data/binghe/h3_proxy/runs/smoke8/checkpoints \
    --training.tracker.run_name h3_abot_smoke8 \
    --callbacks.validation.run_at_start false \
    2>&1 | tee /tmp/smoke8.log
```

### Sixteen-GPU smoke (skip step-0 validation)

Same command on both nodes. Confirms the training loop without paying for Qwen3-VL.

```bash
torchrun --nnodes 2 --nproc_per_node 8 \
    --node_rank "$PET_NODE_RANK" \
    --master_addr "$PET_MASTER_ADDR" --master_port "$PET_MASTER_PORT" \
    -m fastvideo.train.entrypoint.train \
    --config examples/train/scenario/h3_proxy/proxy_bd_finetune_abot.yaml \
    --training.distributed.num_gpus 16 \
    --training.distributed.hsdp_shard_dim 16 \
    --training.loop.gradient_accumulation_steps 4 \
    --training.loop.max_train_steps 2 \
    --training.checkpoint.output_dir /data/binghe/h3_proxy/runs/smoke16/checkpoints \
    --training.tracker.run_name h3_abot_smoke16 \
    --callbacks.validation.run_at_start false \
    2>&1 | tee /tmp/smoke16.log
```

Success looks like `Steps: 100%|…| 2/2` on **both** nodes, `Saving checkpoint to …/checkpoint-2`,
then `Training completed`. W&B will have scalars and `0 media file(s)` — there was no validation.

### Sixteen-GPU full run (3702 steps, validation on)

```bash
torchrun --nnodes 2 --nproc_per_node 8 \
    --node_rank "$PET_NODE_RANK" \
    --master_addr "$PET_MASTER_ADDR" --master_port "$PET_MASTER_PORT" \
    -m fastvideo.train.entrypoint.train \
    --config examples/train/scenario/h3_proxy/proxy_bd_finetune_abot.yaml \
    --training.distributed.num_gpus 16 \
    --training.distributed.hsdp_shard_dim 16 \
    --training.loop.gradient_accumulation_steps 4 \
    2>&1 | tee /tmp/train16.log
```

Checkpoints: `/data/binghe/h3_proxy/runs/h3_proxy_bd_lora_abot/checkpoints`.
W&B project `fastvideo_h3_proxy`, run `h3_proxy_bd_lora_r128_abot`. Validation every 250 steps
and at step 0.

After the training DiT is up, rank 0 (node-0-0) loads a second pipeline and prints
`Loading pipeline modules` / `Loading text_encoder` / `Loading safetensors checkpoint shards: 0/14`.
That 14 is **Qwen3-VL**, not a second NCCL handshake. Node-0-1 sits on a barrier with almost no
new lines. This is several minutes even with a warm cache; without one, individual shards go from
~2 s to ~50 s. Wait for `14/14`, then another quiet period while it samples 50 steps, then
`validation_videos_*` and the training step counter.

## What looks like a hang and is not

| What you see | What it is |
| --- | --- |
| Last line is `via P2P/CUMEM` / `Connected to proxy` | NCCL init finished; H3 is loading. No more NCCL INFO. |
| `Loading safetensors … 0/14` for a while | tqdm updates only when a whole shard is done. First shard is the slowest. |
| Node-0-0 at `Loading text_encoder` / 14 shards; node-0-1 quiet | Step-0 validation. Rank 0 builds the pipeline; the other node waits. |
| `qwen3_vl_text` / `hidden_size: 5120` config dump | Processor / text-encoder config, not a crash. |
| `[ERROR] min_frames` / `max_frames` … `not documented` | transformers docstring check. Does not raise. |
| `set_vital is deprecated` | `torchdata` StatefulDataLoader. Once per rank, harmless. |
| `grpc_wait_for_shutdown_with_timeout` / GPUViz `[non-fatal]` | Google telemetry on the way out. |
| `tee: … Stale file handle` | FUSE dropped the log fd. Training itself is fine if artifacts are on `/data`. |
| `destroy_process_group() was not called` after a real `Training completed` | Shutdown noise. After a crash it means the *other* symptom, not the cause. |
| W&B `View run` | Printed on any exit after `wandb.init`, including crashes. Look for `Steps: 2/2` / `Training completed` before treating it as success. |
| `RuntimeError: CUDA driver error: invalid argument` inside `torch/_inductor` / `triton_poi_fused_*` during validation | The training DiT is wrapped in `checkpoint_wrapper` (AOTAutograd). Validation unwraps those modules and the entrypoint sets `TORCHINDUCTOR_DISABLE=1`. Do **not** switch checkpointing to `REENTRANT` to dodge this — FSDP + LoRA then yields `element 0 of tensors does not require grad` on the first training backward. |

A real failure has a `Traceback`, `NCCL error`, `CUDA out of memory`, `mesh should not be bigger
than the default world size`, or one node printing `View run` and exiting while the other is still
on `0/14` **and** not reading files (`lsof -p <pid> | grep safetensors` empty, `read_bytes`
frozen). Then Ctrl+C both sides and `pkill -f 'fastvideo.train.entrypoint.train'`.

`--dry-run` and a finished smoke exit by themselves. Do not Ctrl+C because NCCL went quiet.
