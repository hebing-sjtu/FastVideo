# The A3-Ultra H200 cluster this repo is being run on

Written against `bingghhe123051-0145-20260904-123051-node-0-*`: two Google Cloud A3-Ultra nodes,
eight H200s each, 141 GB per GPU. Nothing here is specific to one experiment — the filesystem
layout and the NCCL setup are the same for any run on these nodes, and both have already cost a
day of debugging once.

This file is not published to the docs site (`mkdocs` builds `docs/` only).

## Filesystem: what survives the node and what does not

| Path | Lifetime | What belongs there |
| --- | --- | --- |
| `/data` | persistent disk | caches, checkpoints, manifests, model snapshots |
| `/workspace` | dies with the node | the git checkout, and nothing else |
| `/opt/venv` | image | the Python 3.12 environment (`python` is already this one) |

The rule that follows: **every output path in a config must be absolute and under `/data`.** A
relative `output_dir` resolves against the launch directory, which is the checkout under
`/workspace`, so a relative path silently puts the checkpoints *and* the validation mp4s on the
disk that disappears.

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

**Unverified, and it matters:** whether `/data` is shared between the two nodes. A 16-rank job needs
every rank to read the same `.pt` cache and write into the same checkpoint directory. Confirm with
`mount | grep /data` on both, plus `ls /data/binghe/h3_proxy/cache/abot_train | wc -l` on node 1.
If it is per-node local, the cache has to be replicated before multi-node training is possible.

## Interconnect

- Intra-node: NVLink, used by NCCL's P2P transport.
- Inter-node: 8 × Mellanox RoCE NICs, one per GPU (`mlx5_0`..`mlx5_7` on
  `192.168.{4,8,...,80}.x`), plus `eth0` for management. NCCL reaches them through Google's gIB
  plugin.

Present and correct on this image: `/dev/infiniband/uverbs0..7`, `/sys/class/infiniband/mlx5_0..7`,
`/usr/local/gib/lib64/libnccl-net.so`, `/usr/local/gib/configs/`.

## The one thing the image is missing: libibverbs

**`gIB` dlopens `libibverbs.so.1` at runtime, and it is not installed.** Fix it once per container:

```bash
apt-get update && apt-get install -y libibverbs1 ibverbs-providers ibverbs-utils && ldconfig
ibv_devinfo | grep -E "hca_id|state"      # expect mlx5_0..7, PORT_ACTIVE
```

`ibverbs-providers` is not optional — it supplies `libmlx5`, and without it the library opens but
enumerates zero devices, which fails in exactly the same way. `ibverbs-utils` is only for
`ibv_devinfo`, which is the fastest way to tell a driver problem from a NCCL problem.

This is missing from the image rather than from the machine, so **it must be reinstalled after every
container restart, on both nodes.** Google's own A3-Ultra guidance requires `rdma-core` in the base
image; getting it added there is the real fix and removes this whole section.

### Why this is worth a section

The failure is loud but points nowhere near the cause, and it is reachable from a *single-node* run
too, because `/usr/local/gib/scripts/set_nccl_env.sh` sets `NCCL_CONF_FILE` to a file whose first
line is `NCCL_NET=gIB`. A forced network with no usable devices has no fallback:

```
NCCL INFO NET/Plugin: Loaded net plugin gIB (v11)          <- the plugin loads fine
NCCL INFO Successfully loaded external network plugin ...   <- so does its config
NCCL INFO Failed to open libibverbs.so[.1]                  <- the actual cause, at INFO level
NCCL WARN NCCLCHECK failed with 3: ncclIbInitDevices(...)
NCCL WARN Failed to initialize any NET plugin
RuntimeError: NCCL error: invalid usage       (or: internal error)
```

Two properties make this expensive to chase:

- The cause prints at `INFO`, three lines *above* the first `WARN`. `NCCL_DEBUG=WARN` shows only
  the consequence, and `grep -A` around the warning finds the Python traceback rather than the
  reason. Use `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` and `grep -B8`.
- `ldd libnccl-net.so` is clean, because the dependency is `dlopen`ed rather than linked. The
  plugin looks perfectly healthy under every static check.

### Three plausible-sounding fixes that are wrong

All three were tried here. Recording them because each one looks right from the error message alone.

- **`NCCL_P2P_DISABLE=1`** (copied from older launchers under `examples/`). This disables NVLink and
  pushes every collective onto the network plugin — the opposite of a workaround for a broken
  network plugin. It costs throughput even when the plugin works.
- **`unset NCCL_CONF_FILE; export NCCL_NET_PLUGIN=none`** to decline gIB on single-node runs. It
  does work, by making NCCL fall back to Socket, and it hides the missing library until the first
  multi-node job. Intra-node data goes over NVLink either way, so the speed cost is invisible and
  the diagnosis stays wrong.
- **Pinning NCCL to the system 2.23.4** via `LD_PRELOAD` or `FASTVIDEO_NCCL_SO_PATH`, on the theory
  that the plugin was built against a different NCCL than PyTorch ships. There are indeed two
  libraries here — PyTorch's 2.29.3 at `/opt/venv/.../nvidia/nccl/lib/libnccl.so.2` and 2.23.4 at
  `/usr/lib/x86_64-linux-gnu/` — but the plugin exports both `ncclNetPlugin_v8` and
  `ncclNetPlugin_v11` and loads cleanly into 2.29.3. There is no version problem. (`LD_LIBRARY_PATH`
  cannot select between them anyway: PyTorch finds its own copy through an RPATH, which outranks
  it. Only `LD_PRELOAD` would, which is a reason to be suspicious of needing it.)

Note also that `nccl-gib-plugins` installs *plugins only*. There is no `libnccl.so.2` under
`/usr/local/gib/lib64`, so pointing `FASTVIDEO_NCCL_SO_PATH` there finds nothing.

## The launch environment

Once `libibverbs1` and `ibverbs-providers` are installed, this is the whole of it, and it is the
same for one node and for two:

```bash
source /usr/local/gib/scripts/set_nccl_env.sh
export LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH}
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
```

Do not add anything else. In particular do not set `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`,
`NCCL_NET_PLUGIN` or `FASTVIDEO_NCCL_SO_PATH`, and if a previous debugging session exported them,
`unset` them — a stale `NCCL_NET_PLUGIN=none` in the shell is enough to send a 16-rank job over
TCP sockets at a fraction of the bandwidth, with no error.

`TORCH_NCCL_ENABLE_MONITORING=0` keeps the watchdog from killing a rank during the long
first-iteration compile and the multi-minute text-encoder load. `TOKENIZERS_PARALLELISM=false`
silences a warning that appears once per rank after the first fork.

Two more that are situational:

- **Single node:** `unset PET_NNODES` and add `--standalone`. The platform injects multi-node
  rendezvous variables even into single-node jobs, and `torchrun` will otherwise wait for a peer
  that never arrives.
- **Before any launch:** check the GPUs are free (`nvidia-smi`). The encode step runs one process
  per GPU each holding a ~64 GB text encoder, and a detached shard that outlived its shell leaves
  too little memory for NCCL to allocate its buffers — which surfaces as an init error, not an OOM.

`/dev/shm` is 256 GB, which is ample; it is a common cause of NCCL failures in containers but not
one here.

## Verifying a change to any of this

Cheapest first. Each step rules out a different layer, so a failure tells you where to look.

```bash
# 1. Config, paths, dataset, and NCCL init -- no training, no sampling. ~2 minutes.
torchrun --standalone --nproc_per_node 2 -m fastvideo.train.entrypoint.train \
    --config examples/train/scenario/h3_proxy/proxy_bd_finetune_abot.yaml \
    --training.distributed.num_gpus 2 --training.distributed.sp_size 2 \
    --training.distributed.hsdp_shard_dim 2 --dry-run

# 2. That the network is what you think it is.
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET torchrun --standalone --nproc_per_node 2 \
    -m fastvideo.train.entrypoint.train --config <same> --dry-run 2>&1 \
    | grep -E "Using network|NET/Plugin|libibverbs|NCCL WARN"
```

Step 2 must say `Using network gIB`. `Using network Socket` means the plugin failed and NCCL fell
back silently — the job will run, slowly, and the reason will be three lines above in the INFO log.

`--dry-run` parses the config and builds the model, dataloader and process groups, then exits. It
catches wrong paths, a mismatched mesh (`num_gpus != nnodes * nproc_per_node`, which
`init_device_mesh` rejects) and every NCCL problem on this page, for two minutes instead of the
twenty a real launch spends loading the text encoder first.

## Sizing the mesh

`nnodes * nproc_per_node == num_gpus == hsdp_replicate_dim * hsdp_shard_dim`. These are not
independent knobs; `init_device_mesh` rejects a mesh larger than the world size, and a mesh smaller
than it leaves GPUs idle without complaint.

Global batch is `(num_gpus / sp_size) * train_batch_size * gradient_accumulation_steps`, so
doubling the node count at fixed global batch means halving `gradient_accumulation_steps` — not
touching the learning rate.

| | 1 node | 2 nodes |
| --- | --- | --- |
| `num_gpus` | 8 | 16 |
| `hsdp_shard_dim` | 8 | 16 |
| `sp_size` | 8 | 8 |
| `gradient_accumulation_steps` | 8 | 4 |
| global batch | 8 | 8 |
