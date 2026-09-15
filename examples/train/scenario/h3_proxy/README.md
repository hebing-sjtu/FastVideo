# MiniMax-H3 proxy-to-video with camera control

Fine-tunes MiniMax-H3 into a renderer: in goes a cheap proxy render of a scene plus the camera
trajectory it was rendered under, out comes a photoreal video of the same scene under the same
motion.

## Why two different conditioning routes

The proxy and the camera are different kinds of signal, so they enter the model differently.

**The proxy is content**, and H3 already knows how to read content it is shown. The released Ref2VA
checkpoint packs ordered references as prefix rows of the video stream — their own resolution, their
own rotary coordinates, held at a near-clean timestep while the target denoises. Putting the proxy
there needs no architectural change at all. A single RGB anchor frame goes in the slot ahead of it to
fix appearance, which a depth/semantic proxy by construction cannot supply.

**Some leading target latent frames are a given** (`num_given_latent_frames`, 1 by default). The
prefix is read once for the whole clip and sits at its own rotary coordinates, so it says what the
scene looks like but neither where the camera starts nor how fast anything moves — which leaves the
proxy constraining motion relative to an initial pose the model invents, at a rate it also invents.
Handing it real target frames at the target's own coordinates says both. Given rows are held at the
reference prefix's noise amount, left out of the loss, and left out of every scheduler step at
sampling time — the same mask CWM applies as `video_mask[:, :, :VIDEO_PREFIX_LATENTS] = False`.

The count has to match the CWM system prompt the cache's text was wrapped in, because that prompt
states the contract the model is held to. Both the trainer and the validation callback raise rather
than let the two drift apart.

| `num_given_latent_frames` | `--cwm-system` | what the prompt promises | where the pixels come from |
| --- | --- | --- | --- |
| 1 | `w0` | *"this clip is the very beginning of the take"*, with *"the first frame of the target locked to this exact image … camera framing and layout"* | the anchor, re-encoded onto the *target* canvas since the row it lands in is a target row — what the released inference cache does with its `input_video.safetensors` |
| 10 | `wn` | *"The first 34 frames (1.4167 seconds) of this clip are ALREADY GIVEN … continue the video seamlessly: the same ongoing time, positions, poses, action phase and camera simply carry forward"* | real footage; validation takes the record's target clip, as CWM's windows past the first take the previous window's decoded output |

10 is CWM's own `VIDEO_PREFIX_LATENTS`, the count it pairs with those 34 frames. Real footage at the
target's coordinates establishes the motion rate directly, which leaves the proxy steering rather
than also having to set the clock — the layout gives it no frame-to-frame registration with the
target to set it from. The prefix is encoded as a whole clip and sliced in the latent space: the
VAE's group structure only admits 2, 7, 12, … latent frames, so 10 is not something a standalone
34-frame encode can produce, and slicing a full encode is exactly what training does to the cached
target latents.

**The camera is a per-token constraint**, not content. Token `(t, h, w)` must show whatever the world
puts along one specific ray, and the binding has to be tight enough that the same proxy under two
trajectories yields two different videos. A reference sitting in the prefix is read once for the
whole clip and cannot do that. So the trajectory becomes a dense Plücker ray field on the target
latent grid and enters through a ControlNet that adds a residual at the exact token each ray belongs
to.

**The proxy can take the second route too**, and `proxy_controlnet_finetune.yaml` is that variant.
The argument against the reference slot applies to the proxy as much as to a trajectory: H3 packs
references *ahead* of the target, so the proxy sits a whole clip earlier in rotary time — 206.667
units at 124 frames — and no proxy frame ever shares a temporal position with the target frame it is
meant to steer. That is enough for layout and rough heading and not for rate, which is what the LoRA
runs on that pathway showed: predictions that turn the right way at the wrong speed, with coherent
drift down to the random-walk floor by step 150.

So the trunk reads the proxy a second time through the `proxy` modality, on the target's own latent
grid. Either modality is a valid trunk on its own — `enable_control_camera: false` gives a proxy
ControlNet, which is what footage without a captured trajectory gets.

The reference slot keeps the proxy anyway. The cached text embedding was tokenized around `<Picture
1>` then `<Video 1>`, and the CWM system prompt spends most of its length saying what `<Video 1>` is
for; dropping it would invalidate every cached text embedding and contradict a prompt this repo
otherwise reproduces byte-for-byte. The proxy is therefore presented twice — once as content the text
can refer to, once as a constraint the trunk can enforce.

**Nothing extra is cached for it.** The trunk's copy is the same `proxy_latent`, replicated cell by
cell onto the target grid: the VAE's stride of 16 means proxy latent cell `(i, j)` covers the same
pixels as target cells `(4i..4i+3, 4j..4j+3)`, so copying each cell over that block lands it exactly
where the pixels it describes ended up. Registration is exact, not approximate.

Replicating latents rather than re-encoding blown-up pixels is deliberate, and not only cheaper. The
two are not the same tensor — the VAE is not linear — but they differ only in that encoding blown-up
pixels also encodes the block edges the blow-up introduced, which is an artifact rather than scene
detail. Neither route adds detail the reference did not have; what either adds is a *position* at
which the signal is allowed to act. And this tensor is read by a freshly initialised linear and never
decoded, so there is no reason for it to look like something the VAE would have produced.

What does have to hold is that the **target latent grid is an integer multiple of the proxy's**;
otherwise one proxy cell spreads over a fractional number of target cells and the registration
varies across the frame. The released geometries divide exactly (48/12 == 84/21 == 4). 704x1280 is
44 x 80 latents, which is 3.67x by 3.81x — so this variant cannot inherit the canvas `gta_v2_cwm` was
written at, and the deviation from CWM's released canvas those runs carried is the same thing that
blocks replication. One plain re-encode clears both.

Everything else follows from that split:

- The control trunk is **zero-initialised at `proj_out`**, so an untrained branch is exactly a no-op
  and step 0 reproduces the released model bit-for-bit. This is what makes it readable against a
  LoRA baseline: anything a panel shows after step 0 is the trunk's doing.
- The **backbone is frozen** by default. Only `camera_controlnet.*` trains, which is also why a
  checkpoint from this stage contains the branch alone. Training the trunk and a LoRA together is
  possible (`freeze_backbone: false` plus a `lora` block) but gives up that attribution.
- The trunk **mirrors the packed row layout** rather than living over the video rows alone, so the
  two streams shard identically under sequence parallelism and the residual add stays local. The
  residual is masked to the target video rows; references and audio rows are left alone.
- The control blocks **reuse the backbone's rotary table**, which is why `controlnet_dim /
  controlnet_num_heads` has to be at least the `2 * 3 * rope_freq_dim` channels it rotates (96 for
  the release; 1024/8 = 128 clears it).

## Checkpoint

Start from `MiniMaxAI/MiniMax-H3` and make sure the snapshot includes `transformer_ref/`, not just
`transformer/`. The release ships two transformer partitions and this path loads the Ref2VA one:
`transformer/` is the T2VA model and has never been trained to read reference rows, so starting
there would throw away the exact capability the proxy conditioning relies on. A snapshot missing
`transformer_ref/` fails at load rather than silently training the wrong weights.

Do not start from a rank-reduced checkpoint such as `noctuashap/MiniMax-H3-pruned-r16`. Those
factorize AdaLN and pin it to FP16, which the trainer rejects outright — they are inference
artifacts.

## Data

[CACHE_GEOMETRY.md](CACHE_GEOMETRY.md) records the `.pt` payload, the three geometries frozen at
encode time and the consumer flags that must match them, plus the encode failures that report
success. `scripts/h3_proxy/describe_cache.py` reads all of it back off an existing cache.

One `.pt` per clip, written by the encoder:

```bash
python scripts/h3_proxy/prepare_data/encode_proxy_samples.py \
    --manifest data/h3_proxy/manifest.jsonl \
    --root data/h3_proxy/raw \
    --output /data/raw/h3_proxy/train \
    --model-path /data/models/MiniMax-H3
```

That is one process on one GPU. For a set of any size, fan it across the node instead — the wrapper
takes the same arguments and adds `--shard-index` / `--num-shards` per process:

```bash
scripts/h3_proxy/prepare_data/encode_proxy_shards.sh \
    --manifest /data/binghe/h3_proxy/abot_train.jsonl \
    --root /data/binghe/datasets/ABot-sub-2000-clips \
    --output /data/binghe/h3_proxy/cache/abot_train \
    --model-path /data/models/MiniMax-H3 \
    --anchor-short-edge 2048 --proxy-height 192 --proxy-width 336
```

Shard `i` takes `entries[i::n]` and the encoder skips a clip whose `.pt` is already there, so
re-running the identical command retries only what is missing — which is also how a shard that hit
an OOM is recovered. The wrapper waits for every shard even after one fails, prints the tail of
each failing log, and finishes by comparing the cache against the manifest row count.

Starts are staggered (`STAGGER_SEC`, default 45) because each shard deserialises its own copy of a
~64 GB bf16 text encoder. Eight simultaneous starts is one ~500 GB read burst and a host-RAM spike;
the GPUs are idle through that window regardless, so the stagger costs nothing real.

For a dataset laid out as flat `seg_*/` directories holding `video_src.mp4`, `video_target.mp4` and
`prompt.txt`, with the train/val split in `manifests/*_{train,val}.jsonl`, build that manifest with:

```bash
python scripts/h3_proxy/prepare_data/seg_dir_to_encode_manifest.py \
    --root /data/tmp --split train --out /data/binghe/h3_proxy/h3_train.jsonl
```

It also reports the largest `--num-frames` the clips support. That is worth reading before
launching the full set: the encoder resamples to 24 fps *before* trimming, so a 124-frame 30 fps
clip contributes 99 frames and would fail `--num-frames 124` — for every clip, after the text
encoder has loaded.

Each manifest line names a target clip, a proxy, an optional anchor frame, an optional camera
trajectory, and a prompt. The proxy comes in one of three forms, exactly one per line:

| Key | What it is |
| --- | --- |
| `proxy` | An ordinary RGB render, resized to the proxy grid and used as-is |
| `proxy_duv` | A directory of per-frame `NNNNNN.depth.f32` and `NNNNNN.semantic_id.png` |
| `proxy_duv_video` | A DUV some upstream pipeline already composed into a lossless video |

For `proxy_duv`, `fastvideo/pipelines/basic/minimax_h3/proxy.py` packs the pair into the 3-channel
image the VAE encodes: depth log-normalised over 0.3 m to 256 m, and the semantic ID split across
the two chroma channels, which is what lets one RGB VAE carry both.

`proxy_duv_video` skips that packing, so the channel convention is the producer's rather than this
repo's. That is workable because this stage starts from the base Ref2VA checkpoint, which has no DUV
prior of any kind — what the three channels mean is learned here either way. The requirement is only
that sampling presents the same convention the cache was written with, which is why nothing rewrites
the pixels on the way in and why validation reads the same files. It is also why a set must not mix
conventions: a depth channel that means near-is-bright on some clips and near-is-dark on others is
unreadable, and no shape, count or loss check would show it.

The proxy is encoded at `--proxy-height 192 --proxy-width 336` regardless of the resolution the
render was supplied at: a quarter of the target's edge length, a sixteenth of its area, a sixteenth
of its tokens. A proxy carries layout and motion, and both survive downsampling in a way appearance
would not — which is also why the appearance comes from the anchor instead.

`proxy_duv_video` is the one proxy form that is never resized. A DUV frame holds integer codes
wearing an RGB costume, so interpolating it averages unrelated depths and paints class boundaries a
code no backend predicted, while producing a perfectly plausible-looking image. A grid mismatch is
therefore an error rather than something to resample away.

### A clip-per-directory dataset

For a dataset that ships one already-windowed clip per directory — `target/rgb.mp4`,
`target/anchor.png`, `proxy/duv.mp4` and a `clip_report.json` — build the manifest with:

```bash
python scripts/h3_proxy/prepare_data/clip_dir_to_encode_manifest.py \
    --root /data/binghe/datasets/ABot-sub-2000-clips \
    --split train --val-episodes 24 --out /data/binghe/h3_proxy/abot_train.jsonl
```

It splits **episodes**, not clips. Five windows cut from one 60-second episode share weather,
lighting and terrain, so splitting by clip validates on footage already trained on and reports a
loss that looks much better than the model is. Train and val are complements of one another for a
given `--val-episodes`, so there is no split file to keep in sync — but the two scripts have to be
passed the same number.

It also refuses to write a manifest for a set that would fail late: mixed DUV conventions, clips
whose `deliverable` flag is false, and — on a machine that has the decoder — a sampled palette check
that catches a DUV arriving in the wrong channel order or through a lossy re-encode. All three are
invisible downstream. The cache would be written, training would converge, and the proxy would be
describing noise.

**Prompts have to be scoped to the window.** Text comes from `<clip>/prompt.txt`, which should hold
a CWM window sentence — `[0.00s-5.17s] ` followed by prose about those 5.17 seconds and nothing
else. The datapipe's `captions-export --write-txt` writes them. A clip without one is now rejected;
`annotations/caption.json` is episode-level and only used if you pass `--allow-episode-caption`,
which describes the whole 60 seconds, names things the clip never shows, and teaches the model to
cover a minute of story in five seconds. Both this script and the validation-JSON builder print a
`window-scoped prompts: N/N` line and warn with an example when the count falls short — check it
before spending an encode, because text scope is invisible in every downstream shape check.

Trajectories are `.npz` files with `extrinsics` `[F, 4, 4]` world-to-camera, `intrinsics` `[F, 3, 3]`
in pixel units, and optionally `pixel_size` naming the resolution the intrinsics were measured at.
They are normalised at build time — rebased onto frame 0, recentred, rescaled — so the model never
sees the dataset's world origin or unit scale.

Write the cache to the persistent data disk. The `gcsfuse` faults that once forced a staged
write — `Errno 107` and `SIGBUS` under sustained throughput — are fixed, so the detour through
`/workspace` followed by a copy is no longer worth its cost: at ~15 MB per clip a 10k-clip set is a
~150 GB round trip, and anything left under `/workspace` dies with the node.

That applies to `training.checkpoint.output_dir` too, and it has a second reason. The path must be
absolute: a relative one resolves against the launch directory, which is the checked-out repo. Both
the checkpoints and the validation mp4s the callback writes live under it, so a relative path puts
every artifact a run produces on the ephemeral disk. Only the model snapshot belongs on
`/workspace`, since it can be downloaded again.

## Training

Two configs, in this order:

| Config | What trains | Needs camera poses |
| --- | --- | --- |
| `proxy_bd_finetune.yaml` | rank-128 LoRA on all 50 blocks' attention (~321M params) | no |
| `proxy_bd_finetune_abot.yaml` | the same stage, wired to a `clip_*/` dataset | no |
| `proxy_camera_finetune.yaml` | only `camera_controlnet.*`, backbone frozen | yes |

Run the first on its own if the dataset has no trajectories; that is the entire usable stage in that
case, because a ControlNet with no input to read cannot train. Run the second on top of the first
once trajectories exist, pointing `init_from` at the stage-1 output.

H3 has no AR or DMD stage. It is bidirectional over a fully packed sequence, so "the BD stage" is
the only stage it has — the three-stage BD/AR/DMD ladder belongs to the causal Wan path in
`examples/train/scenario/game_v2v_depth/`.

```bash
source /usr/local/gib/scripts/set_nccl_env.sh
export LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH}
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false

torchrun --nnodes 2 --nproc_per_node 8 -m fastvideo.train.entrypoint.train \
    --config examples/train/scenario/h3_proxy/proxy_bd_finetune.yaml \
    --training.distributed.num_gpus 16 \
    --training.distributed.hsdp_shard_dim 16 \
    --training.loop.gradient_accumulation_steps 4
```

On a Google Cloud A3-Ultra or A4 node that environment is the whole of it, but it has a
prerequisite the image does not satisfy: the gIB plugin `dlopen`s `libibverbs.so.1`, which is not
installed, and `set_nccl_env.sh` forces `NCCL_NET=gIB` — so a forced network with no usable devices
takes down `ncclCommInitRank` with `invalid usage` or `internal error`, on one node as readily as on
two. `NODE_ENVIRONMENT.md` at the repo root has the one-line fix, the log lines that identify it,
and three plausible-looking workarounds that are wrong — including the `NCCL_P2P_DISABLE=1` that
older launchers under `examples/` carry, which disables NVLink and makes things worse.

Do not skip `--dry-run` on two GPUs before a 16-rank launch. It builds the config, dataloader and
process groups and then exits, so it catches a wrong path, a mesh that disagrees with torchrun's
process count, and every NCCL problem above — in two minutes rather than after a 32B text encoder
has loaded.

On a single node, `unset PET_NNODES` first and add `--standalone`; the platform injects multi-node
rendezvous variables even for single-node jobs and `torchrun` will otherwise wait for a second node
that never arrives.

Check the GPUs are actually free before launching. The encode step runs one process per GPU, each
holding a ~64 GB text encoder, and a detached shard that outlived its shell leaves too little memory
for NCCL to build its buffers — which also surfaces as an unhelpful init error rather than an OOM.

`train_batch_size` must stay 1 and `training_cfg_rate` must stay 0. Packed row indices describe one
document with no batch offset, and H3 has no zero-embedding branch for text CFG. Drop conditioning
with `camera_dropout` instead, which removes the branch the model can actually be asked to do
without — and which is what makes camera guidance available at sampling time.

`controlnet_num_heads` must divide `sp_size`: the trunk shards its own heads across the sequence
parallel group through the same Ulysses all-to-all the backbone uses.

### LoRA

Stage 1 adapts the backbone with LoRA rather than a full finetune, following CWM. The module set is
the 4 attention projections across all 50 blocks, which is 200 adapted modules — the count
`cwm_h3_inference/constants.py` pins as `EXPECTED_LORA_MODULES` and `strict_merge_lora` refuses to
deviate from. That lands at ~321M trainable at rank 128. Adding the SwiGLU (`fc_in`/`fc_out`) would
make it 300 modules and a differently-shaped adapter than the released one.

Two things about how LoRA interacts with this plugin:

`freeze_backbone: true` and `lora.enable: true` are rejected together. LoRA is the mechanism by
which the backbone trains here, so freezing it leaves nothing to optimize — the combination almost
always means a config was half-edited.

LoRA and the control trunk can train jointly. `enable_lora_training` freezes the whole module before
inserting adapters, so `_restore_trainable_after_lora` re-enables `camera_controlnet.*` afterwards.
The trunk trains as full parameters, not adapters: it has no pretrained weights to adapt, and its
zero-initialised `proj_out` is what makes an untrained branch a no-op. That is also why
`camera_controlnet` sits in the arch config's `exclude_lora_layers`, alongside `token_refiner` and
`time_embedder` — all three share leaf names (`to_q`, `fc_in`, ...) with the 50 main blocks and would
otherwise be adapted by substring match.

## Validation

Proxy-to-video is judged on whether the output follows its proxy, which a prediction on its own
cannot show. `MiniMaxH3ProxyValidationCallback` therefore logs a `proxy | prediction | target` panel
per held-out clip, as one video the tracker keeps aligned across steps, rather than three artifacts
the viewer has to scrub in sync.

Build the dataset file from the same tree and the same held-out split the encode step used:

```bash
python scripts/h3_proxy/prepare_data/seg_dir_to_validation_json.py \
    --root /data/tmp --split val --limit 6 \
    --out /data/binghe/h3_proxy/validation_val6.json

# or, for a clip-per-directory dataset, with the same --val-episodes the encode manifest used
python scripts/h3_proxy/prepare_data/clip_dir_to_validation_json.py \
    --root /data/binghe/datasets/ABot-sub-2000-clips \
    --val-episodes 24 --limit 6 \
    --out /data/binghe/h3_proxy/abot_validation_val6.json
```

Paths in that file are absolute on purpose. `ValidationDataset` resolves only the media keys it
already knows about against the dataset directory, and the proxy and target arrive through keys it
does not know.

With `trackers: [wandb]` already set under `training.tracker`, each event logs two keys:
`validation_videos_50_steps` for the predictions alone and `validation_videos_50_steps_compare` for
the panels, both captioned with the prompt. MP4s also land in `training.checkpoint.output_dir`, so a
run whose W&B is offline still keeps them.

Three settings exist to keep validation comparable to training rather than to the released defaults:

- **`anchor_short_edge`** must equal the encoder's `--anchor-short-edge`. Both default to the
  released 2048, so the two agree unless you move one. An anchor's canvas decides how many vision
  tokens it occupies *and* how many Ref2VA reference rows it fills, so cutting it to 768 is a ~7x
  reduction on both streams; a mismatch between encoder and validation presents the model a token
  grid it never trained on and makes the checkpoint look worse for no reason of its own.
- **`proxy_height` / `proxy_width`** must equal the encoder's `--proxy-height` / `--proxy-width`,
  for the same reason and then some. A video reference otherwise resolves its canvas from its aspect
  ratio, which puts a 336x192 proxy on the full 1344x768 canvas: 37296 reference rows where training
  used 2442, from LANCZOS-upsampled frames. On a DUV that upsampling is not merely off-distribution
  — on a synthetic blocky frame it leaves the majority of pixels carrying a class code no backend
  predicted, and moves decoded depth by far more than the encoding's own quantisation step.
- **`use_validation_media_conditioning`** must stay false. H3 Ref2VA conditions on an ordered
  reference list, so `image_path` would present the proxy a second time; the callback rejects true
  rather than silently doubling it.

A record whose proxy or anchor cannot be read fails the run rather than sampling without it, because
a clip generated from the wrong conditioning is worse than no clip: it looks like a model result.
Keep `run_at_start: true` so that surfaces at step 0, before the run has spent anything, instead of
at the first scheduled event hours in.

Validation re-encodes text and references through the pipeline; it does not read the training
`.pt` caches. That is what makes it an end-to-end check, and also what makes it expensive: every
record is a full sampling trajectory at ~41.5k tokens, alongside a second text encoder and VAE. Six
clips every 250 steps is a few percent of wall clock, `every_steps: 20` would dominate the run.
`offload_training_state` and `unload_pipeline_after_validation` are both on for the same reason.

### Evaluating one checkpoint, without training

To sample a checkpoint and stop, use the wrapper rather than assembling the overrides:

```bash
scripts/h3_proxy/eval_checkpoint.sh \
    --step 150 --run /data/binghe/h3_proxy/runs/gta_v2_cwm/checkpoints
```

`--list` finds them first. Directory names drift, but every checkpoint records the cache it trained
on, and that is what separates one experiment from another:

```bash
scripts/h3_proxy/eval_checkpoint.sh --list                    # every run, its last loadable step
scripts/h3_proxy/eval_checkpoint.sh --list --run <dir>        # one run, every step, loadable or not
```

A checkpoint a live run is still writing shows as `incomplete`; resuming it raises rather than
loading half a model.

`CheckpointManager._write_metadata` stores the whole training config in the checkpoint's
`metadata.json`, so the config, the cache and the validation set are read back from the checkpoint
instead of being named again. That is not just brevity: evaluating against the run's own config
makes every setting that shapes a state-dict key agree *by construction* rather than because
someone picked the same file. `--config`, `--cache` and `--val-json` still override it, and
`--step 0` requires the last two, being the one case with no checkpoint to read them from.

Anything after a bare `--` is passed through to the trainer, last on the launch line, and recorded
in `eval_manifest.json` — because it changes what was sampled, and a wn baseline and a w0 baseline
are both "step 0" without being the same picture. This is how a regime the config does not describe
gets sampled; `--step 0`'s default config is w0, so a wn cache needs it:

```bash
scripts/h3_proxy/eval_checkpoint.sh --step 0 \
    --cache /data/binghe/h3_proxy/cache/gta_v2_cwm_wn \
    --val-json /data/binghe/h3_proxy/gta_v2_validation.json \
    -- --models.student.num_given_latent_frames 10 \
       --callbacks.validation.num_given_latent_frames 10 \
       --callbacks.validation.cwm_system_prompt wn
```

Forgetting them used to sample silently: the wn prompt promises the model 34 given frames while the
rows supply one. Both the callback and the trainer now refuse that pairing — the trainer against
`info["cwm_system"]` in the cache itself, which is the only record of what the text was wrapped in
and cannot be wrong in the same direction as the config.

The proof that the weights arrived is the resume's own log line, before any sampling is paid for:

```
Loaded 400/400 requested model tensors (400 supplied, 0 unmatched); lora_B norm 0 -> 0.597282
```

`lora_B` is zero-initialised, so that norm leaving zero is the single fact that distinguishes "this
checkpoint's weights are in the model" from "this is the base model". A nonzero count of `lora_B`
tensors whose norm is still exactly zero after a load raises instead of sampling.

Four settings have to agree for that to mean what it says, and each one fails quietly on its own:

- **`resume_from_checkpoint`** is the dangerous one. Omit it and the job does not complain — it
  starts at step 0, and because a LoRA is zero-initialised it samples the *base model*, writes the
  videos, and exits successfully. Compare those against a real step-0 baseline and they are
  identical, which reads as "training changed nothing" rather than "the checkpoint never loaded".
  The only evidence is the step in the filename, which you read after paying for the sampling. An
  explicit path that does not exist does raise, so this is specifically the risk of *leaving the
  flag out* — which is exactly what happens when a step-0 command gets reused as a template.
  `Trainer.run` now logs which of the two cases it is at the moment it decides.
- **`max_train_steps`** must equal the resumed step, or the loop is not empty and the job trains on.
- **`every_steps`** must divide the resumed step, or `on_validation_begin` returns early and the job
  finishes having logged nothing at all.
- **the geometry** must match the cache, per the three settings above. The wrapper reads it from the
  cache with `describe_cache.py --emit-flags`, which refuses to print a partial or mixed answer, so
  it cannot drift from what was encoded.
- **the LoRA rank, `target_modules` and `enable_gradient_checkpointing_type`** must match the run
  that wrote the checkpoint, because all three rename or reshape every saved key and DCP matches by
  name under `strict=False`. The last one is the easiest to miss: it wraps each block in a module
  that inserts a `.checkpointed.` segment, so enabling it on one side only matches nothing at all.
  `probe_resume.py` diffs these against the checkpoint's saved config with no GPU and no model
  build, and the wrapper runs it as a preflight.

`--step 0` is the base-model baseline and is the one case that legitimately has no checkpoint. The
wrapper also writes to a fresh `output_dir` by default, since a training directory prunes by highest
step and would delete the early checkpoints the eval exists to inspect.

## Sampling

`MiniMaxH3ProxyCameraPipeline` is the Ref2VA pipeline plus one stage that turns a requested
trajectory into control rows. Pass the anchor and proxy as ordered references exactly as any Ref2VA
request does, and the trajectory as `minimax_h3_camera` in the request's extras — either a path to
the same `.npz` format the encoder reads, or a mapping with `extrinsics` and `intrinsics` arrays.

Omitting the trajectory samples with the branch inert rather than failing, which is the
unconditional side of camera guidance.

## What is not here

- **Streaming.** H3 is bidirectional over a fully packed sequence, so this path renders whole clips.
  The causal Wan pipeline in `examples/train/scenario/game_v2v_depth/` remains the streaming route.
- **Audio.** `supervise_audio` is off. The packed layout still requires audio rows, so silent
  footage gets zero latents, and returning a video-only prediction is what tells the finetune loss
  to leave the audio head alone rather than train it towards placeholder silence.
- **A captured camera trajectory**, for the datasets here. ABot ships sparse non-metric COLMAP poses
  and the GTA capture ships none, so `enable_control_camera` has nothing to build a ray field from
  and the trunk runs on the proxy alone. A trajectory would be the more direct fix for camera
  control than a proxy is, since it states the pose instead of implying it.
- **The trunk together with `camera_dropout`.** Dropout drops the whole trunk. With the proxy on it
  that is the proxy's only per-token route, so dropping it would train the model to ignore the thing
  the variant exists to test; the plugin rejects the combination rather than letting it through.
