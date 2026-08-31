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

**The camera is a per-token constraint**, not content. Token `(t, h, w)` must show whatever the world
puts along one specific ray, and the binding has to be tight enough that the same proxy under two
trajectories yields two different videos. A reference sitting in the prefix is read once for the
whole clip and cannot do that. So the trajectory becomes a dense Plücker ray field on the target
latent grid and enters through a ControlNet that adds a residual at the exact token each ray belongs
to.

Everything else follows from that split:

- The control trunk is **zero-initialised at `proj_out`**, so an untrained branch is exactly a no-op
  and step 0 reproduces the released model bit-for-bit.
- The **backbone is frozen** by default. Only `camera_controlnet.*` trains, which is also why a
  checkpoint from this stage contains the branch alone.
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

One `.pt` per clip, written by the encoder:

```bash
python scripts/h3_proxy/prepare_data/encode_proxy_samples.py \
    --manifest data/h3_proxy/manifest.jsonl \
    --root data/h3_proxy/raw \
    --output /data/raw/h3_proxy/train \
    --model-path /workspace/models/MiniMax-H3
```

For a dataset laid out as flat `seg_*/` directories holding `video_src.mp4`, `video_target.mp4` and
`prompt.txt`, with the train/val split in `manifests/*_{train,val}.jsonl`, build that manifest with:

```bash
python scripts/h3_proxy/prepare_data/seg_dir_to_encode_manifest.py \
    --root /data/tmp --split train --out /workspace/h3_train.jsonl
```

It also reports the largest `--num-frames` the clips support. That is worth reading before
launching the full set: the encoder resamples to 24 fps *before* trimming, so a 124-frame 30 fps
clip contributes 99 frames and would fail `--num-frames 124` — for every clip, after the text
encoder has loaded.

Each manifest line names a target clip, a proxy, an optional anchor frame, an optional camera
trajectory, and a prompt. The proxy may be either RGB video or a DUV directory — depth as
`.depth.f32` plus semantic ID as `.semantic_id.png` per frame, which
`fastvideo/pipelines/basic/minimax_h3/proxy.py` packs into the 3-channel image the VAE encodes.
Depth is log-normalised over 0.3 m to 256 m and the semantic ID is split across the two chroma
channels, which is what lets one RGB VAE carry both.

The proxy is encoded at `--proxy-height 192 --proxy-width 336` regardless of the resolution the
render was supplied at: a quarter of the target's edge length, a sixteenth of its area, a sixteenth
of its tokens. A proxy carries layout and motion, and both survive downsampling in a way appearance
would not — which is also why the appearance comes from the anchor instead.

Trajectories are `.npz` files with `extrinsics` `[F, 4, 4]` world-to-camera, `intrinsics` `[F, 3, 3]`
in pixel units, and optionally `pixel_size` naming the resolution the intrinsics were measured at.
They are normalised at build time — rebased onto frame 0, recentred, rescaled — so the model never
sees the dataset's world origin or unit scale.

Cache to local disk, not to the GCS FUSE mount. Writing at speed through `gcsfuse` is what produced
the earlier `Errno 107` and `SIGBUS` failures; write to `/workspace` or `/tmp` and copy afterwards.

## Training

Two configs, in this order:

| Config | What trains | Needs camera poses |
| --- | --- | --- |
| `proxy_bd_finetune.yaml` | rank-128 LoRA on all 50 blocks (~665M params) | no |
| `proxy_camera_finetune.yaml` | only `camera_controlnet.*`, backbone frozen | yes |

Run the first on its own if the dataset has no trajectories; that is the entire usable stage in that
case, because a ControlNet with no input to read cannot train. Run the second on top of the first
once trajectories exist, pointing `init_from` at the stage-1 output.

H3 has no AR or DMD stage. It is bidirectional over a fully packed sequence, so "the BD stage" is
the only stage it has — the three-stage BD/AR/DMD ladder belongs to the causal Wan path in
`examples/train/scenario/game_v2v_depth/`.

```bash
torchrun --nnodes 2 --nproc_per_node 8 -m fastvideo.train.entrypoint.train \
    --config examples/train/scenario/h3_proxy/proxy_bd_finetune.yaml
```

On a single node, `unset PET_NNODES` first and add `--standalone`; the platform injects multi-node
rendezvous variables even for single-node jobs and `torchrun` will otherwise wait for a second node
that never arrives.

`train_batch_size` must stay 1 and `training_cfg_rate` must stay 0. Packed row indices describe one
document with no batch offset, and H3 has no zero-embedding branch for text CFG. Drop conditioning
with `camera_dropout` instead, which removes the branch the model can actually be asked to do
without — and which is what makes camera guidance available at sampling time.

`controlnet_num_heads` must divide `sp_size`: the trunk shards its own heads across the sequence
parallel group through the same Ulysses all-to-all the backbone uses.

### LoRA

Stage 1 adapts the backbone with LoRA rather than a full finetune, following CWM. Two things about
how it interacts with this plugin:

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
- **Depth on the trunk.** `enable_control_depth` exists and works, but is off by default: the proxy
  already carries geometry through the reference slot, so a second copy of it on the trunk mostly
  costs memory. It is also incompatible with `camera_dropout`, which drops the whole trunk and would
  therefore drop depth along with the camera.
