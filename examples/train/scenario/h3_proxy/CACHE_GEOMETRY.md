# The proxy cache format, and the flags that must match it

A proxy cache is a directory of `<name>.pt` files written once by
`scripts/h3_proxy/prepare_data/encode_proxy_samples.py`. Three geometries are frozen into those
files at that moment, and nothing downstream re-derives any of them. Train and sample time only
*assert* them, so every consumer has to be told the same numbers the encoder was told.

This exists because those numbers live in a shell history that does not survive the week. Read them
off the cache instead:

```bash
python scripts/h3_proxy/describe_cache.py /data/binghe/h3_proxy/cache/gta_v2_train \
    --manifest /data/binghe/h3_proxy/gta_v2_train.jsonl
```

That prints the geometry, flags a cache assembled by two different commands, reconciles the clip
names against the manifest, and emits both the flags needed to consume the cache and the flags
needed to rebuild it.

## The three geometries

| What | Encoder flag | Consumer | Released value |
| --- | --- | --- | --- |
| Target canvas | `--height` / `--width` | `training.data.num_height` / `num_width` | 768 x 1344 |
| Proxy grid | `--proxy-height` / `--proxy-width` | none; baked into `proxy_latent` | 192 x 336 |
| Anchor short edge | `--anchor-short-edge` | `callbacks.validation.anchor_short_edge` | 2048 |

Both shipped YAMLs already set `anchor_short_edge: 2048`, so the usual mistake is not forgetting it
but *overriding* it on the command line to match a cache that was itself built wrong.

The proxy grid has no consumer flag, which is why it is the easiest of the three to get wrong
without noticing: a cache encoded at the wrong proxy resolution trains and validates without
complaint, and only shows up as a proxy that cannot steer the camera.

## Reading geometry off the shapes

Every latent axis is its pixel axis over **16** — the VAE's 8x spatial compression times the
transformer's 2x2 patch. So the shapes are the geometry, and a cache is self-documenting:

```
vae_latent    (24, 37, 48, 84)    ->  768 x 1344 target
proxy_latent  (24, 37, 12, 21)    ->  192 x 336 proxy
anchor_latent (24,  1, 128, 224)  ->  2048 x 3584, short edge 2048
```

`num_frames 124` gives 37 latent frames. The encoder enforces `num_frames % 17 == 5` for the causal
VAE and rejects anything else up front.

The anchor keeps its source image's aspect at a fixed short edge, so its long edge differs from clip
to clip — 224 and 232 in the same cache is normal. Only the short edge has to be single-valued.

## What a `.pt` holds

| Key | Shape | Notes |
| --- | --- | --- |
| `vae_latent` | `(24, T, h, w)` | the denoising target |
| `proxy_latent` | `(24, T, h, w)` | the conditioning render, DUV or RGB |
| `anchor_latent` | `(24, 1, h, w)` | appearance dictionary for the whole take, not a first frame |
| `text_embedding` | `(tokens, 5120)` | Qwen3-VL, already wrapped in the CWM chat |
| `text_token_tags` | `(tokens,)` | per-token modality tags |
| `info` | dict | `num_frames`, `pixel_size`, `prompt`, `cwm_system` |
| `extrinsics` / `intrinsics` | `(F, 4, 4)` / `(F, 3, 3)` | only when training the camera ControlNet |

`info["pixel_size"]` is the one geometry the encoder records explicitly. The proxy grid and anchor
short edge are not recorded and have to be recovered from the shapes.

Because the text embedding is baked in, **changing the prompt text or the chat wrap requires
re-encoding the text rows**. `--text-only` rewrites `text_embedding` and `text_token_tags` on
existing `.pt` files without touching the latents or loading the VAE.

## Caches built so far

Confirmed by reading the files; re-confirm with `describe_cache.py` rather than trusting this table.

| Cache | Clips | Target | Proxy | Anchor | DUV convention |
| --- | --- | --- | --- | --- | --- |
| `cache/abot_train` (v2 text) | 9545 | — | 192 x 336 | 2048 | predicted proxy, flickers |
| `cache/gta_train` (v1) | 972 | — | 704 x 1280 | 704 | pre-CWM; **both off-spec** |
| `cache/gta_v2_train` | 762 | 768 x 1344 | 192 x 336 | 2048 | pre-CWM depth range and palette |
| `cache/gta_v2_cwm` | 771 | 704 x 1280 | 192 x 336 | 2048 | CWM: 0.3–256 m, injective 11-class |

`gta_v2_train` is what run `gta_v2_b32_sp1` trained on, so its eval jobs need `768 / 1344 / 2048`.
It holds 762 clips against a 771-row manifest: nine clips were never encoded, and since the loader
scans the directory rather than the manifest, that never surfaced as an error.

The v1 GTA cache is the one that deviates on both the proxy grid and the anchor short edge. The
encoder's own note records that the run which cut the anchor to 768 could not make the proxy steer
the camera, so treat any cache below a 2048 anchor as a separate experiment rather than a baseline.

## Failures that do not announce themselves

**A sharded encode with the wrong `--num-shards`.** `entries[shard_index::num_shards]` means
`--num-shards 96 --shard-index 0` on a 771-row manifest encodes nine clips and reports
`9 written, 0 skipped, 0 failed`. Nothing is wrong from the script's point of view. The tell is the
progress denominator, `[1/9]`, which is the post-shard row count — and `0 skipped`, which rules out
the more natural suspicion that `--overwrite` was missing, since that path increments `skipped`.

Prefer `prepare_data/encode_proxy_shards.sh`. It defaults `--num-shards` to the GPU count and
forwards everything else untouched, so this failure cannot be expressed; it waits for every shard
even after one fails, tails each failing log, and ends by comparing the `.pt` count against the
manifest rows. The rest of this section describes failures it surfaces rather than prevents.

**An unset shell variable.** `--root $ROOT` with `ROOT` unset collapses to a bare `--root`, argparse
takes the following `--output` as its value, and all eight shards exit 2 before loading a model. An
`Exit 2` with nothing in the log above it is almost always this. The `preflight_media` check cannot
help, because it runs after argparse has already succeeded.

**Mixed conventions inside one set.** `proxy_duv_video` frames reach the VAE unchanged, so the
channel convention is the producer's. A depth channel meaning near-is-bright on some clips and
near-is-dark on others is unreadable, and no shape, count or loss check would show it. This is why
`read_duv_video_clip` has no resize branch and errors on a grid mismatch instead: a DUV frame holds
integer codes wearing an RGB costume, and interpolating it averages unrelated depths and paints
class boundaries a code no backend ever emitted, while producing a plausible-looking image.

**Comparing runs across caches.** `gta_v2_train` and `gta_v2_cwm` differ in target canvas *and* DUV
convention, so a LoRA drift-to-noise curve measured on one is not comparable to the other. When
changing cache, change one axis at a time or accept that the comparison is qualitative.
