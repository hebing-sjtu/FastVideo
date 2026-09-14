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

| Cache | Clips | Target | Proxy | Anchor | Notes |
| --- | --- | --- | --- | --- | --- |
| `cache/abot_train` | 9865 | 768 x 1344 | 192 x 336 | **768** | v1 window-scoped prompts; anchor off-spec |
| `cache/gta_train` | 972 | 768 x 1344 | **704 x 1280** | **704** | v1 GTA; proxy and anchor both off-spec |
| `cache/gta_v2_train` | 762 / 771 | 768 x 1344 | 192 x 336 | 2048 | pre-CWM depth range and palette |
| `cache/gta_v2_cwm` | 771 | 704 x 1280 | 192 x 336 | 2048 | CWM: 0.3–256 m, injective 11-class |

`gta_v2_train` is what run `gta_v2_b32_sp1` trained on, so its eval jobs need `768 / 1344 / 2048`.
It holds 762 clips against a 771-row manifest: nine clips were never encoded, and since the loader
scans the directory rather than the manifest, that never surfaced as an error.

`cache/abot_train` has no `cwm_system` in `info`, which dates it to before the encoder recorded that
field. A missing key there is a useful staleness signal: it means the cache predates whatever the
encoder has learned since.

Both v1 caches sit below a 2048 anchor, and the encoder's own note records that the run which cut
the anchor to 768 could not make the proxy steer the camera. Treat either as a separate experiment
rather than a baseline, and see the mismatch below before reading anything into their validation.

## The validation callback re-derives conditioning, so four values must agree

Training reads the anchor and proxy latents straight out of the cache. Validation does not: it
rebuilds both from the source media, using its own settings. Four of them therefore have to repeat
what the encoder was told, and none of them is checked against the cache.

| `callbacks.validation.*` | Must equal |
| --- | --- |
| `anchor_short_edge` | the encoder's `--anchor-short-edge` |
| `proxy_height` / `proxy_width` | the encoder's `--proxy-height` / `--proxy-width` |
| `cwm_system_prompt` | the encoder's `--cwm-system` |
| `lock_first_frame` | `models.student.lock_first_frame` |

The callback's defaults are the released values — 2048, 192 x 336, `w0`, locked — so a cache built at
the defaults needs no overrides at all. The failure mode is the reverse: overriding one of these to
chase a cache that was itself built off-spec, or leaving a default in place against a cache that was
not.

An anchor mismatch is the expensive one, because it looks like a bad checkpoint. The anchor's canvas
decides how many vision tokens it occupies, so validating a LoRA trained on a 768 anchor against a
2048 one presents a token grid the model never saw. That is the configuration `proxy_bd_finetune_abot.yaml`
shipped with: `data_path` pointed at a cache encoded with a 768 anchor while the callback rendered
2048. Any conclusion drawn from that run's validation — including "the proxy does not steer the
camera" — is about the mismatch until it is re-run with the two in agreement.

A proxy mismatch is worse in kind though easier to spot. Left unpinned, a video reference resolves
its canvas from its aspect ratio, which puts a 336 x 192 proxy onto the full 1344 x 768 canvas:
37296 reference rows where training used 2442, from LANCZOS-upsampled frames. On a DUV that
upsampling also averages unrelated depth codes into values no backend ever predicted.

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
