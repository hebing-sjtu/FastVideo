"""Preference pairs for SolarWM's Stage0.5 Diffusion-DPO from multi-seed serve takes and their scores.

For every sample, the k-th best seed by ``--reward`` is paired with the k-th worst, kept only if the
reward gap is at least ``--min-gap``. A pair is the sample's cached ``.pt`` with ``vae_latent`` set
to the preferred take's latents and ``vae_latent_lose`` to the rejected one's; both keep the cached
given latents, so the two differ only where the model generated. Writes ``<out>/<sample>-pK.pt``,
the training manifest ``<out>/pairs.txt`` and ``<out>/pairs.jsonl`` with the choices.

    python dpo_pairs.py --scores scores.jsonl --spool /workspace/dpo0/spool --data train128.txt \\
      --out /workspace/dpo0/pairs --reward depth --pairs-per-sample 2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file


def choose(rows: list[dict], reward: str, per_sample: int, min_gap: float) -> list[tuple[dict, dict]]:
    ranked = sorted(rows, key=lambda row: row[reward], reverse=True)
    pairs = []
    for k in range(min(per_sample, len(ranked) // 2)):
        win, lose = ranked[k], ranked[-1 - k]
        if win[reward] - lose[reward] >= min_gap:
            pairs.append((win, lose))
    return pairs


def take_latents(spool: Path, job_id: str) -> torch.Tensor:
    latents = load_file(str(spool / "out" / job_id / "segments" / "seg-00" / "generated.safetensors"))["latents"]
    if latents.dim() != 5 or latents.shape[0] != 1:
        raise ValueError(f"{job_id}: expected [1, C, T, H, W] latents, got {tuple(latents.shape)}")
    return latents[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--spool", required=True)
    p.add_argument("--data", required=True, help="manifest of the cached samples the takes were made from")
    p.add_argument("--out", required=True)
    p.add_argument("--reward", default="depth")
    p.add_argument("--pairs-per-sample", type=int, default=1)
    p.add_argument("--min-gap", type=float, default=0.0)
    args = p.parse_args()

    samples = {Path(line.split()[0]).stem: Path(line.split()[0])
               for line in Path(args.data).read_text().splitlines() if line.strip()}
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for path in args.scores:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row.get("source") == "serve" and row.get("kind") != "pairwise" and row["sample"] in samples:
                by_sample[row["sample"]].append(row)

    out, spool = Path(args.out), Path(args.spool)
    out.mkdir(parents=True, exist_ok=True)
    manifest, records = [], []
    for sample, rows in sorted(by_sample.items()):
        pairs = choose(rows, args.reward, args.pairs_per_sample, args.min_gap)
        if not pairs:
            continue
        payload = torch.load(samples[sample], map_location="cpu", weights_only=True)
        cached = payload["vae_latent"]
        given = 10 if str(payload.get("info", {}).get("cwm_system", "")).startswith("wn") else 1
        for k, (win, lose) in enumerate(pairs):
            pair = dict(payload)
            for key, row in (("vae_latent", win), ("vae_latent_lose", lose)):
                latents = take_latents(spool, row["id"]).to(cached.dtype)
                if latents.shape != cached.shape:
                    raise ValueError(f"{row['id']}: latents {tuple(latents.shape)} != cached {tuple(cached.shape)}")
                latents[:, :given] = cached[:, :given]
                pair[key] = latents.contiguous()
            name = f"{sample}-p{k}"
            torch.save(pair, out / f"{name}.pt")
            manifest.append(str(out / f"{name}.pt"))
            records.append({"pair": name, "sample": sample, "reward": args.reward,
                            "win_seed": win["seed"], "lose_seed": lose["seed"],
                            "win_reward": win[args.reward], "lose_reward": lose[args.reward],
                            "win_psnr_late": win.get("psnr_late"), "lose_psnr_late": lose.get("psnr_late")})
    (out / "pairs.txt").write_text("".join(f"{path}\n" for path in manifest), encoding="utf-8")
    (out / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    psnr_gap = [r["win_psnr_late"] - r["lose_psnr_late"] for r in records if r["win_psnr_late"] is not None]
    print(f"[pairs] {len(records)} pairs from {len(by_sample)} samples -> {out / 'pairs.txt'}; "
          f"mean psnr_late gap {sum(psnr_gap) / max(1, len(psnr_gap)):+.3f} dB", flush=True)


if __name__ == "__main__":
    main()
