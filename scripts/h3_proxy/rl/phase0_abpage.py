"""Blind A/B rating page for the phase-0 clips, and the tally of its answers.

``build`` copies every ``phase0_pairs.py`` clip under a random name into the eval board's media
folder and writes one static page that mixes the rewards in random order. The page keeps the
rater's choices in the browser and exports them as JSON; the map from random names back to
clips stays next to the clips, outside the board.
``score`` turns an exported answer file into, per reward, how often the rater preferred the
clip that reward picked (ties count half), with a 95% Wilson interval.

    python phase0_abpage.py build --clips /workspace/rl0/abtest --board /workspace/eval-board --data <manifest>
    python phase0_abpage.py score --clips /workspace/rl0/abtest answers.json
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import secrets
import shutil
from pathlib import Path

PAGE = "rl0-ab.html"
MEDIA = "media/rl0ab"

TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>RL 第 0 步盲评</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px auto; max-width: 1500px; color: #1a202c; }}
.card {{ border: 1px solid #cbd5e0; border-radius: 8px; padding: 14px; margin: 18px 0; }}
.card.done {{ border-color: #38a169; background: #f0fff4; }}
video {{ width: 100%; background: #000; }}
.labels {{ display: grid; grid-template-columns: 1fr 1fr 1fr; text-align: center; font-weight: 600; color: #4a5568; }}
.caption {{ font-size: 13px; color: #4a5568; margin: 8px 0; }}
button {{ font-size: 15px; padding: 6px 18px; margin-right: 8px; border-radius: 6px; border: 1px solid #a0aec0; background: #fff; cursor: pointer; }}
button.on {{ background: #2b6cb0; color: #fff; border-color: #2b6cb0; }}
#bar {{ position: sticky; top: 0; background: #fff; padding: 10px 0; border-bottom: 1px solid #e2e8f0; z-index: 2; }}
</style></head><body>
<div id="bar"><b>RL 第 0 步盲评</b>：每组左边是 proxy，中间是 A，右边是 B（同一模型、同一提示，只有随机种子不同）。
请按"动作和交互是否符合描述、画面质量、是否贴合 proxy"综合判断哪个更好。已完成 <span id="count">0</span> / {total}。
<button onclick="exportAnswers()">导出结果</button></div>
{cards}
<script>
const KEY = "rl0-ab-answers";
const answers = JSON.parse(localStorage.getItem(KEY) || "{{}}");
function render() {{
  document.querySelectorAll(".card").forEach(card => {{
    const id = card.dataset.id;
    card.classList.toggle("done", id in answers);
    card.querySelectorAll("button").forEach(b => b.classList.toggle("on", answers[id] === b.dataset.v));
  }});
  document.getElementById("count").textContent = Object.keys(answers).length;
}}
function choose(id, v) {{ answers[id] = v; localStorage.setItem(KEY, JSON.stringify(answers)); render(); }}
function exportAnswers() {{
  const blob = new Blob([JSON.stringify(answers, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "rl0-ab-answers.json"; a.click();
}}
render();
</script></body></html>
"""

CARD = """<div class="card" data-id="{id}"><div>第 {n} 组</div>
<div class="labels"><span>proxy</span><span>A</span><span>B</span></div>
<video src="{src}" controls loop muted preload="none"></video>
<div class="caption">{caption}</div>
<button data-v="A" onclick="choose('{id}','A')">A 更好</button>
<button data-v="B" onclick="choose('{id}','B')">B 更好</button>
<button data-v="tie" onclick="choose('{id}','tie')">差不多</button></div>"""


def captions(data: Path) -> dict[str, str]:
    import torch

    out = {}
    for line in data.read_text().splitlines():
        if line.strip():
            path = Path(line.split()[0])
            out[path.stem] = torch.load(path, map_location="cpu", weights_only=False)["info"]["prompt"]
    return out


def build(args: argparse.Namespace) -> None:
    clips, board = Path(args.clips), Path(args.board)
    keys = {}
    for key_file in sorted(clips.glob("key-*.json")):
        keys.update(json.loads(key_file.read_text()))
    text = captions(Path(args.data))
    names = sorted(keys)
    random.Random(args.seed).shuffle(names)
    media = board / MEDIA
    media.mkdir(parents=True, exist_ok=True)
    blind, cards = {}, []
    for n, name in enumerate(names, 1):
        token = secrets.token_hex(6)
        shutil.copyfile(clips / name, media / f"{token}.mp4")
        blind[token] = name
        cards.append(CARD.format(id=token, n=n, src=f"{MEDIA}/{token}.mp4",
                                 caption=html.escape(text.get(keys[name]["sample"], ""))))
    (clips / "blind-map.json").write_text(json.dumps(blind, indent=2), encoding="utf-8")
    (board / PAGE).write_text(TEMPLATE.format(total=len(names), cards="\n".join(cards)), encoding="utf-8")
    print(f"{len(names)} clips -> {board / PAGE}")


def wilson(wins: float, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p, z = wins / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def score(args: argparse.Namespace) -> None:
    clips = Path(args.clips)
    blind = json.loads((clips / "blind-map.json").read_text())
    keys = {}
    for key_file in sorted(clips.glob("key-*.json")):
        keys.update(json.loads(key_file.read_text()))
    answers = json.loads(Path(args.answers).read_text())
    tally: dict[str, list[float]] = {}
    for token, choice in answers.items():
        clip = keys[blind[token]]
        tally.setdefault(clip["reward"], []).append(0.5 if choice == "tie" else float(choice == clip["pick"]))
    print("| reward | rated | human prefers the reward's pick | 95% Wilson |\n|---|---|---|---|")
    for reward, values in sorted(tally.items()):
        low, high = wilson(sum(values), len(values))
        print(f"| {reward} | {len(values)} | {sum(values) / len(values):.2f} | [{low:.2f}, {high:.2f}] |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--clips", required=True)
    b.add_argument("--board", required=True)
    b.add_argument("--data", required=True)
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--clips", required=True)
    s.add_argument("answers")
    args = p.parse_args()
    build(args) if args.command == "build" else score(args)


if __name__ == "__main__":
    main()
