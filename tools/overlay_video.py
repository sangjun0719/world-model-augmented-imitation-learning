"""녹화 프레임에 telemetry 수치를 입힌다.

왜 이렇게 하나
--------------
Isaac 의 검증 패널(omni.ui 창)을 띄운 채로 캡처하면 **뷰포트가 검게 나온다**
(2026-10-07, 패널 있는 캡처 5회 전부 검정 / 없는 캡처 전부 정상). 그리고
`capture_viewport_to_file` 은 뷰포트만 담으므로 패널은 어차피 영상에 안 들어간다.
그래서 녹화는 패널 없이 하고, 수치는 여기서 CSV 를 읽어 프레임에 직접 그린다.
화면에 보이는 값과 파일에 남는 값이 같은 측정기(`motion/telemetry.py`)에서 나온다.

사용:
  ~/env_isaacsim/bin/python tools/overlay_video.py \
      --frames logs/frames_demo --telemetry episodes/ep_demo.csv [--out <폴더>]
"""
import argparse
import csv
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"]


def load_font(size):
    for p in FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="run.py --capture-video 폴더")
    ap.add_argument("--telemetry", required=True, help="같은 실행의 telemetry CSV")
    ap.add_argument("--out", default="", help="출력 폴더 (기본: <frames>/over)")
    ap.add_argument("--font-size", type=int, default=19)
    args = ap.parse_args()

    out = args.out or os.path.join(args.frames, "over")
    os.makedirs(out, exist_ok=True)

    with open(args.telemetry, encoding="utf-8") as f:
        rows = {int(r["step"]): r for r in csv.DictReader(f)}
    if not rows:
        print("telemetry 가 비어 있습니다", file=sys.stderr); return 1

    # 프레임 -> step. 없으면 균등 분배로 근사한다.
    fmap_path = os.path.join(args.frames, "frames.csv")
    frames = sorted(f for f in os.listdir(args.frames) if f.startswith("f") and f.endswith(".png"))
    if os.path.exists(fmap_path):
        with open(fmap_path, encoding="utf-8") as f:
            fmap = {int(r["frame"]): int(r["step"]) for r in csv.DictReader(f)}
    else:
        steps = sorted(rows)
        fmap = {i + 1: steps[min(int(i * len(steps) / max(len(frames), 1)), len(steps) - 1)]
                for i in range(len(frames))}

    meta = {}
    mpath = os.path.splitext(args.telemetry)[0] + ".json"
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            meta = json.load(f)
    c = meta.get("condition", {})
    cond = (f"seed {c.get('seed')} | base ({c.get('base_xy', [0, 0])[0]:.3f},"
            f" {c.get('base_xy', [0, 0])[1]:.3f}) | door0 {c.get('door_start_deg', 0):.2f} deg"
            f" | lift {c.get('lift', 0):+.2f}") if c else ""

    font = load_font(args.font_size)
    small = load_font(max(args.font_size - 3, 10))
    steps_sorted = sorted(rows)
    n = 0
    for fn in frames:
        idx = int(fn[1:-4])
        step = fmap.get(idx)
        if step is None or step not in rows:
            step = min(steps_sorted, key=lambda s: abs(s - (step or 0)))
        r = rows[step]
        img = Image.open(os.path.join(args.frames, fn)).convert("RGB")
        d = ImageDraw.Draw(img, "RGBA")

        lines = [
            f"phase {r['phase']:<9} step {r['step']:>5}   t {float(r['sim_time']):6.2f} s",
            f"door  {float(r['door_deg']):7.2f} deg   vel {float(r['door_vel_deg_s']):6.2f} deg/s",
            f"bar depth {float(r['bar_depth_mm']):+7.1f} mm   hand-bar {float(r['hand_bar_dist_mm']):6.1f} mm",
            f"jaw gap   {float(r['jaw_half_gap_mm']):7.1f} mm   ctr err  {float(r['jaw_center_err_mm']):+6.1f} mm",
            f"contact   {int(r['contact_n']):>3} pts      {float(r['contact_force_N']):6.1f} N",
            f"grip cmd {float(r['grip_cmd']):.3f}  act R {float(r['grip_act']):.3f}  L {float(r['grip_left_act']):.3f}",
            f"base {float(r['base_drift_mm']):6.3f} mm      lift err {float(r['lift_err_mm']):+6.3f} mm",
        ]
        if r.get("other_contacts"):
            lines.append(f"other contact: {r['other_contacts'][:46]}")

        pad, lh = 10, args.font_size + 6
        w = max(d.textlength(t, font=font) for t in lines) + pad * 2
        h = lh * len(lines) + pad * 2
        d.rectangle([8, 8, 8 + w, 8 + h], fill=(0, 0, 0, 165))
        for i, t in enumerate(lines):
            d.text((8 + pad, 8 + pad + i * lh), t, font=font, fill=(255, 255, 255, 255))
        if cond:
            cw = d.textlength(cond, font=small) + 16
            d.rectangle([8, img.height - 34, 8 + cw, img.height - 6], fill=(0, 0, 0, 165))
            d.text((16, img.height - 29), cond, font=small, fill=(200, 220, 255, 255))

        img.save(os.path.join(out, fn))
        n += 1
    print(f"오버레이 {n} 장 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
