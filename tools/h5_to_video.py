"""에피소드 HDF5 를 사람이 볼 수 있는 mp4 로 바꾼다.

HDF5 는 영상 파일이 아니다. 배열 여러 개를 담는 바이너리 컨테이너라 VLC 로 열리지 않는다.
이 도구는 안에 든 **로봇 카메라 JPEG** 를 꺼내 타일로 붙이고, `auxiliary` 의 수치를 찍어
mp4 로 만든다. 100 개를 만든 뒤 "이 에피소드가 제대로 된 건가" 를 눈으로 확인할 때 쓴다.

`tools/run.py --capture-video` 로 만드는 영상과 다르다.
  - 그쪽: 바깥에서 보는 제3자 시점. 동작 전체를 본다.
  - 이쪽: **로봇이 실제로 보는 화면** (ACT 가 학습할 입력). 실패 원인을 볼 때 이쪽이 필요하다.

사용:
  ~/env_isaacsim/bin/python tools/h5_to_video.py episodes/ep_000001.hdf5
  ... --out ~/ep1.mp4 --fps 30 --cams d405_right,zed_rgb
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

AUX = [("door_deg", "door", "{:7.2f} deg"),
       ("bar_depth_mm", "bar depth", "{:+7.1f} mm"),
       ("hand_bar_dist_mm", "hand-bar", "{:7.1f} mm"),
       ("jaw_half_gap_mm", "jaw gap", "{:7.1f} mm"),
       ("contact_n", "contact", "{:7.0f} pts"),
       ("contact_force_N", "force", "{:7.1f} N"),
       ("base_drift_mm", "base", "{:7.3f} mm")]
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--out", default="", help="기본: <h5 와 같은 이름>.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cams", default="", help="쉼표 구분. 기본: 파일에 있는 전부")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    import cv2
    import h5py
    from PIL import Image, ImageDraw, ImageFont

    out = args.out or os.path.splitext(args.h5)[0] + ".mp4"
    font = None
    for p in FONTS:
        if os.path.exists(p):
            font = ImageFont.truetype(p, 17); break
    font = font or ImageFont.load_default()

    with h5py.File(args.h5, "r") as f:
        cams = list(f["observations/images"]) if "observations/images" in f else []
        if args.cams:
            want = [c.strip() for c in args.cams.split(",")]
            missing = [c for c in want if c not in cams]
            if missing:
                print(f"없는 카메라: {missing}. 파일에 있는 것: {cams}", file=sys.stderr)
                return 1
            cams = want
        if not cams:
            print("이미지가 없습니다", file=sys.stderr); return 1
        n = len(f[f"observations/images/{cams[0]}/jpeg"])
        aux = {k: np.asarray(f[f"auxiliary/{k}"]) for k, _, _ in AUX
               if f"auxiliary/{k}" in f}
        meta = dict(f["metadata"].attrs) if "metadata" in f else {}
        t = np.asarray(f["time/sim_time"]) if "time/sim_time" in f else np.arange(n) / args.fps

        head = (f"{os.path.basename(args.h5)}   success={meta.get('success', '?')}   "
                f"final {meta.get('final_door_deg', float('nan')):.2f} deg   n={n}")
        cond = ""
        if "base_xy" in meta:
            try:
                b = json.loads(meta["base_xy"]) if isinstance(meta["base_xy"], str) else meta["base_xy"]
                cond = (f"seed {meta.get('seed')} | base ({b[0]:.3f}, {b[1]:.3f}) | "
                        f"door0 {float(meta.get('door_start_deg', 0)):.2f} deg | "
                        f"lift {float(meta.get('lift', 0)):+.2f}")
            except Exception:
                cond = ""

        tmp = tempfile.mkdtemp(prefix="h5vid_")
        for i in range(n):
            tiles = []
            for c in cams:
                buf = np.asarray(f[f"observations/images/{c}/jpeg"][i], dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)          # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.putText(img, c, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2, cv2.LINE_AA)
                tiles.append(img)
            h = max(x.shape[0] for x in tiles)
            tiles = [np.pad(x, ((0, h - x.shape[0]), (0, 0), (0, 0))) for x in tiles]
            canvas = np.hstack(tiles)

            pil = Image.fromarray(canvas)
            if args.scale != 1.0:
                pil = pil.resize((int(pil.width * args.scale), int(pil.height * args.scale)))
            bar_h = 26 * (len(aux) + 2) // 2 + 20
            full = Image.new("RGB", (pil.width, pil.height + bar_h), (18, 18, 18))
            full.paste(pil, (0, 0))
            d = ImageDraw.Draw(full)
            d.text((10, pil.height + 6), head, font=font, fill=(255, 255, 255))
            y = pil.height + 28
            line = f"t {float(t[i]):7.2f} s   "
            for k, label, fmt in AUX:
                if k in aux:
                    line += f"{label} {fmt.format(float(aux[k][i]))}   "
            d.text((10, y), line, font=font, fill=(200, 230, 255))
            if cond:
                d.text((10, y + 22), cond, font=font, fill=(170, 190, 220))
            full.save(os.path.join(tmp, f"f{i + 1:06d}.png"))

    cmd = ["gst-launch-1.0", "-q",
           "multifilesrc", f"location={tmp}/f%06d.png", "index=1",
           f"caps=image/png,framerate={args.fps}/1",
           "!", "pngdec", "!", "videoconvert", "!", "video/x-raw,format=I420",
           "!", "x264enc", "bitrate=8000", "speed-preset=medium", "key-int-max=60",
           "!", "h264parse", "!", "mp4mux", "!", f"filesink", f"location={out}"]
    subprocess.run(cmd, check=True)
    print(f"완료: {out} ({n} 프레임, {n / args.fps:.1f} 초, 카메라 {', '.join(cams)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
