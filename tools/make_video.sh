#!/usr/bin/env bash
# run.py --capture-video 가 쌓은 PNG 프레임을 mp4 로 묶는다.
#
# 이 PC 에는 ffmpeg 이 없다. GStreamer(x264enc + mp4mux)로 인코딩한다.
#   gst-inspect-1.0 x264enc / mp4mux 로 확인됨 (2026-10-07).
#
# 사용:
#   tools/make_video.sh <프레임폴더> [출력.mp4] [fps]
set -euo pipefail

DIR="${1:?프레임 폴더를 지정하세요}"
OUT="${2:-$DIR/../d8_success.mp4}"
FPS="${3:-30}"

N=$(find "$DIR" -maxdepth 1 -name 'f*.png' | wc -l)
if [ "$N" -lt 2 ]; then
    echo "프레임이 $N 장뿐입니다. 인코딩할 수 없습니다." >&2
    exit 1
fi
echo "프레임 $N 장, ${FPS}fps -> $OUT"

# multifilesrc 는 f000001.png 부터 연속 번호를 요구한다. run.py 가 그렇게 쓴다.
gst-launch-1.0 -q \
    multifilesrc location="$DIR/f%06d.png" index=1 caps="image/png,framerate=${FPS}/1" \
    ! pngdec ! videoconvert ! video/x-raw,format=I420 \
    ! x264enc bitrate=8000 speed-preset=medium key-int-max=60 \
    ! h264parse ! mp4mux ! filesink location="$OUT"

echo "완료: $OUT ($(du -h "$OUT" | cut -f1)), 길이 약 $(echo "scale=1; $N/$FPS" | bc) 초"
