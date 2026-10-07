"""성공/실패 에피소드 배치 생성기.

왜 필요한가
-----------
학습용 데이터셋은 한 번에 100 개를 만든다. 중간에 하나가 깨져도 멈추면 안 되고, 중단 뒤
이어서 돌릴 수 있어야 하며, **만들어진 파일이 쓸 만한지 그 자리에서 검사**해야 한다.
검사 없이 쌓으면 8 시간 뒤에 못 쓰는 파일 100 개를 발견하게 된다.

에피소드마다 하는 일
  1) tools/run.py 를 별도 프로세스로 실행 (Isaac Sim 은 프로세스당 1 회 실행이 안전하다)
  2) 생성된 HDF5 를 열어 검사: 샘플 수, 카메라 3 대, 이미지가 검지 않은지, action 이 상수가
     아닌지, auxiliary 가 있는지
  3) 결과를 manifest.csv 한 줄로 남긴다 (모드·seed·성공여부·문 각도·샘플수·소요시간·검사결과)

이미 있는 파일은 건너뛴다. 그래서 중단 후 같은 명령을 다시 돌리면 이어서 진행한다.

사용:
  ~/env_isaacsim/bin/python tools/batch.py --out episodes/batch1
  ... --n-success 50 --n-fail 50 --dry-run
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.expanduser("~/env_isaacsim/bin/python")

# 실패 모드. 전부 2026-10-07 에 실제로 관측된 실패를 되돌린 것이다.
# `--no-grasp-gate` 는 전 모드 공통 — 파지에 실패해도 끝까지 가야 "당기다 놓치는" 구간이 남는다.
FAIL_MODES = [
    ("substeps2", ["--substeps", "2", "--dq", "0.015"],
     "계단 입력에 의한 구동계 진동 -> 막대가 패드 끝으로 밀려 이탈"),
    ("nowheel", ["--no-wheel-lock", "--base-tol-mm", "1000"],
     "바퀴 잠금 해제 -> 베이스가 밀려 손이 손잡이에 못 미침"),
    ("noleft", ["--no-left-drive"],
     "왼쪽 손가락 드라이브 없음 -> 한쪽만 접촉"),
    ("graspback0", ["--grasp-back", "0"],
     "손끝이 문짝을 밀어 왼쪽 mimic 이 막힘"),
    ("bigrand", ["--rand-xy", "0.05"],
     "베이스 무작위를 여유 밖으로 -> 파지 빗나감 (자연 발생 실패)"),
]
SUCCESS_ARGS = ["--rand-xy", "0.015", "--rand-door-deg", "3"]


def check(path, expect_success):
    """만들어진 HDF5 가 학습에 쓸 만한지 본다. (ok, 메모) 를 돌려준다."""
    try:
        import h5py
        import numpy as np
        with h5py.File(path, "r") as f:
            n = len(f["time/sim_time"])
            if n < 100:
                return False, f"샘플 {n} 개뿐"
            cams = list(f["observations/images"]) if "observations/images" in f else []
            if len(cams) < 3:
                return False, f"카메라 {len(cams)} 대"
            for c in cams:                       # 첫·중간 프레임이 검지 않은지
                d = f[f"observations/images/{c}/jpeg"]
                for i in (0, n // 2):
                    if len(d[i]) < 2000:
                        return False, f"{c} 프레임 {i} 가 {len(d[i])} 바이트 (검은 화면 의심)"
            a = np.asarray(f["action"])
            if len(np.unique(a, axis=0)) < 10:
                return False, "action 이 거의 상수 (버퍼 복사 버그)"
            if "auxiliary/door_deg" not in f:
                return False, "auxiliary 없음"
            got = str(f["metadata"].attrs.get("success", "")) == "True"
            note = "" if got == expect_success else f"기대와 다름(실제 success={got})"
            return True, note
    except Exception as e:
        return False, f"열기 실패: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "episodes", "batch1"))
    ap.add_argument("--n-success", type=int, default=50)
    ap.add_argument("--n-fail", type=int, default=50)
    ap.add_argument("--seed0-success", type=int, default=1)
    ap.add_argument("--seed0-fail", type=int, default=101)
    ap.add_argument("--timeout", type=int, default=1800, help="에피소드당 초")
    ap.add_argument("--modes", default="", help="쉼표 구분. 비우면 전부")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    modes = FAIL_MODES
    if args.modes:
        keep = [m.strip() for m in args.modes.split(",")]
        modes = [m for m in FAIL_MODES if m[0] in keep]
    if not modes and args.n_fail:
        print("쓸 수 있는 실패 모드가 없습니다", file=sys.stderr); return 1

    jobs = []
    for i in range(args.n_success):
        jobs.append({"name": f"succ_{i:03d}", "mode": "success", "seed": args.seed0_success + i,
                     "extra": SUCCESS_ARGS, "expect": True})
    per = args.n_fail // max(len(modes), 1)
    rest = args.n_fail - per * len(modes)
    k = 0
    for mi, (mname, margs, _) in enumerate(modes):
        cnt = per + (1 if mi < rest else 0)
        for j in range(cnt):
            jobs.append({"name": f"fail_{mname}_{j:03d}", "mode": mname,
                         "seed": args.seed0_fail + k,
                         "extra": margs + (SUCCESS_ARGS if "--rand-xy" not in margs else
                                           ["--rand-door-deg", "3"]),
                         "expect": False})
            k += 1

    print(f"[batch] 총 {len(jobs)} 개 (성공 {args.n_success} / 실패 {args.n_fail})")
    for mname, _, why in modes:
        print(f"[batch]   실패 모드 {mname:<12} {why}")
    if args.dry_run:
        for j in jobs[:5] + (["..."] if len(jobs) > 5 else []):
            print("   ", j if isinstance(j, str) else (j["name"], j["seed"], j["extra"]))
        return 0

    man = os.path.join(args.out, "manifest.csv")
    new = not os.path.exists(man)
    mf = open(man, "a", newline="", encoding="utf-8")
    w = csv.writer(mf)
    if new:
        w.writerow(["episode", "mode", "seed", "expect_success", "actual_success",
                    "final_door_deg", "n_samples", "size_mb", "wall_s", "ok", "note"])
        mf.flush()

    t_all = time.time()
    done = skipped = bad = 0
    for idx, j in enumerate(jobs, 1):
        h5 = os.path.join(args.out, j["name"] + ".hdf5")
        if os.path.exists(h5):
            skipped += 1
            print(f"[batch] ({idx}/{len(jobs)}) {j['name']} 이미 있음 — 건너뜀", flush=True)
            continue
        cmd = [PY, os.path.join(ROOT, "tools", "run.py"), "--headless",
               "--seed", str(j["seed"]), "--no-grasp-gate",
               "--hdf5", h5] + list(j["extra"])
        t0 = time.time()
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
        log = os.path.join(args.out, j["name"] + ".log")
        try:
            with open(log, "w") as lf:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"[batch] ({idx}/{len(jobs)}) {j['name']} 시간 초과", flush=True)
        dt = time.time() - t0

        ok, note = (False, "파일 없음")
        succ = door = nsmp = size = ""
        if os.path.exists(h5):
            ok, note = check(h5, j["expect"])
            try:
                import h5py
                with h5py.File(h5, "r") as f:
                    succ = str(f["metadata"].attrs.get("success", ""))
                    door = round(float(f["metadata"].attrs.get("final_door_deg", float("nan"))), 3)
                    nsmp = len(f["time/sim_time"])
            except Exception:
                pass
            size = round(os.path.getsize(h5) / 1048576, 1)
        if not ok:
            bad += 1
        else:
            done += 1
        w.writerow([j["name"], j["mode"], j["seed"], j["expect"], succ, door, nsmp, size,
                    round(dt, 1), ok, note])
        mf.flush()
        el = time.time() - t_all
        rem = (len(jobs) - idx) * (el / max(idx - skipped, 1))
        print(f"[batch] ({idx}/{len(jobs)}) {j['name']:<22} {dt:6.1f}s  success={succ:<5} "
              f"door={door}  ok={ok} {note}  남은 예상 {rem / 3600:.1f}h", flush=True)

    mf.close()
    print(f"[batch] 완료: 정상 {done}, 건너뜀 {skipped}, 문제 {bad}, "
          f"총 {(time.time() - t_all) / 3600:.2f} 시간")
    print(f"[batch] manifest: {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
