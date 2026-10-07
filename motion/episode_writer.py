"""ROS 없이 에피소드를 HDF5 로 쓴다.

왜 ROS 를 안 쓰나
-----------------
`~/kimm_record/episode_recorder.py` 는 ROS 토픽을 받아 적는다. 그런데 문 동작(`tools/run.py`)은
Isaac Sim 단독 실행기라 ROS 로 아무것도 발행하지 않는다. 그래서 recorder 의 게이트
(`/isaac_sim/joint_commands` 수신 필요)를 통과할 수 없다. 여기서는 시뮬 안에서 카메라
렌더 프로덕트를 직접 떠서 같은 스키마로 쓴다. 부수 효과로 `/task/door_state` 퍼블리셔가
없는 문제도 사라진다 — 문 각도를 이미 알고 있기 때문이다.

스키마는 `episode_recorder.py` 와 **같다**. 그래야 `~/kimm_record/to_act.py` 가 그대로 먹는다.

  time/sim_time                     (N,)     float64
  observations/qpos, qvel, effort   (N, 31)  float32   attrs: joint_names
  action                            (N, 31)  float32   attrs: joint_names
  observations/images/<cam>/jpeg    (N,)     vlen uint8
  sync/<key>_dt_ms                  (N,)     float64   (여기서는 전부 0 — 아래 참고)
  auxiliary/...                     이 과제 전용 상세 기록 (문 각도·막대 깊이·접촉 등)
  metadata                          attrs

sync 가 0 인 이유: ROS 경로는 토픽마다 도착 시각이 달라 동기화 오차가 생긴다. 여기서는 한
물리 스텝에서 상태와 이미지를 동시에 뜨므로 정의상 0 이다. 스키마를 맞추려고 남겨 둔다.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np


class EpisodeWriter:
    def __init__(self, cfg, art, jpeg_quality=90, rate_hz=30.0, cameras=None):
        import omni.replicator.core as rep

        self.cfg, self.art = cfg, art
        self.q = int(jpeg_quality)
        self.rate = float(rate_hz)
        self.names = list(art.dof_names)
        self.samples = []
        self.t_wall0 = time.time()

        cams = cameras or (cfg.get("cameras") or {})
        res = tuple((cfg.get("camera_res") or [640, 480]))
        self.cams = {}
        for key, path in cams.items():
            rp = rep.create.render_product(path, res)
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp])
            self.cams[key] = {"path": path, "rp": rp, "ann": ann, "res": res}
        self.res = res

    def camera_keys(self):
        return list(self.cams)

    def sample(self, sim_time, action_full, aux=None):
        """한 샘플(상태 + 이미지)을 모은다. 물리 스텝 직후에 부른다."""
        import cv2

        imgs = {}
        for key, c in self.cams.items():
            a = c["ann"].get_data()
            if a is None or not len(a):
                return False                       # 아직 렌더가 안 돌았다 — 이 샘플은 버린다
            rgba = np.asarray(a, dtype=np.uint8).reshape(c["res"][1], c["res"][0], -1)
            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.q])
            if not ok:
                return False
            imgs[key] = np.frombuffer(buf.tobytes(), dtype=np.uint8)

        try:
            eff = np.asarray(self.art.get_measured_joint_efforts(), np.float32).ravel()
        except Exception:
            eff = np.zeros(len(self.names), np.float32)
        # **반드시 복사한다.** np.asarray 는 이미 float32 인 배열을 그대로 돌려주므로,
        # 호출자가 매 스텝 갱신하는 action 버퍼를 그대로 담으면 전 샘플이 같은 객체를 가리켜
        # 저장 시점의 마지막 값으로 덮인다 (2026-10-07 실제로 그렇게 나왔다).
        self.samples.append({
            "t": float(sim_time),
            "qpos": np.array(self.art.get_joint_positions(), np.float32).ravel(),
            "qvel": np.array(self.art.get_joint_velocities(), np.float32).ravel(),
            "effort": np.array(eff, np.float32).ravel(),
            "action": np.array(action_full, np.float32).ravel(),
            "img": imgs,
            "aux": dict(aux or {}),
        })
        return True

    def save(self, path, success, meta=None):
        import h5py

        if not self.samples:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        S = self.samples
        n = len(S)
        t = np.array([s["t"] for s in S], np.float64)
        ep = os.path.splitext(os.path.basename(path))[0]
        wall = time.time() - self.t_wall0

        with h5py.File(path, "w") as f:
            f.create_dataset("time/sim_time", data=t)
            for k in ("qpos", "qvel", "effort"):
                f.create_dataset(f"observations/{k}",
                                 data=np.stack([s[k] for s in S]), compression="gzip")
            f.create_dataset("action", data=np.stack([s["action"] for s in S]),
                             compression="gzip")
            f["observations"].attrs["joint_names"] = json.dumps(self.names)
            f["action"].attrs["joint_names"] = json.dumps(self.names)
            f["observations/effort"].attrs["source"] = "SingleArticulation.get_measured_joint_efforts"

            for key, c in self.cams.items():
                g = f.create_group(f"observations/images/{key}")
                d = g.create_dataset("jpeg", (n,), dtype=h5py.special_dtype(vlen=np.uint8))
                for i, s in enumerate(S):
                    d[i] = s["img"][key]
                g.attrs["width"], g.attrs["height"] = int(c["res"][0]), int(c["res"][1])
                g.attrs["format"], g.attrs["color_order"] = "jpeg", "bgr"
                g.attrs["jpeg_quality"] = self.q
                g.attrs["camera_prim"] = c["path"]

            gs = f.create_group("sync")
            for k in list(self.cams) + ["action", "joint_states"]:
                ds = gs.create_dataset(k + "_dt_ms", data=np.zeros(n, np.float64))
                ds.attrs["abs_max_ms"] = ds.attrs["abs_mean_ms"] = 0.0
                ds.attrs["signed_mean_ms"] = 0.0
                ds.attrs["note"] = "한 물리 스텝에서 상태와 이미지를 동시에 떠 정의상 0"

            if S[0]["aux"]:
                ga = f.create_group("auxiliary")
                for k in S[0]["aux"]:
                    ga.create_dataset(k, data=np.array([s["aux"].get(k, np.nan) for s in S],
                                                       np.float64), compression="gzip")
                ga.attrs["note"] = ("자동 판정·분석용. ACT 입력(카메라/qpos)과 구분한다. "
                                    "측정 정의는 motion/telemetry.py 와 같다.")

            m = f.create_group("metadata")
            m.attrs["episode_id"] = ep
            m.attrs["sim_time_start"], m.attrs["sim_time_end"] = float(t[0]), float(t[-1])
            m.attrs["duration_sim_s"] = float(t[-1] - t[0])
            m.attrs["duration_wall_s"] = float(wall)
            m.attrs["realtime_factor"] = float((t[-1] - t[0]) / max(wall, 1e-9))
            m.attrs["success"] = str(bool(success))
            m.attrs["control_rate_hz"] = self.rate
            m.attrs["n_samples"] = n
            m.attrs["expected_samples"] = int(round((t[-1] - t[0]) * self.rate)) + 1
            m.attrs["missed_samples"] = 0
            m.attrs["writer"] = "kimm_door_v2/motion/episode_writer.py 1.0 (ROS 미사용)"
            for k, v in (meta or {}).items():
                m.attrs[k] = v if isinstance(v, (int, float, bool, str)) else json.dumps(v)
        return path
