"""FFW-SG2 URDF로 순기구학 + 다중 출발점 IK를 풀어 kimm_sink 목표 도달 가능성을 조사 (numpy만, Isaac 불필요).

EE = arm_*_link7 에서 로컬 z -0.2 (Isaac Lab mimic_ik body_offset과 동일).
목표 자세: 그리퍼가 아래(link7 z = root +z), 손가락 닫힘축(EE 로컬 y)이 root x → 누운 컵(축 root y)을 가로질러 잡음.

사용: python3 urdf_ik_scan.py [--check]
"""
import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np

import os
# 원본: ~/Downloads/RVLab_isaacsim_aiworker/isaacsim_ai_worker_kimm/usd_ai_worker/ffw_sg2_follower.urdf (컨테이너에서도 보이게 복사)
URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "ffw_sg2_follower.urdf")


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_mat(a, t):
    a = np.asarray(a, float) / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def T(R=np.eye(3), p=(0, 0, 0)):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M


class Chain:
    def __init__(self, tip):
        root = ET.parse(URDF).getroot()
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], tip
        while link in joints:
            j = joints[link]; chain.append(j); link = j.find("parent").get("link")
        self.js = []
        for j in reversed(chain):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()] if o is not None else [0, 0, 0]
            rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()] if o is not None else [0, 0, 0]
            ax = j.find("axis"); ax = [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]
            lim = j.find("limit")
            lo, hi = (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None and lim.get("lower") else (0, 0)
            self.js.append(dict(name=j.get("name"), type=j.get("type"), T0=T(rpy_mat(*rpy), xyz), axis=ax, lo=lo, hi=hi))
        self.act = [i for i, j in enumerate(self.js) if j["type"] in ("revolute", "prismatic") and "arm_" in j["name"]]
        self.names = [self.js[i]["name"] for i in self.act]
        self.lo = np.array([self.js[i]["lo"] for i in self.act]); self.hi = np.array([self.js[i]["hi"] for i in self.act])

    def fk(self, q, lift):
        M = np.eye(4); k = 0
        for i, j in enumerate(self.js):
            M = M @ j["T0"]
            if j["type"] == "prismatic":
                M = M @ T(p=np.asarray(j["axis"]) * lift)
            elif j["type"] in ("revolute", "continuous"):
                if i in self.act:
                    M = M @ T(axis_mat(j["axis"], q[k])); k += 1
        return M @ T(p=(0, 0, -0.2))


def rot_err(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1, min(1, c))))


def ik(ch, lift, p_t, R_t, q0, iters=300, lam=0.05):
    q = q0.copy()
    for _ in range(iters):
        M = ch.fk(q, lift)
        ep = p_t - M[:3, 3]
        Re = R_t @ M[:3, :3].T
        w = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / 2
        e = np.r_[ep, 0.5 * w]
        if np.linalg.norm(ep) < 1e-3 and np.linalg.norm(w) < 1e-2:
            break
        J = np.zeros((6, len(q))); h = 1e-5
        for k in range(len(q)):
            dq = q.copy(); dq[k] += h
            Md = ch.fk(dq, lift)
            J[:3, k] = (Md[:3, 3] - M[:3, 3]) / h
            Rd = Md[:3, :3] @ M[:3, :3].T
            J[3:, k] = 0.5 * np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]]) / h
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), ch.lo, ch.hi)
    M = ch.fk(q, lift)
    return q, np.linalg.norm(p_t - M[:3, 3]) * 1000, rot_err(M[:3, :3], R_t)


def w2r(pw, rx, ry, yaw=math.pi):
    d = np.array(pw) - np.array([rx, ry, 0.0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="리셋 자세 FK만 출력 (sim 값과 비교)")
    ap.add_argument("--starts", type=int, default=40)
    ap.add_argument("--x", type=float, default=-3.33)
    ap.add_argument("--ys", type=float, nargs="+", default=[-0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3])
    ap.add_argument("--lifts", type=float, nargs="+", default=[0.0, -0.15, -0.3, -0.45])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    chains = {s: Chain(f"arm_{s}_link7") for s in "lr"}
    if a.check:
        for s, ch in chains.items():
            q = np.zeros(7); q[0] = 0.75; q[3] = -2.30
            M = ch.fk(q, -0.15)
            print(s, "reset EE (lift -0.15):", np.round(M[:3, 3], 3), "joints", ch.names)
        raise SystemExit
    R_down = axis_mat([0, 0, 1], math.pi / 2)   # Rz(90°): 로컬 y → root -x, 로컬 z → root z
    targets_w = {"grasp": (-3.91, -0.747, 0.749), "pre_grasp": (-3.91, -0.747, 0.869),
                 "carry": (-3.90, -1.10, 1.48), "place": (-3.88, -1.38, 1.37)}
    print("robot y | lift | arm | " + " | ".join(f"{k:>14s}" for k in targets_w))
    for y in a.ys:
        for lift in a.lifts:
            for s, ch in chains.items():
                row = []
                for name, pw in targets_w.items():
                    pr = w2r(pw, a.x, y)
                    best = (1e9, 1e9)
                    for _ in range(a.starts):
                        q0 = rng.uniform(ch.lo, ch.hi)
                        _, pe, re = ik(ch, lift, pr, R_down, q0, iters=150)
                        if pe + re < best[0] + best[1]:
                            best = (pe, re)
                        if pe < 5 and re < 3:
                            break
                    row.append(f"{best[0]:5.0f}mm/{best[1]:3.0f}°")
                print(f"{y:+.2f} | {lift:+.2f} | {s} | " + " | ".join(f"{r:>14s}" for r in row), flush=True)
