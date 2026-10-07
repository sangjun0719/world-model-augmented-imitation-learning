"""문 밀기 궤적의 '자세 품질' 비교: 로봇 위치 후보별로 손목 카메라 방향·관절 꺾임·도달 오차를 잰다 (URDF, numpy).

손목 카메라: link7 로컬 +x 쪽(0.107 m)에 달림 → link7 x축의 월드 z성분 cam_up (1이면 카메라가 그리퍼 바로 위, -1이면 아래).
camera_up=True: 손목 회전까지 고정(link7 x = 위쪽) — 카메라가 항상 위.
posture: IK에 null-space 항을 넣어 자연스러운 기준 자세 쪽으로 당김 (팔꿈치 꺾임·손목 비틀림 억제).
"""
import math
import sys

import numpy as np

from urdf_ik_scan import Chain
from kimm_traj import w2r
from kimm_door_traj import HINGE

# 자연스러운 기준 자세 (오른팔: 어깨 앞으로, 팔꿈치 적당히 굽힘, 손목 비틀림 0)
Q_NOM_R = np.array([-0.6, -0.25, 0.0, -1.3, 0.0, 0.0, 0.0])


def door_R(phi, k, camera_up):
    p = math.radians(phi)
    u = np.array([-math.sin(p), math.cos(p), 0.0]); n = np.array([math.cos(p), math.sin(p), 0.0])
    a = -u + k * n; a /= np.linalg.norm(a)
    zl = -a
    if not camera_up:
        R = np.eye(3); R[:, 2] = zl; return R
    up = np.array([0.0, 0.0, 1.0]); xl = up - zl * (up @ zl); xl /= np.linalg.norm(xl)
    yl = np.cross(zl, xl)
    return np.c_[xl, yl, zl]


def ik_post(ch, lift, p_t, R_t, q0, iters=80, camera_up=True, q_nom=None, k_null=0.3, lam=0.05, w_roll=1.0, w_yaw=1.0):
    q = q0.copy()
    for _ in range(iters):
        M = ch.fk(q, lift); ep = p_t - M[:3, 3]
        if camera_up:
            Re = R_t @ M[:3, :3].T; w = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            ax = M[:3, 2]; w = (w - ax * (ax @ w)) + w_roll * ax * (ax @ w)     # roll 성분만 약하게
            ay = np.array([0.0, 0.0, 1.0]); w = (w - ay * (ay @ w)) + w_yaw * ay * (ay @ w)   # 월드 수직축 둘레(yaw) 약하게
        else:
            w = np.cross(M[:3, 2], R_t[:, 2])
        e = np.r_[ep, 0.5 * w]
        J = np.zeros((6, 7)); h = 1e-5
        for kk in range(7):
            dq = q.copy(); dq[kk] += h; Md = ch.fk(dq, lift)
            J[:3, kk] = (Md[:3, 3] - M[:3, 3]) / h
            if camera_up:
                Rd = Md[:3, :3] @ M[:3, :3].T; jw = 0.25 * np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]]) / h
                ax = M[:3, 2]; jw = (jw - ax * (ax @ jw)) + w_roll * ax * (ax @ jw)
                ay = np.array([0.0, 0.0, 1.0]); J[3:, kk] = (jw - ay * (ay @ jw)) + w_yaw * ay * (ay @ jw)
            else:
                J[3:, kk] = -0.5 * (np.cross(Md[:3, 2], R_t[:, 2]) - np.cross(M[:3, 2], R_t[:, 2])) / h
        Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(6))
        dq = Jp @ e
        if q_nom is not None:
            dq += (np.eye(7) - Jp @ J) @ (k_null * (q_nom - q))
        q = np.clip(q + np.clip(dq, -0.2, 0.2), ch.lo, ch.hi)
    M = ch.fk(q, lift)
    return q, float(np.linalg.norm(p_t - M[:3, 3])) * 1000, M


def evaluate(robot_xy, lift=-0.45, z=0.62, k=0.5, d=0.025, camera_up=True, posture=True, seed=0, starts=300, w_roll=1.0):
    ch = Chain("arm_r_link7"); rng = np.random.default_rng(seed)
    path = [(20.0, r) for r in np.linspace(0.60, 0.38, 12)] + [(phi, 0.38) for phi in np.linspace(20, 90, 36)]
    def pw(phi, r):
        p = math.radians(phi); u = np.array([-math.sin(p), math.cos(p), 0.0]); n = np.array([math.cos(p), math.sin(p), 0.0])
        return np.r_[HINGE, z] + r * u + d * n
    # 시작점: 다중 출발점으로 먼저 해를 찾고(null-space 없이), 그중 기준 자세에 가장 가까운 해를 고름
    phi, r = path[0]; best = None
    pr0, R0 = w2r(pw(phi, r), robot_xy, yaw=0.0), door_R(phi, k, camera_up)
    for _ in range(starts):
        q, pe, M = ik_post(ch, lift, pr0, R0, rng.uniform(ch.lo, ch.hi), iters=200, camera_up=camera_up, q_nom=None, w_roll=w_roll)
        rot_ok = (float(M[:3, 2] @ R0[:, 2]) > 0.99) and (not camera_up or float(M[2, 0]) > 0.5)
        if pe < 5 and rot_ok:
            margin = float(np.min(np.minimum(q - ch.lo, ch.hi - q)))
            cost = float(np.abs(q - Q_NOM_R).sum()) - 2.0 * min(margin, 0.5) if posture else 0.0
            if best is None or cost < best[0]:
                best = (cost, q)
    if best is None:
        return None
    q = best[1]; worst, cam_min, qs = 0.0, 1.0, []
    for phi, r in path:
        q, pe, M = ik_post(ch, lift, w2r(pw(phi, r), robot_xy, yaw=0.0), door_R(phi, k, camera_up), q, camera_up=camera_up,
                           q_nom=Q_NOM_R if posture else None, k_null=0.1, w_roll=w_roll)
        worst = max(worst, pe); cam_min = min(cam_min, float(M[2, 0])); qs.append(q.copy())
    qs = np.array(qs)
    margin = float(np.min(np.minimum(qs - ch.lo, ch.hi - qs)))
    wrist = np.abs(qs[:, 4:7]).max(0)
    elbow_z = min(ch.fk(qq, lift)[2, 3] for qq in qs[::6])   # 대략적 (EE z) 참고용
    return dict(err=round(worst, 1), cam_up_min=round(cam_min, 2), limit_margin=round(margin, 2),
                wrist_max=np.round(wrist, 2).tolist(), j4_range=np.round([qs[:, 3].min(), qs[:, 3].max()], 2).tolist(),
                q_start=np.round(qs[0], 2).tolist())


if __name__ == "__main__":
    cands = [(-2.90, -2.80), (-2.90, -2.70), (-2.90, -2.60), (-2.90, -2.50), (-2.80, -2.70), (-2.80, -2.60), (-3.00, -2.60)]
    for cam in (False, True):
        for xy in cands:
            for lift in (-0.45, -0.30):
                r = evaluate(xy, lift=lift, camera_up=cam)
                print(f"camera_up={cam} robot={xy} lift={lift}: {r}", flush=True)
