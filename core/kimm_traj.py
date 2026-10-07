"""kimm_sink 왼팔 궤적 생성 (URDF IK, numpy).

핵심 자세는 다중 출발점 IK로 전역에서 풀고(가지 선택), 하강/상승은 같은 가지에서 손끝 직선 경로로,
가지가 바뀌는 구간(리셋→컵 위, 컵 위→오븐)은 관절 공간 보간 + FK로 손끝 높이 검사.

순서: to_above(관절보간) → descend(직선) → close → lift_up(직선) → to_place(관절보간, 리프트 올림)
      → lower(직선) → open → retreat(직선)
URDF FK는 시뮬레이션(USD)과 0.0mm 일치 (logs/20260930_S6_fk_check.log).
"""
import math

import numpy as np

from urdf_ik_scan import Chain, axis_mat, rot_err

COUNTER_Z = 0.89           # 싱크대 상판
OVEN_TOP_Z = 1.305         # 오븐 윗면
SAFE_Z = 1.06              # 컵 위 (상판 위) 손끝 높이
CARRY_Z = 1.45             # 오븐 위 운반 높이 (상부장 바닥 1.67 아래)
RELEASE_Z = 1.37           # 놓는 높이 (오븐 윗면 + 컵 반지름 0.041 + 여유)
PLACE_XY = (-3.76, -1.30)  # 오븐 위, 가장자리(y -1.10)에서 20cm 안쪽. -1.20에선 놓은 뒤 누운 컵이 ~5cm 굴러 가장자리로 간 적 있음 (2026-09-30). 손끝 연속 IK 7mm, 기울기 11도


def w2r(pw, robot_xy, yaw=math.pi):
    d = np.array(pw, float) - np.array([robot_xy[0], robot_xy[1], 0.0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])


def ik_w(ch, lift, p_t, R_t, q0, iters=150, w_rot=1.0, tilt_only=False, lam=0.05):
    """가중 IK. tilt_only=True면 그리퍼 축(로컬 z)만 R_t의 z에 맞춤 (손목 회전 자유)."""
    q = q0.copy()
    for _ in range(iters):
        M = ch.fk(q, lift)
        ep = p_t - M[:3, 3]
        if tilt_only:
            z, zt = M[:3, 2], R_t[:, 2]
            w = np.cross(z, zt)
        else:
            Re = R_t @ M[:3, :3].T
            w = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / 2
        e = np.r_[ep, 0.5 * w_rot * w]
        if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(w) < 5e-3:
            break
        J = np.zeros((6, 7)); h = 1e-5
        for k in range(7):
            dq = q.copy(); dq[k] += h
            Md = ch.fk(dq, lift)
            J[:3, k] = (Md[:3, 3] - M[:3, 3]) / h
            if tilt_only:
                J[3:, k] = 0.5 * w_rot * (np.cross(Md[:3, 2], R_t[:, 2]) - np.cross(M[:3, 2], R_t[:, 2])) / h * -1
            else:
                Rd = Md[:3, :3] @ M[:3, :3].T
                J[3:, k] = 0.5 * w_rot * np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]]) / h
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), ch.lo, ch.hi)
    M = ch.fk(q, lift)
    tilt = math.degrees(math.acos(max(-1, min(1, float(M[:3, 2] @ R_t[:, 2])))))
    return q, float(np.linalg.norm(p_t - M[:3, 3])) * 1000, (tilt if tilt_only else rot_err(M[:3, :3], R_t))


class LeftArmPlanner:
    def __init__(self, robot_xy, grasp_lift=-0.30, place_lift=0.0, step=0.015, seed=0, travel_lift=None, grasp_dz=-0.02, carry_slow=2, grasp_psi_fixed=False):
        """travel_lift: 컵 위까지 이동·들어 올린 뒤의 리프트. None이면 grasp_lift와 같음.
        0이면 상체 최대 높이로 이동하고, 개수대 안으로 내려갈 때만 grasp_lift까지 내림 (2026-09-30 사용자 요청)."""
        self.ch = Chain("arm_l_link7")
        self.xy = robot_xy
        self.L0, self.L1 = grasp_lift, place_lift
        self.LT = grasp_lift if travel_lift is None else travel_lift
        self.carry_slow = carry_slow
        self.grasp_psi_fixed = grasp_psi_fixed
        self.grasp_dz = grasp_dz   # 잡는 손끝 높이 = 컵 중심 + dz (손가락이 적도에 걸리게, 닫을 때 ~1.2~3.8cm 밀려 올라감)
        self.step = step
        self.rng = np.random.default_rng(seed)
        self.R_across = axis_mat([0, 0, 1], math.pi / 2)   # 그리퍼 아래(로컬 z=+z), 닫힘축(로컬 y) = root -x
        self.R_down = np.eye(3)
        self.cache = {}

    # ---- 전역 IK ----
    def _global(self, key, p, R, lift, tilt_only=False, starts=80, seed=None, near=None):
        """다중 출발점 IK. near가 있으면 허용 오차 안의 해 중 near와 관절 거리(손목 가중)가 가장 작은 해."""
        tol_r = 2 if not tilt_only else 20
        best, ok = None, []
        cands = ([seed] if seed is not None else []) + ([self.cache[key]] if key in self.cache else [])
        cands += [self.rng.uniform(self.ch.lo, self.ch.hi) for _ in range(starts)]
        wj = np.array([1, 1, 1, 1, 2, 2, 2.0])   # 손목(j5~j7) 크게 도는 것에 벌점
        for q0 in cands:
            q, pe, re = ik_w(self.ch, lift, p, R, q0, iters=200, tilt_only=tilt_only)
            sc = pe + re
            if best is None or sc < best[0]:
                best = (sc, q, pe, re)
            if pe < 3 and re < tol_r:
                ok.append((float(np.sum(wj * np.abs(q - near))) if near is not None else 0.0, q, pe, re))
                if near is None:
                    break
        if ok:
            _, q, pe, re = min(ok, key=lambda c: c[0])
        else:
            _, q, pe, re = best
        self.cache[key] = q
        return q, pe, re

    def _line(self, name, q, lift, p0, p1, R, g, tilt_only=False, lift1=None):
        """손끝 직선 경로. lift1이 있으면 리프트도 lift→lift1로 같이 보간 (p는 root 좌표이므로 리프트와 무관)."""
        lift1 = lift if lift1 is None else lift1
        n = max(int(math.ceil(np.linalg.norm(p1 - p0) / self.step)), int(math.ceil(abs(lift1 - lift) / self.step)), 3)
        out, worst = [], (0.0, 0.0)
        for k in range(1, n + 1):
            ps = p0 + (p1 - p0) * k / n
            L = lift + (lift1 - lift) * k / n
            q, pe, re = ik_w(self.ch, L, ps, R, q, iters=40, tilt_only=tilt_only)
            worst = (max(worst[0], pe), max(worst[1], re))
            out.append((name, q.copy(), g, L))
        return out, q, worst

    def _joint(self, name, q0, q1, L0, L1, g, n=None):
        d = float(np.max(np.abs(q1 - q0)))
        n = n or max(int(math.ceil(d / 0.04)), int(math.ceil(abs(L1 - L0) / self.step)), 5)
        out, zmin = [], 9.0
        for k in range(1, n + 1):
            s = k / n; s2 = s * s * (3 - 2 * s)
            q = q0 + s2 * (q1 - q0); L = L0 + s2 * (L1 - L0)
            zmin = min(zmin, self.ch.fk(q, L)[2, 3])
            out.append((name, q, g, L))
        return out, zmin

    def _joint_safe(self, name, q0, q1, L0, L1, g, z_floor, via_z, depth=0):
        """관절 보간 중 손끝이 z_floor 아래로 가면, 중간 지점을 via_z로 들어 올린 경유 자세를 끼움."""
        seg, zmin = self._joint(name, q0, q1, L0, L1, g)
        if zmin >= z_floor or depth >= 3:
            return seg, zmin
        qm = 0.5 * (q0 + q1); Lm = 0.5 * (L0 + L1)
        pm = self.ch.fk(qm, Lm)[:3, 3].copy(); pm[2] = max(pm[2], via_z)
        qv, _, _ = ik_w(self.ch, Lm, pm, self.R_down, qm, iters=200, tilt_only=True, w_rot=0.3)
        a, z1 = self._joint_safe(name, q0, qv, L0, Lm, g, z_floor, via_z, depth + 1)
        b, z2 = self._joint_safe(name, qv, q1, Lm, L1, g, z_floor, via_z, depth + 1)
        return a + b, min(z1, z2)

    def _seg_hits(self, seg):
        return sorted({h for (_, q, g, L) in seg for h in collisions(self.ch, q, L, self.xy, holding=g > 0.5)})

    def _joint_free(self, name, q0, q1, L0, L1, g, tries=60):
        """관절 보간 구간이 장애물(오븐·상판·상부장)에 닿으면 경유 자세를 찾아 끼움 (최대 2개)."""
        seg, zmin = self._joint(name, q0, q1, L0, L1, g)
        hits = self._seg_hits(seg)
        if not hits:
            return seg, zmin
        Lm = 0.5 * (L0 + L1)
        best = None
        for t in range(tries):
            # 몸 앞 빈 공간 (root): 상판 모서리(x 0.28)보다 몸 쪽, 높게
            pv = np.array([self.rng.uniform(0.10, 0.30), self.rng.uniform(-0.05, 0.40), self.rng.uniform(1.10, 1.45)])
            seed = q0 + self.rng.uniform(0.2, 0.8) * (q1 - q0)
            qv, pe, _ = ik_w(self.ch, Lm, pv, self.R_down, seed, iters=150, tilt_only=True, w_rot=0.2)
            if pe > 20 or collisions(self.ch, qv, Lm, self.xy, holding=g > 0.5):
                continue
            a, za = self._joint(name, q0, qv, L0, Lm, g)
            b, zb = self._joint(name, qv, q1, Lm, L1, g)
            if self._seg_hits(a) or self._seg_hits(b):
                continue
            cost = float(np.abs(qv - q0).sum() + np.abs(q1 - qv).sum())
            if best is None or cost < best[0]:
                best = (cost, a + b, min(za, zb))
        if best is None:
            self.last_hits = {name: hits}
            return seg, zmin          # 못 찾으면 원래 경로 (충돌 목록은 last_hits)
        return best[1], best[2]

    def plan(self, q_start, lift_start, glass_center_w):
        gc = np.asarray(glass_center_w, float)
        p_above = w2r((gc[0], gc[1], SAFE_Z), self.xy)
        p_grasp = w2r(gc + np.array([0, 0, self.grasp_dz]), self.xy)
        p_over = w2r((*PLACE_XY, CARRY_Z), self.xy)
        p_place = w2r((*PLACE_XY, RELEASE_Z), self.xy)
        # 전역: 잡기 자세 (컵을 가로지름). 두 방향(±90°) 중 좋은 것
        cand = []
        # 손목 방향: +90°·-90° 중 IK가 좋은 쪽. (+90° 고정 시험 diag8: 3/8로 더 나빠서 기본은 둘 다 허용)
        for R in ((self.R_across,) if self.grasp_psi_fixed else (self.R_across, axis_mat([0, 0, 1], -math.pi / 2))):
            q, pe, re = self._global(("grasp", R[0, 1] > 0), p_grasp, R, self.L0)
            cand.append((pe + re, q, pe, re, R))
        _, q_grasp, e_g, r_g, R_g = min(cand, key=lambda c: c[0])
        # 컵 위는 잡기 해에서 출발해 같은 가지로 (수직 직선)
        up, q_above, w_up = self._line("tmp", q_grasp, self.L0, p_grasp, p_above, R_g, 0.0, lift1=self.LT)
        # 오븐 위: 그리퍼 아래 향함만 요구(손목 회전 자유)
        q_over, e_o, r_o = self._global("over", p_over, self.R_down, self.L1, tilt_only=True, seed=q_above, near=q_above)
        LT = self.LT
        traj, errs = [], {"grasp_ik": (round(e_g, 1), round(r_g, 1)), "over_ik": (round(e_o, 1), round(r_o, 1))}
        seg, zmin = self._joint_free("to_above", q_start, q_above, lift_start, LT, 0.0)
        traj += seg; errs["to_above_min_z"] = round(zmin, 3)
        down = [(n_, q_, g_, L_) for (n_, q_, g_, L_) in reversed(up)][1:] + [("descend", q_grasp, 0.0, self.L0)]
        # (reversed(up)의 리프트는 LT→L0 순서로 자동으로 내려감)
        traj += [("descend", q_, 0.0, L_) for (_, q_, _, L_) in down]; errs["line"] = tuple(round(x, 1) for x in w_up)
        # 그리퍼는 천천히 닫음: 한 번에 닫으면 한쪽 손가락이 컵을 밀어 손목 j7(토크 한계 5.1 Nm)이 역구동됨 (diag8 ep5·8)
        traj += [("close", q_grasp.copy(), min(1.0, k / 30), self.L0) for k in range(1, 46)]
        traj += [("lift_up", q_, 1.0, L_) for (_, q_, _, L_) in up for _rep in (0, 1)]   # 들어 올리기는 절반 속도
        # 운반: 손끝 직선 연속 IK (그리퍼 아래 향함만 유지, 손목 회전 자유) → 잡은 가지에서 이어짐
        p_up = np.array([p_above[0], p_above[1], CARRY_Z])
        s1, q_mid, w1 = self._line("to_place", q_above, LT, p_above, p_up, self.R_down, 1.0, tilt_only=True, lift1=self.L1)
        s2, q_ov2, w2 = self._line("to_place", q_mid, self.L1, p_up, p_over, self.R_down, 1.0, tilt_only=True)
        M2 = self.ch.fk(q_ov2, self.L1)
        e2 = float(np.linalg.norm(M2[:3, 3] - p_over)) * 1000
        tilt2 = math.degrees(math.acos(max(-1, min(1, float(M2[2, 2])))))
        cart = s1 + s2
        if e2 < 25 and tilt2 < 30 and not self._seg_hits(cart):
            traj += cart; q_over = q_ov2
            errs["to_place"] = ("cartesian", round(e2, 1), round(tilt2, 1))
        else:
            seg, zmin = self._joint_free("to_place", q_above, q_over, LT, self.L1, 1.0)
            traj += seg
            errs["to_place"] = ("joint", round(e2, 1), round(tilt2, 1))
        M = self.ch.fk(q_over, self.L1); R_o = M[:3, :3]
        seg, q_pl, w_pl = self._line("lower", q_over, self.L1, p_over, p_place, R_o, 1.0)
        traj += seg; errs["lower"] = tuple(round(x, 1) for x in w_pl)
        traj += [("open", q_pl.copy(), max(0.0, 1.0 - k / 20), self.L1) for k in range(1, 31)]
        seg, _, _ = self._line("retreat", q_pl, self.L1, p_place, p_over, R_o, 0.0)
        traj += seg
        errs["collisions"] = traj_collisions(self.ch, traj, self.xy)
        if self.carry_slow > 1:   # 컵을 든 이동 구간은 점을 반복해 속도를 낮춤 (손목 j7 토크 한계 5.1 Nm)
            traj = [pt for pt in traj for _ in range(self.carry_slow if pt[0] in ("to_place", "lower") else 1)]
        return traj, errs


# ---- 충돌 검사 (장애물 상자, 팔 링크 선분 샘플) ----
# world 좌표 상자 (x0,x1,y0,y1,z0,z1). 방 메시 bbox 기준 + 여유 0.02
OBSTACLES_W = {
    "sink_counter": (-4.26, -3.59, -2.23, -0.15, 0.00, 0.91),   # 싱크대(상판 0.89) — 개수대 구멍은 아래 BASIN으로 뺌
    "oven": (-4.14, -3.62, -1.68, -1.08, 0.89, 1.315),   # 윗면 1.305 (놓을 때 닿는 건 정상)
    "upper_cabinet": (-4.26, -3.91, -2.23, -0.15, 1.65, 2.40),
}
BASIN_W = (-4.12, -3.80, -1.05, -0.33)   # 개수대 개구부 (이 xy 안에서는 z 0.705까지 허용)
# 로봇 자신 (root 좌표, 로봇이 바라보는 쪽 +x, 왼쪽 +y): 몸통 기둥, 오른팔(리셋 자세 그대로)
SELF_R = {
    "torso": (-0.20, 0.10, -0.17, 0.17, 0.30, 1.75),
    "right_arm": (-0.10, 0.45, -0.36, -0.08, 0.75, 1.50),
}
GLASS_HALF = 0.075   # 누운 컵 반길이 (+여유), 손끝 로컬 x축 방향으로 뻗음
GLASS_R = 0.045


def link_points(ch, q, lift, per_seg=4):
    """팔 체인의 각 관절 원점 + EE, 그 사이를 per_seg개로 나눈 점들 (root 좌표)."""
    M = np.eye(4); k = 0; pts = []
    from urdf_ik_scan import T as _T, axis_mat as _ax
    for i, j in enumerate(ch.js):
        M = M @ j["T0"]
        if j["type"] == "prismatic":
            M = M @ _T(p=np.asarray(j["axis"]) * lift)
        elif j["type"] in ("revolute", "continuous") and i in ch.act:
            M = M @ _T(_ax(j["axis"], q[k])); k += 1
        if "arm_l" in j["name"]:
            pts.append(M[:3, 3].copy())
    pts.append((M @ _T(p=(0, 0, -0.2)))[:3, 3])
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        for s in np.linspace(0, 1, per_seg, endpoint=False):
            out.append(a + s * (b - a))
    out.append(pts[-1])
    return np.array(out[per_seg:])   # 어깨 근처(몸통 안)는 제외


def r2w(pr, robot_xy, yaw=math.pi):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * pr[0] - s * pr[1] + robot_xy[0], s * pr[0] + c * pr[1] + robot_xy[1], pr[2]])


def held_glass_points(ch, q, lift):
    """쥔 컵 (손끝 로컬 x축으로 ±GLASS_HALF, 아래로 반지름) 표면 샘플 (root)."""
    M = ch.fk(q, lift); c = M[:3, 3]; ax = M[:3, 0]; dn = -M[:3, 2]
    pts = []
    for s in np.linspace(-GLASS_HALF, GLASS_HALF, 5):
        for off in (0.0, GLASS_R):
            pts.append(c + s * ax + off * dn)
    return np.array(pts)


def collisions(ch, q, lift, robot_xy, holding=False):
    hits = []
    pts = link_points(ch, q, lift)
    if holding:
        pts = np.vstack([pts, held_glass_points(ch, q, lift)])
        for pr in pts:
            for name, (x0, x1, y0, y1, z0, z1) in SELF_R.items():
                if x0 < pr[0] < x1 and y0 < pr[1] < y1 and z0 < pr[2] < z1:
                    hits.append("self_" + name)
    for pr in pts:
        w = r2w(pr, robot_xy)
        for name, (x0, x1, y0, y1, z0, z1) in OBSTACLES_W.items():
            if x0 < w[0] < x1 and y0 < w[1] < y1 and z0 < w[2] < z1:
                # 개수대 개구부 안은 허용. 바닥(0.705)에 놓인 컵을 잡는 순간 든-컵 검사점이 바닥 아래로 ~2cm 계산되므로 0.68까지
                if name == "sink_counter" and BASIN_W[0] < w[0] < BASIN_W[1] and BASIN_W[2] < w[1] < BASIN_W[3] and w[2] > 0.68:
                    continue
                hits.append(name)
    return sorted(set(hits))


def traj_collisions(ch, traj, robot_xy):
    """구간별 충돌 장애물 목록."""
    out = {}
    for name, q, g, L in traj:
        h = collisions(ch, q, L, robot_xy, holding=g > 0.5 and name not in ("close",))
        if h:
            out.setdefault(name, set()).update(h)
    return {k: sorted(v) for k, v in out.items()}
