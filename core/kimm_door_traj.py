"""kimm_door 오른팔 궤적: 문 D8 틈에 손끝을 넣어 경첩 중심 원호로 밀어 연다 (URDF IK, numpy).

문 각도 φ: 문 방향 u = (-sinφ, cosφ) (경첩→바깥 끝), 서랍장 쪽 법선 n = (cosφ, sinφ). 문이 열리면 문 위 점은 -n 쪽(로봇 쪽)으로 감.
손끝 위치 = 경첩 + r·u + d·n (d>0: 문 안쪽 면 뒤, 서랍장 쪽).
그리퍼 축(가리키는 방향) = normalize(-u + K·n) (경첩 쪽에서 서랍장 안쪽으로 ~27° 기울임), 손목 회전 자유.
도달성 (tools/urdf_ik_door_scan3.py): 로봇 (-2.90,-2.80) yaw 0, 리프트 -0.45, z 0.62, K 0.5 → 삽입+20→92° 원호 최대 2.9mm.

구간: above(문 위로 접근, 리프트 0→-0.45) → descend(수직 하강) → insert(r 0.60→0.38) → push(φ 20→95°) → hold
"""
import math

import numpy as np

from urdf_ik_scan import Chain
from kimm_traj import ik_w, w2r

HINGE = np.array([-2.212, -3.463])
DOOR_TOP_Z = 0.85


def door_hand(phi_deg, r, d, z, k):
    p = math.radians(phi_deg)
    u = np.array([-math.sin(p), math.cos(p), 0.0]); n = np.array([math.cos(p), math.sin(p), 0.0])
    pos = np.r_[HINGE, z] + r * u + d * n
    a = -u + k * n; a /= np.linalg.norm(a)
    R = np.eye(3); R[:, 2] = -a          # link7 z = -(가리키는 방향) → EE(link7 -z 0.2)가 a 방향으로 뻗음
    return pos, R


class RightArmDoorPlanner:
    def __init__(self, robot_xy=(-2.90, -2.80), lift_push=-0.45, z=0.62, k=0.5, d=0.025, robot_yaw=0.0,
                 r_pre=0.60, r_in=0.38, phi0=20.0, phi_end=90.0, step=0.012, seed=0):
        self.ch = Chain("arm_r_link7")
        self.xy, self.L, self.z, self.k, self.d = robot_xy, lift_push, z, k, d
        self.yaw = robot_yaw
        self.r_pre, self.r_in, self.phi0, self.phi_end = r_pre, r_in, phi0, phi_end
        self.step = step
        self.rng = np.random.default_rng(seed)
        self.cache = None

    def _r(self, pw):
        return w2r(pw, self.xy, yaw=self.yaw)

    def _solve(self, pw, R, lift, seed, iters=60):
        return ik_w(self.ch, lift, self._r(pw), R, seed, iters=iters, tilt_only=True)

    def _global(self, pw, R, lift):
        cands = ([self.cache] if self.cache is not None else []) + [self.rng.uniform(self.ch.lo, self.ch.hi) for _ in range(80)]
        best = None
        for q0 in cands:
            q, pe, re = self._solve(pw, R, lift, q0, iters=200)
            if best is None or pe + re < best[1] + best[2]:
                best = (q, pe, re)
            if pe < 2 and re < 2:
                break
        self.cache = best[0]
        return best

    def plan(self, q_start, lift_start, phi0=None, direct=False):
        """반환: traj [(구간, q, 그리퍼(0~1), 리프트)], errs."""
        phi0 = self.phi0 if phi0 is None else phi0
        ch, errs, traj = self.ch, {}, []
        # 1) 삽입 시작 자세 (문 바깥 끝 너머) — 전역 IK
        p_pre, R_pre = door_hand(phi0, self.r_pre, self.d, self.z, self.k)
        if direct is None or direct is False:
            q_pre, e, r = self._global(p_pre, R_pre, self.L)
            errs["pre_ik"] = (round(e, 1), round(r, 1))
        else:
            # direct = 현재 손 위치(월드). 문 앞에 둔 채 문과 나란히 바깥 끝 너머(r_pre)까지 옆으로 → 높이 z로 → 문 뒤(d)로.
            A = np.asarray(direct, float)
            p = math.radians(phi0); u = np.array([-math.sin(p), math.cos(p), 0.0]); nn = np.array([math.cos(p), math.sin(p), 0.0])
            H = np.r_[HINGE, 0.0]
            dA = float((A - H) @ nn); rA = float((A - H) @ u)
            B = H + self.r_pre * u + dA * nn + np.array([0, 0, A[2]])
            C = B.copy(); C[2] = self.z
            # 시작 자세에서 손 방향만 R_pre 로 (제자리)
            q, worst = q_start.copy(), 0.0
            q_a = None
            best = None
            for q0 in [q_start] + [self.rng.uniform(self.ch.lo, self.ch.hi) for _ in range(60)]:
                qa, e, r = self._solve(A, R_pre, self.L, q0, iters=200)
                sc = e + r + 5 * float(np.max(np.abs(qa - q_start)))
                if best is None or sc < best[0]:
                    best = (sc, qa)
            q_a = best[1]
            n = max(int(math.ceil(float(np.max(np.abs(q_a - q_start))) / 0.03)), 10)
            for i in range(1, n + 1):
                s_ = i / n; s2 = s_ * s_ * (3 - 2 * s_)
                traj.append(("turn", q_start + s2 * (q_a - q_start), 1.0, lift_start + s2 * (self.L - lift_start)))
            q = q_a
            for name, P0, P1 in (("slide", A, B), ("lower", B, C), ("to_edge", C, p_pre)):
                m = max(int(math.ceil(np.linalg.norm(P1 - P0) / self.step)), 2)
                for i in range(1, m + 1):
                    q, e, r = self._solve(P0 + (P1 - P0) * i / m, R_pre, self.L, q); worst = max(worst, e + r)
                    traj.append((name, q.copy(), 1.0, self.L))
            q_pre = q
            errs["pre_ik"] = (0.0, 0.0)
            errs["descend"] = round(worst, 1)
            return self._insert_push(traj, errs, q_pre, phi0)
        # 2) 같은 가지에서 위로 (문 윗면 위) — 거꾸로 풀어 descend 경로 확보
        p_high = p_pre + np.array([0, 0, max(DOOR_TOP_Z + 0.12 - self.z, 0.0)])
        up, q, worst = [], q_pre.copy(), 0.0
        n = max(int(math.ceil((p_high[2] - p_pre[2]) / self.step)), 3)
        for i in range(1, n + 1):
            s = i / n
            L = self.L                                              # 하강은 리프트 -0.45 고정 (상체는 above 구간에서 내림)
            q, e, r = self._solve(p_pre + s * (p_high - p_pre), R_pre, L, q)
            worst = max(worst, e + r); up.append((q.copy(), L))
        errs["descend"] = round(worst, 1)
        q_high, L_high = up[-1]
        # 3) 시작 → 문 위 (관절 보간, 리프트 유지)
        d = float(np.max(np.abs(q_high - q_start)))
        n = max(int(math.ceil(d / 0.03)), 10)
        zmin = 9.0
        for i in range(1, n + 1):
            s = i / n; s2 = s * s * (3 - 2 * s)
            qi = q_start + s2 * (q_high - q_start); Li = lift_start + s2 * (L_high - lift_start)
            zmin = min(zmin, ch.fk(qi, Li)[2, 3]); traj.append(("above", qi, 1.0, Li))
        errs["above_min_z_root"] = round(zmin, 3)
        # 4) 수직 하강 (리프트 같이 내림)
        traj += [("descend", qq, 1.0, LL) for (qq, LL) in reversed(up[:-1])] + [("descend", q_pre, 1.0, self.L)]
        return self._insert_push(traj, errs, q_pre, phi0)

    def _insert_push(self, traj, errs, q_pre, phi0):
        # 5) 삽입 r_pre → r_in
        q, worst = q_pre.copy(), 0.0
        n = max(int(math.ceil((self.r_pre - self.r_in) / self.step)), 3)
        for i in range(1, n + 1):
            r_ = self.r_pre + (self.r_in - self.r_pre) * i / n
            pw, R = door_hand(phi0, r_, self.d, self.z, self.k)
            q, e, r = self._solve(pw, R, self.L, q); worst = max(worst, e + r)
            traj += [("insert", q.copy(), 1.0, self.L)] * 2         # 삽입은 천천히
        errs["insert"] = round(worst, 1)
        # 6) 원호로 밀기 φ0 → phi_end
        worst = 0.0
        n = max(int(math.ceil(math.radians(self.phi_end - phi0) * self.r_in / self.step)), 5)
        for i in range(1, n + 1):
            phi = phi0 + (self.phi_end - phi0) * i / n
            pw, R = door_hand(phi, self.r_in, self.d, self.z, self.k)
            q, e, r = self._solve(pw, R, self.L, q); worst = max(worst, e + r)
            traj.append(("push", q.copy(), 1.0, self.L))
        errs["push"] = round(worst, 1)
        traj += [("hold", q.copy(), 1.0, self.L)] * 20
        return traj, errs


# ---------------------------------------------------------------------------------------------
# 1단계: 닫힌 문 손잡이를 잡고 살짝(≈22°) 당겨 연다 → 놓고 뒤로 빠짐 → 위로 올림. 이후 RightArmDoorPlanner.
# 손잡이 막대 중심: 경첩에서 r 0.33, 문 앞면에서 -0.0435 (로봇 쪽), 높이 0.82.
# 그리퍼는 서랍장 쪽(+n)을 가리키되 경첩 반대쪽으로 20° 기울임, 손가락 닫힘축 = 수직 → 막대를 위아래로 집음.
# 도달성 (URDF): 로봇 (-2.90,-2.80), 리프트 -0.45 에서 접근·당기기·후퇴 최대 3.4 mm, 팔꿈치 정상(j4 -1.3~-1.9).
HANDLE_R, HANDLE_D, HANDLE_Z, HANDLE_TILT = 0.33, -0.0435, 0.82, -20.0


def handle_hand(phi_deg, back=0.0, r=HANDLE_R, z=HANDLE_Z, tilt=HANDLE_TILT):
    p = math.radians(phi_deg)
    u = np.array([-math.sin(p), math.cos(p), 0.0]); n = np.array([math.cos(p), math.sin(p), 0.0])
    t = math.radians(tilt); a = math.cos(t) * n - math.sin(t) * u
    zl = -a; yl = np.array([0.0, 0.0, 1.0]); xl = np.cross(yl, zl)
    return np.r_[HINGE, z] + r * u + (HANDLE_D - back) * n, np.c_[xl, yl, zl]


class HandlePullPlanner:
    # back_grasp / dz_grasp / pull_rep 은 2026-10-07 에 추가한 선택 보정이다 (기본값 = 이전 동작).
    #   back_grasp : 잡기·당기기 구간에서 손을 문에서 이만큼 뒤로 뺀다.
    #                바퀴를 잠근 뒤에는 손이 계획대로 들어가는데, 그러면 막대가 패드 원점에서 14 mm 에 물려
    #                손가락 끝이 문짝(막대 중심에서 28 mm)에 닿고 왼쪽 mimic 이 막힌다. 8 mm 빼면 22 mm 로 들어간다.
    #   dz_grasp   : 턱 중앙 보정. 모델이 예측한 턱 중앙보다 실제가 2.7 mm 아래라 그만큼 올려 잡는다.
    #   pull_rep   : 당기기 한 점당 반복 횟수. 문 저항이 각속도에 비례하므로 천천히 당기면 덜 미끄러진다.
    def __init__(self, robot_xy=(-2.90, -2.80), lift=-0.45, phi_pull=22.0, back_pre=0.12, back_post=0.15,
                 rise=0.20, step=0.01, dz=0.03, deep=0.015, tilt=HANDLE_TILT, seed=0,
                 back_grasp=0.0, dz_grasp=0.0, pull_rep=2):
        self.back_grasp, self.dz_grasp, self.pull_rep = back_grasp, dz_grasp, pull_rep
        self.dz, self.deep, self.tilt = dz, deep, tilt
        from door_pose_eval import ik_post, Q_NOM_R
        self.ik, self.qn = ik_post, Q_NOM_R
        self.ch = Chain("arm_r_link7")
        self.xy, self.L, self.phi_pull = robot_xy, lift, phi_pull
        self.back_pre, self.back_post, self.rise, self.step = back_pre, back_post, rise, step
        self.rng = np.random.default_rng(seed)

    def _hh(self, *a, **k):
        return handle_hand(*a, tilt=self.tilt, **k)

    def _ik(self, p, R, q):
        return self.ik(self.ch, self.L, w2r(p, self.xy, yaw=0.0), R, q, iters=150, camera_up=True,
                       q_nom=self.qn, k_null=0.1, w_yaw=0.15)

    def plan(self, q_start, lift_start):
        """반환: traj [(구간, q, 그리퍼(0 열림~1 닫힘), 리프트)], errs."""
        ch, errs, traj = self.ch, {}, []
        # 접근 시작점: 다중 출발점 → 기준 자세에 가깝고 관절 여유 큰 해
        p0, R0 = self._hh(0.0, self.back_pre, z=HANDLE_Z - self.dz); best = None
        for _ in range(150):
            q, pe, M = self.ik(ch, self.L, w2r(p0, self.xy, yaw=0.0), R0, self.rng.uniform(ch.lo, ch.hi),
                               iters=200, camera_up=True, w_yaw=0.15)
            if pe < 3:
                margin = float(np.min(np.minimum(q - ch.lo, ch.hi - q)))
                cost = float(np.abs(q - self.qn).sum()) - 3 * min(margin, 0.5) + (10 if q[3] > -0.2 else 0)
                if best is None or cost < best[0]:
                    best = (cost, q)
        q_pre = best[1]
        for _ in range(3):
            q_pre, _, _ = self._ik(p0, R0, q_pre)
        # 1) 상체(리프트)만 먼저 내림 (팔 정지) → 잠깐 정지 → 팔만 접근 시작점으로 (관절 보간, 그리퍼 열림)
        n = max(int(math.ceil(abs(self.L - lift_start) / 0.01)), 5)
        for i in range(1, n + 1):
            s = i / n; s2 = s * s * (3 - 2 * s)
            traj.append(("lift_down", q_start.copy(), 0.0, lift_start + s2 * (self.L - lift_start)))
        traj += [("lift_down", q_start.copy(), 0.0, self.L)] * 10
        n = max(int(math.ceil(float(np.max(np.abs(q_pre - q_start))) / 0.03)), 10)
        for i in range(1, n + 1):
            s = i / n; s2 = s * s * (3 - 2 * s)
            traj.append(("reach", q_start + s2 * (q_pre - q_start), 0.0, self.L))
        q, worst = q_pre.copy(), 0.0

        def seg(name, pts, grip, rep=1):
            nonlocal q, worst
            for p, R in pts:
                q, pe, _ = self._ik(p, R, q); worst = max(worst, pe)
                traj.extend([(name, q.copy(), grip, self.L)] * rep)
        # 2) 접근: 손잡이보다 dz 낮은 높이로 직선 진입 → 위로 올려 막대를 손가락 사이에 끼움 (천천히)
        dz = self.dz
        m = int(math.ceil(self.back_pre / self.step))
        # 벌린 손가락은 막대 앞면까지밖에 안 닿음 → deep 만큼 더 넣음 (손끝이 막대 뒤 틈까지)
        dp = self.deep
        seg("approach", [self._hh(0.0, (self.back_pre + dp) * (1 - i / m) - dp, z=HANDLE_Z - dz) for i in range(1, m + 1)], 0.0, 2)
        m = int(math.ceil(dz / 0.005))
        seg("lift_in", [self._hh(0.0, -dp, z=HANDLE_Z - dz * (1 - i / m)) for i in range(1, m + 1)], 0.0, 2)
        errs["approach"] = round(worst, 1); worst = 0.0
        # 3) 잡기: 그리퍼를 서서히 닫으면서(닫힐 때 손끝이 ~2.8cm 앞으로 나감) 팔을 deep 만큼 뒤로 → 문짝 면에 안 닿음
        for i in range(1, 26):
            g = min(1.0, i / 15)
            back = -dp + (dp + self.back_grasp) * g
            q, pe, _ = self._ik(*self._hh(0.0, back, z=HANDLE_Z + self.dz_grasp), q); worst = max(worst, pe)
            traj.append(("grasp", q.copy(), g, self.L))
        # 4) 당기기: 손잡이를 원호로 φ 0 → phi_pull
        m = max(int(math.ceil(math.radians(self.phi_pull) * HANDLE_R / self.step)), 5)
        seg("pull", [self._hh(self.phi_pull * i / m, self.back_grasp, z=HANDLE_Z + self.dz_grasp)
                     for i in range(1, m + 1)], 1.0, self.pull_rep)
        errs["pull"] = round(worst, 1); worst = 0.0
        # 5) 놓기
        traj += [("release", q.copy(), max(0.0, 1 - i / 15), self.L) for i in range(1, 21)]
        # 6) 뒤로 빠지기 → 위로 올리기 (그리퍼 닫아 둠: 밀기 단계는 손가락 닫은 채로 넣음)
        m = int(math.ceil(self.back_post / self.step))
        seg("retreat", [self._hh(self.phi_pull, self.back_post * i / m) for i in range(1, m + 1)], 0.0)
        errs["retreat"] = round(worst, 1)
        return traj, errs
