"""에피소드 상세 기록. 매 물리 스텝의 기하·접촉·관절 상태를 한 줄로 만든다.

왜 필요한가
-----------
성공/실패 에피소드로 학습하려면 "열렸다/안 열렸다" 말고 **왜 그렇게 됐는지**가 숫자로 남아야 한다.
`tools/run.py` 의 구간 끝 출력만으로는 실패 지점을 되짚을 수 없다. 여기서 매 스텝 기록하는 것:

  문        각도(deg), 각속도(deg/s)
  파지      막대가 패드 원점에서 몇 mm 들어갔는지(깊이), 두 패드 중앙에서 얼마나 치우쳤는지,
            현재 턱 반간격, 패드 원점에서 막대까지 직선거리 (= 손과 손잡이 거리)
  접촉      손잡이 막대 접촉 점수와 힘(N), 손가락이 닿은 다른 물체
  그리퍼    목표 / 실제(오른쪽) / 실제(왼쪽 mimic)
  팔        관절 7개의 목표와 실제
  베이스    기준 위치에서의 이동(mm), 몸통 높이 오차(mm)

측정 정의는 `tools/grasp_probe.py` 와 같다 (2026-10-07 그 도구로 파지 실패 원인 3건을 찾았다).
"""
from __future__ import annotations

import math

import numpy as np
from pxr import PhysicsSchemaTools, UsdGeom

from gripper_geom import jaw_half


class Telemetry:
    """한 실행분의 측정기. `header()` 한 번, 매 스텝 `row()`."""

    def __init__(self, stage, art, door, cfg, door_deg, base0, lift_target,
                 arm_idx, grip_idx, grip_left_idx, lift_idx):
        self.stage, self.art, self.door, self.cfg = stage, art, door, cfg
        self.door_deg = door_deg
        self.base0 = np.asarray(base0, dtype=float).ravel()
        self.lift_target = float(lift_target)
        self.ARM, self.GRIP, self.GRIP_L, self.LIFT = arm_idx, grip_idx, grip_left_idx, lift_idx
        self.robot = cfg["scene"]["robot_prim"]
        self.door_path = cfg["scene"]["door_prim"]
        self.bar0 = np.array(cfg["door"]["handle_bar_world"], dtype=float)
        self.hinge = np.array([*cfg["door"]["hinge_xy"], 0.0], dtype=float)
        self.xf = UsdGeom.XformCache()
        self.arm_names = list(cfg["robot"]["arm_joints"])

    # --- 기하 ---------------------------------------------------------------------
    def _bar_now(self):
        """문이 열린 만큼 손잡이 막대 중심을 회전시킨 월드 좌표."""
        a = math.radians(self.door_deg())
        r = self.bar0 - self.hinge
        c, s = math.cos(a), math.sin(a)
        return self.hinge + np.array([c * r[0] - s * r[1], s * r[0] + c * r[1], r[2]])

    def _pad_local(self, side, point):
        """패드 좌표계에서 본 점. z=손가락 길이 방향(깊이), y=집는 축."""
        prim = self.stage.GetPrimAtPath(f"{self.robot}/gripper_r_rh_p12_rn_{side}")
        M = np.array(self.xf.GetLocalToWorldTransform(prim)).T
        return (np.linalg.inv(M) @ np.r_[point, 1.0])[:3]

    def _contacts(self):
        """손잡이 막대 접촉 점수·힘(N)과, 손가락이 닿은 다른 물체."""
        from omni.physx import get_physx_simulation_interface
        bar_path = self.door_path + "/col_bar"
        headers, data = get_physx_simulation_interface().get_contact_report()
        total = np.zeros(3); n = 0; others = {}
        for h in headers:
            a, b = (str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1))
            if "gripper_" not in a + b:
                continue
            imp = np.zeros(3)
            for d in data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]:
                imp += np.asarray(d.impulse, dtype=float)
            if bar_path in (a, b):
                total += imp; n += 1
            else:
                other = a if "gripper_" not in a else b
                key = other.split("/")[-1] if "col_" in other else other.split("/")[4]
                others[key] = round(others.get(key, 0.0) + float(np.linalg.norm(imp)) * 60, 1)
        return n, float(np.linalg.norm(total)) * 60, others

    # --- 기록 ---------------------------------------------------------------------
    def header(self):
        cols = ["step", "sim_time", "phase",
                "door_deg", "door_vel_deg_s",
                "bar_depth_mm", "jaw_center_err_mm", "jaw_half_gap_mm", "hand_bar_dist_mm",
                "contact_n", "contact_force_N", "other_contacts",
                "grip_cmd", "grip_act", "grip_left_act",
                "base_drift_mm", "lift_err_mm"]
        cols += [f"{n}_tgt" for n in self.arm_names]
        cols += [f"{n}_act" for n in self.arm_names]
        return cols

    def row(self, step, sim_time, phase, target_q, target_grip):
        self.xf.Clear()
        j = np.asarray(self.art.get_joint_positions()).ravel()
        bar = self._bar_now()
        lr, ll = self._pad_local("r2", bar), self._pad_local("l2", bar)
        th = float(j[self.GRIP[0]])
        n, force, others = self._contacts()
        try:
            w = np.asarray(self.door.get_angular_velocity(), dtype=float).ravel()
            dvel = math.degrees(float(np.linalg.norm(w)))
        except Exception:
            dvel = float("nan")
        base = np.asarray(self.art.get_world_pose()[0], dtype=float).ravel()
        lift = float(j[self.LIFT[0]])
        gl = float(j[self.GRIP_L[0]]) if self.GRIP_L is not None else float("nan")

        out = [step, round(sim_time, 5), phase,
               round(self.door_deg(), 4), round(dvel, 4),
               round(float(lr[2]) * 1000, 3),
               round((abs(float(lr[1])) - abs(float(ll[1]))) * 500, 3),
               round(jaw_half(th) * 1000, 3),
               round(float(np.linalg.norm(lr)) * 1000, 3),
               n, round(force, 3), ("" if not others else str(others).replace(",", ";")),
               round(float(target_grip), 5), round(th, 5), round(gl, 5),
               round(float(np.linalg.norm(base - self.base0)) * 1000, 4),
               round((lift - self.lift_target) * 1000, 4)]
        out += [round(float(v), 6) for v in np.asarray(target_q, dtype=float).ravel()]
        out += [round(float(v), 6) for v in j[self.ARM]]
        return out
