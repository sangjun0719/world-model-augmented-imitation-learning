"""상체 높이별로 1단계(손잡이 당기기)·2단계(예전 밀기) 계획 오차 — 가장 높은 가능 높이 찾기."""
import itertools
import sys
import numpy as np
from kimm_door_traj import HandlePullPlanner, RightArmDoorPlanner
for L in [float(sys.argv[1])]:
    best_o = None
    for tilt in (-20, -35, -50):
        P = HandlePullPlanner(lift=L, phi_pull=12, tilt=tilt)
        t, e = P.plan(np.zeros(7), L)
        sc = max(e.values())
        if best_o is None or sc < best_o[0]:
            best_o = (sc, tilt, t[0][1])
    best_p = None
    for z, k, rp in itertools.product((0.62, 0.70, 0.78), (0.3, 0.5), (0.60, 0.52)):
        _, e = RightArmDoorPlanner(lift_push=L, z=z, k=k, r_pre=rp).plan(np.zeros(7), L, phi0=22.0)
        sc = e["descend"] + e["push"] + e["insert"] + sum(e["pre_ik"])
        if best_p is None or sc < best_p[0]:
            best_p = (sc, z, k, rp)
    print(f"lift {L}: open err {best_o[0]:.1f}mm (tilt {best_o[1]}) | push err {best_p[0]:.1f}mm (z {best_p[1]} k {best_p[2]} r_pre {best_p[3]})", flush=True)
