"""떨림 진단: 구간 실행 중 팔 관절이 얼마나 진동하는지 측정한다.

왜 필요한가
-----------
`tools/run.py` 는 구간 끝 값만 찍으므로 "움직이는 동안 떨린다" 를 숫자로 잡을 수 없다.
이 도구는 매 물리 스텝마다 관절 위치·목표를 기록해 아래를 뽑는다.

  - 떨림   : 위치의 2차 차분 RMS. 목표가 일정 속도로 움직일 때 2차 차분은 0 이어야 하므로
             그대로 진동 성분이 된다 (단위 µrad/step^2).
  - 울림   : 속도 부호가 뒤집힌 횟수. 한 방향으로만 가면 0 에 가깝다.
  - 추종   : |q - q_target| 의 최대·RMS.

2026-10-07 측정 배경: USD 드라이브의 damping/stiffness 가 전 관절 0.0004 s 로 동일하다.
변환기가 기계적으로 찍은 값이고 튜닝된 적이 없다. 반사 관성으로 계산한 감쇠비는
joint1 0.057 / joint2 0.052 / joint4 0.079 로 임계감쇠의 1/18 수준이다.

사용:
  OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/tremor_probe.py
  ... --damp-scale 18            # 전 관절 감쇠 18배 (joint1 기준 임계감쇠 근처)
  ... --phase reach --dq 0.008   # 다른 구간 / 더 촘촘한 목표 간격
"""
import argparse
import importlib.util
import math
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "motion"))

DEG = 180.0 / math.pi          # USD 각도 드라이브는 도 단위다. rad 단위로 보려면 곱한다.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift", type=float, default=-0.25, choices=[-0.25, -0.45])
    ap.add_argument("--phase", default="home",
                    help="측정할 구간 이름. home 은 현재 자세 -> home_arm")
    ap.add_argument("--damp-scale", type=float, default=1.0, help="전 팔관절 감쇠 배수")
    ap.add_argument("--stiff-scale", type=float, default=1.0, help="전 팔관절 강성 배수")
    ap.add_argument("--dq", type=float, default=0.015, help="한 스텝당 최대 관절 변화(rad)")
    ap.add_argument("--substeps", type=int, default=2)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "config", "d8.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    spec = importlib.util.spec_from_file_location("runner", os.path.join(ROOT, "tools", "run.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import PhysxSchema

        from kimm_door_traj import HandlePullPlanner

        scene = os.path.join(ROOT, cfg["scene"]["root"])
        omni.usd.get_context().open_stage(scene)
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        robot = cfg["scene"]["robot_prim"]
        runner.lock_lift(stage, robot, cfg["robot"]["lift_joint"], args.lift)
        runner.lock_wheels(stage, robot, tuple(cfg["robot"]["wheel_joints"]))
        runner.drive_left_finger(stage, robot)

        arm_names = list(cfg["robot"]["arm_joints"])
        gains = {}
        for jn in arm_names:
            prim = stage.GetPrimAtPath(f"{robot}/joints/{jn}")
            k = prim.GetAttribute("drive:angular:physics:stiffness")
            c = prim.GetAttribute("drive:angular:physics:damping")
            k0, c0 = float(k.Get()), float(c.Get())
            k.Set(k0 * args.stiff_scale)
            c.Set(c0 * args.damp_scale)
            gains[jn] = (k0 * args.stiff_scale, c0 * args.damp_scale)
        print(f"[tremor] 강성 x{args.stiff_scale:g}, 감쇠 x{args.damp_scale:g}", flush=True)
        for jn in arm_names:
            k, c = gains[jn]
            print(f"[tremor]   {jn:<14} k={k * DEG:9.1f} Nm/rad  c={c * DEG:8.3f} Nms/rad"
                  f"  c/k={c / k:.5f} s", flush=True)

        api = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(robot))
        it = cfg.get("solver") or {}
        api.CreateSolverPositionIterationCountAttr().Set(int(it.get("position_iterations", 64)))
        api.CreateSolverVelocityIterationCountAttr().Set(int(it.get("velocity_iterations", 16)))

        sim = SimulationContext(physics_dt=1 / 60, rendering_dt=1 / 60)
        sim.initialize_physics()
        art = SingleArticulation(robot); art.initialize()
        names = list(art.dof_names)
        ARM = np.array([names.index(n) for n in arm_names], dtype=np.int32)
        GRIP = np.array([names.index(cfg["robot"]["gripper_joint"])], dtype=np.int32)
        GRIP_L = np.array([names.index(cfg["robot"]["gripper_joint_left"])], dtype=np.int32)
        LIFT = np.array([names.index(cfg["robot"]["lift_joint"])], dtype=np.int32)
        art.set_joint_positions(np.array([args.lift], np.float32), joint_indices=LIFT)

        g = cfg["gripper"]
        home = np.asarray(cfg["robot"]["home_arm"], dtype=float)
        cur = np.asarray(art.get_joint_positions()).ravel()[ARM].astype(float)
        if args.phase == "home":
            qs, grips = [cur, home], [0.0, 0.0]
        else:
            planner = HandlePullPlanner(robot_xy=tuple(cfg["robot"]["base_xy"]), lift=args.lift,
                                        back_grasp=float(g.get("grasp_back", 0.0)),
                                        dz_grasp=float(g.get("grasp_dz", 0.0)),
                                        pull_rep=int(g.get("pull_rep", 2)))
            traj, _ = planner.plan(home, args.lift)
            found = [(n, q, gr) for n, q, gr in runner.phases(traj) if n == args.phase]
            if not found:
                print(f"[tremor] 구간 {args.phase} 없음. 가능: "
                      f"home, {', '.join(n for n, _, _ in runner.phases(traj))}", flush=True)
                return 1
            # 해당 구간 앞까지는 순간 이동으로 맞춰 놓고 그 구간만 측정한다.
            prev = home
            for n, q_, _ in runner.phases(traj):
                if n == args.phase:
                    break
                prev = np.asarray(q_[-1], dtype=float)
            art.set_joint_positions(np.asarray(prev, np.float32), joint_indices=ARM)
            qs, grips = found[0][1], found[0][2]
            cur = prev

        rows = runner.resample(np.column_stack(([cur] + list(qs),
                                                [grips[0] * g["close_handle"]]
                                                + [v * g["close_handle"] for v in grips])), args.dq)
        log_q, log_t = [], []
        for row in rows:
            tgt = np.asarray(row[:-1], dtype=float)
            art.apply_action(ArticulationAction(joint_positions=np.asarray(tgt, np.float32), joint_indices=ARM))
            for idx in (GRIP, GRIP_L):
                art.apply_action(ArticulationAction(joint_positions=np.array([row[-1]], np.float32),
                                                   joint_indices=idx))
            art.apply_action(ArticulationAction(joint_positions=np.array([args.lift], np.float32),
                                               joint_indices=LIFT))
            for _ in range(args.substeps):
                sim.step(render=False)
                log_q.append(np.asarray(art.get_joint_positions()).ravel()[ARM].astype(float))
                log_t.append(tgt)
        for _ in range(60):                      # 구간 끝에서 1 초 정지 — 잔진동을 본다
            sim.step(render=False)
            log_q.append(np.asarray(art.get_joint_positions()).ravel()[ARM].astype(float))
            log_t.append(np.asarray(rows[-1][:-1], dtype=float))

        q = np.array(log_q); t = np.array(log_t)
        d2 = np.diff(q, n=2, axis=0)                       # 2차 차분 = 진동 성분
        v = np.diff(q, axis=0)
        err = q - t
        tail = q[-60:]                                     # 정지 구간 잔진동
        print(f"\n[tremor] 구간 {args.phase}: 목표 {len(rows)}개, 물리 {len(q)} 스텝", flush=True)
        print(f"{'joint':<14}{'떨림[urad]':>12}{'울림[회]':>10}{'추종max[mrad]':>14}"
              f"{'추종rms[mrad]':>14}{'정지잔진동[urad]':>16}")
        for i, jn in enumerate(arm_names):
            flips = int(np.sum(np.diff(np.sign(v[:, i])) != 0))
            print(f"{jn:<14}{np.sqrt(np.mean(d2[:, i] ** 2)) * 1e6:>12.1f}{flips:>10d}"
                  f"{np.max(np.abs(err[:, i])) * 1e3:>14.3f}"
                  f"{np.sqrt(np.mean(err[:, i] ** 2)) * 1e3:>14.3f}"
                  f"{np.ptp(tail[:, i]) * 1e6:>16.1f}")
        print(f"\n[tremor] 합계 떨림 {np.sqrt(np.mean(d2 ** 2)) * 1e6:.1f} urad, "
              f"울림 {int(np.sum(np.diff(np.sign(v), axis=0) != 0))} 회, "
              f"정지 잔진동 최대 {np.max(np.ptp(tail, axis=0)) * 1e6:.1f} urad", flush=True)
        sim.stop()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
