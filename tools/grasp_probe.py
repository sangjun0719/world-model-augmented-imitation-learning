"""파지 진단: 손잡이 막대가 손가락 패드 안에서 어디에 물렸고, 당길 때 어디로 밀리는지 측정한다.

왜 필요한가
-----------
문 각도만 보면 "파지 실패"까지만 알 수 있다. 이 도구는 매 구간에서
  - 막대 중심이 패드 원점에서 몇 mm 들어갔는지 (패드 길이 45 mm, 끝으로 갈수록 턱이 벌어진다)
  - 막대가 두 패드 사이 중앙에서 얼마나 치우쳤는지
  - 막대에 걸린 접촉 점수와 힘
  - 손가락이 문짝·캐비닛 같은 다른 것에 닿았는지
를 같이 찍는다. 2026-10-07 에 이 측정으로 세 가지를 찾았다.
  1) 바퀴가 굴러 베이스가 40 mm 밀린다 (tools/run.py lock_wheels 로 수정)
  2) 왼쪽 손가락에 드라이브가 없어 0.27 rad 뒤처진다 (drive_left_finger 로 수정)
  3) 손이 계획대로 들어가면 손가락 끝이 문짝에 닿는다 (config 의 grasp_back 으로 수정)

사용:
  OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/grasp_probe.py
  ... --lift -0.45 --no-wheel-lock      # 바퀴를 풀어 베이스가 밀리는 것을 재현
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift", type=float, default=-0.25, choices=[-0.25, -0.45])
    ap.add_argument("--no-wheel-lock", action="store_true", help="바퀴를 풀어 베이스 밀림을 재현")
    ap.add_argument("--no-left-drive", action="store_true", help="왼쪽 손가락 드라이브를 넣지 않는다")
    ap.add_argument("--every", type=int, default=6, help="몇 스텝마다 찍을지")
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
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, PhysxSchema, UsdGeom

        from gripper_geom import jaw_half
        from kimm_door_traj import HandlePullPlanner

        scene = os.path.join(ROOT, cfg["scene"]["root"])
        omni.usd.get_context().open_stage(scene)
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        robot, door_path = cfg["scene"]["robot_prim"], cfg["scene"]["door_prim"]
        runner.lock_lift(stage, robot, cfg["robot"]["lift_joint"], args.lift)
        if not args.no_wheel_lock:
            runner.lock_wheels(stage, robot, tuple(cfg["robot"]["wheel_joints"]))
        if not args.no_left_drive:
            runner.drive_left_finger(stage, robot)
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(door_path)).CreateThresholdAttr().Set(0.0)
        api = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(robot))
        api.CreateSolverPositionIterationCountAttr().Set(int((cfg.get("solver") or {}).get("position_iterations", 64)))
        api.CreateSolverVelocityIterationCountAttr().Set(int((cfg.get("solver") or {}).get("velocity_iterations", 16)))

        sim = SimulationContext(physics_dt=1 / 60, rendering_dt=1 / 60)
        sim.initialize_physics()
        art = SingleArticulation(robot); art.initialize()
        door = SingleRigidPrim(door_path); door.initialize()
        names = list(art.dof_names)
        ARM = np.array([names.index(n) for n in cfg["robot"]["arm_joints"]], dtype=np.int32)
        GRIP = np.array([names.index(cfg["robot"]["gripper_joint"])], dtype=np.int32)
        GRIP_L = np.array([names.index(cfg["robot"]["gripper_joint_left"])], dtype=np.int32)
        LIFT = np.array([names.index(cfg["robot"]["lift_joint"])], dtype=np.int32)
        art.set_joint_positions(np.array([args.lift], np.float32), joint_indices=LIFT)
        base0 = np.asarray(art.get_world_pose()[0], dtype=float).ravel()
        q0 = np.asarray(door.get_world_pose()[1], dtype=float).ravel()
        xf = UsdGeom.XformCache()
        bar0 = np.array(cfg["door"]["handle_bar_world"], dtype=float)
        hinge = np.array([*cfg["door"]["hinge_xy"], 0.0])

        def door_deg():
            return math.degrees(runner.yaw_delta(np.asarray(door.get_world_pose()[1], dtype=float).ravel(), q0))

        def bar_now():
            a = math.radians(door_deg()); r = bar0 - hinge
            c, s = math.cos(a), math.sin(a)
            return hinge + np.array([c * r[0] - s * r[1], s * r[0] + c * r[1], r[2]])

        def pad_local(side, point):
            prim = stage.GetPrimAtPath(f"{robot}/gripper_r_rh_p12_rn_{side}")
            M = np.array(xf.GetLocalToWorldTransform(prim)).T
            return (np.linalg.inv(M) @ np.r_[point, 1.0])[:3]

        def contacts():
            bar_path = door_path + "/col_bar"
            headers, data = get_physx_simulation_interface().get_contact_report()
            total = np.zeros(3); n = 0; others = {}
            for h in headers:
                a, b = (str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1))
                if "gripper_" not in a + b:
                    continue
                impulse = np.zeros(3)
                for d in data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]:
                    impulse += np.asarray(d.impulse, dtype=float)
                if bar_path in (a, b):
                    total += impulse; n += 1
                else:
                    other = a if "gripper_" not in a else b
                    key = other.split("/")[-1] if "col_" in other else other.split("/")[4]
                    others[key] = round(others.get(key, 0.0) + float(np.linalg.norm(impulse)) * 60, 1)
            return n, float(np.linalg.norm(total)) * 60, others

        def send(q, grip):
            art.apply_action(ArticulationAction(joint_positions=np.asarray(q, np.float32), joint_indices=ARM))
            for idx in (GRIP, GRIP_L) if not args.no_left_drive else (GRIP,):
                art.apply_action(ArticulationAction(joint_positions=np.array([grip], np.float32), joint_indices=idx))
            art.apply_action(ArticulationAction(joint_positions=np.array([args.lift], np.float32), joint_indices=LIFT))

        g = cfg["gripper"]
        planner = HandlePullPlanner(robot_xy=tuple(cfg["robot"]["base_xy"]), lift=args.lift,
                                    back_grasp=float(g.get("grasp_back", 0.0)),
                                    dz_grasp=float(g.get("grasp_dz", 0.0)),
                                    pull_rep=int(g.get("pull_rep", 2)))
        home = np.asarray(cfg["robot"]["home_arm"], dtype=float)
        traj, errs = planner.plan(home, args.lift)
        print(f"[probe] 계획 오차 {errs}", flush=True)
        cur = np.asarray(art.get_joint_positions()).ravel()[ARM].astype(float)
        q = cur
        for name, qs, grips in [("home", [cur, home], [0.0, 0.0])] + runner.phases(traj):
            rows = runner.resample(np.column_stack(([q] + list(qs), [grips[0] * g["close_handle"]]
                                                    + [v * g["close_handle"] for v in grips])), 0.015)
            for k, row in enumerate(rows):
                send(row[:-1], row[-1])
                for _ in range(2):
                    sim.step(render=False)
                if name in ("grasp", "pull") and (k % args.every == 0 or k == len(rows) - 1):
                    xf.Clear()
                    bar = bar_now(); lr = pad_local("r2", bar); ll = pad_local("l2", bar)
                    nb, force, others = contacts()
                    j = np.asarray(art.get_joint_positions()).ravel()
                    th = float(j[GRIP[0]])
                    print(f"[{name:<6} {k:>3}/{len(rows)}] 문 {door_deg():6.2f}도  θ {th:.3f}"
                          f" (왼쪽 {float(j[GRIP_L[0]]):.3f})  막대 깊이 {lr[2] * 1000:+6.1f}mm"
                          f"  턱중앙 오차 {(abs(lr[1]) - abs(ll[1])) * 500:+5.1f}mm (반간격 {jaw_half(th) * 1000:.1f})"
                          f"  접촉 {nb}점 {force:5.1f}N"
                          f"{'  다른접촉 ' + str(others) if others else ''}", flush=True)
            for _ in range(10):
                sim.step(render=False)
            q = rows[-1, :-1]
            drift = float(np.linalg.norm(np.asarray(art.get_world_pose()[0], dtype=float).ravel() - base0))
            print(f"=== {name} 끝: 문 {door_deg():.2f}도, 베이스 이동 {drift * 1000:.2f} mm", flush=True)
            if name == "pull":
                break
        sim.stop()
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
