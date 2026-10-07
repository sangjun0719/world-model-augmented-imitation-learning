"""kimm_door v2 — 문 D8 열기. Isaac Sim 단독 (Isaac Lab / cyclo_lab 불필요).

하는 일은 네 가지뿐이다.
  1. 물리 시작 전에 몸통 높이를 고정하고 액추에이터 게인을 설정한다
  2. core/ 의 검증된 플래너로 궤적을 만든다
  3. 구간별로 흘려보내며 문 각도를 읽는다
  4. 문 각도 > success_deg 면 성공

사용:
  OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/run.py
  ... --wrist 17            # 손목 stiffness 덮어쓰기
  ... --door-start 20       # 문을 20도 열어 놓고 시작 (2단계만 볼 때)
  ... --phase pull|push     # 한 단계만
  ... --headless
  ... --selftest --headless # 몸통 고정만 120스텝 검사
  ... --keep-open          # GUI에 검증 결과를 남기고 창 유지
"""
import argparse
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "motion"))

from gripper_geom import jaw_half          # noqa: E402  (sys.path 설정 뒤라야 한다)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def yaw_delta(q, q0):
    """닫힘 자세 q0 대비 열린 각도(rad). mdp.door_angle 과 같은 식."""
    w, x, y, z = quat_mul(q, np.array([q0[0], -q0[1], -q0[2], -q0[3]]) / float(q0 @ q0))
    a = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (a + math.pi) % (2 * math.pi) - math.pi


def resample(path, dq):
    """팔과 그리퍼를 함께 보간한다. 정지 구간도 유지한다."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path
    seg = np.maximum(np.max(np.abs(np.diff(path, axis=0)), axis=1), dq)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(math.ceil(s[-1] / dq * 1.5)), 2)      # 1.5 = smoothstep 첨두 보정
    u = np.linspace(0, 1, n + 1)
    u = u * u * (3 - 2 * u)
    return np.stack([np.interp(u * s[-1], s, path[:, j]) for j in range(path.shape[1])], axis=1)


def phases(traj):
    """플래너 출력 [(이름,q,grip,lift)] 를 [(이름, qs, grips)] 로 묶는다."""
    out = []
    for name, q, grip, _ in traj:
        if name == "lift_down":          # 몸통은 시작부터 고정되어 있다
            continue
        if not out or out[-1][0] != name:
            out.append((name, [], []))
        out[-1][1].append(np.asarray(q, dtype=float))
        out[-1][2].append(float(grip))
    return out


def lock_lift(stage, robot_prim, joint_name, lift):
    """물리 시작 전 리프트의 초기 상태와 관절 한계를 같은 높이로 설정한다."""
    from pxr import Sdf, UsdPhysics

    prim = stage.GetPrimAtPath(f"{robot_prim}/joints/{joint_name}")
    joint = UsdPhysics.PrismaticJoint(prim)
    if not joint:
        raise RuntimeError(f"리프트 PrismaticJoint 없음: {prim.GetPath()}")
    lo, hi = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
    if not lo <= lift <= hi:
        raise ValueError(f"리프트 {lift} m가 관절 한계 [{lo}, {hi}] 밖이다")
    joint.CreateLowerLimitAttr().Set(float(lift))
    joint.CreateUpperLimitAttr().Set(float(lift))
    UsdPhysics.DriveAPI.Apply(prim, "linear").CreateTargetPositionAttr().Set(float(lift))
    for name, value in (("position", lift), ("velocity", 0.0)):
        prim.CreateAttribute(f"state:linear:physics:{name}", Sdf.ValueTypeNames.Float).Set(float(value))


def bind_friction(stage, mu, prims):
    """고무 패드 마찰 머티리얼을 만들어 지정 prim 들에 물린다 (physics 바인딩, 결합 방식 max).

    이 씬에는 물리 머티리얼이 하나도 없어 PhysX 기본값(마찰 0.5)으로 계산된다. RH-P12-RN 손가락은
    고무 패드라 0.5 보다 크다. 미끄러짐이 마찰 때문인지 구분하려면 명시적으로 넣고 비교해야 한다.
    """
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    path = "/World/pad_rubber"
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(mu))
    api.CreateDynamicFrictionAttr().Set(float(mu) * 0.9)
    api.CreateRestitutionAttr().Set(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr().Set("max")
    for prim_path in prims:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"마찰 바인딩 대상 prim 없음: {prim_path}")
        UsdShade.MaterialBindingAPI(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")
    return path


def drive_left_finger(stage, robot_prim, right="gripper_r_joint1", left="gripper_r_joint3"):
    """왼쪽 손가락 관절에 오른쪽과 같은 드라이브를 복사한다.

    실제 RH-P12-RN 은 한 모터가 링크로 양 손가락을 함께 구동한다. 그런데 이 USD 에는
    `gripper_r_joint1` 에만 드라이브가 있고 `joint3` 는 mimic 스프링으로만 따라간다.
    2026-10-07 실측: 손잡이를 물는 순간 joint1 0.976 / joint3 0.708 로 왼쪽이 0.27 rad 뒤처져
    손가락 한쪽만 막대에 닿았다(막대 접촉 r1/r2 뿐). 드라이브를 복사하면 양쪽이 같이 닫혀
    턱 중앙 오차가 11.4 mm -> 1.3 mm 로 줄고 클램프가 양면 접촉이 된다.
    """
    from pxr import UsdPhysics

    src = stage.GetPrimAtPath(f"{robot_prim}/joints/{right}")
    dst = stage.GetPrimAtPath(f"{robot_prim}/joints/{left}")
    if not src.IsValid() or not dst.IsValid():
        raise RuntimeError(f"그리퍼 관절 prim 없음: {right} / {left}")
    drive = UsdPhysics.DriveAPI.Apply(dst, "angular")
    gains = {}
    for name, attr in (("stiffness", "drive:angular:physics:stiffness"),
                       ("damping", "drive:angular:physics:damping"),
                       ("maxForce", "drive:angular:physics:maxForce")):
        value = src.GetAttribute(attr).Get()
        gains[name] = float(value)
        getattr(drive, f"Create{name[0].upper()}{name[1:]}Attr")().Set(float(value))
    drive.CreateTargetPositionAttr().Set(0.0)
    return gains


WHEEL_JOINTS = ("left_wheel_drive", "right_wheel_drive", "rear_wheel_drive",
                "left_wheel_steer", "right_wheel_steer", "rear_wheel_steer")


def lock_wheels(stage, robot_prim, joints=WHEEL_JOINTS):
    """물리 시작 전 바퀴 6관절을 0 에 잠근다.

    바퀴 구동 관절의 drive maxForce 가 1 N·m 뿐이라, 팔을 앞으로 뻗으면 베이스가 뒤로 굴러간다.
    2026-10-07 실측: 손잡이로 접근하는 동안 left/right/rear_wheel_drive 가 0.43 rad 돌아
    손이 목표보다 약 40 mm 뒤에 섰다. 그래서 손가락이 막대 끝만 걸치거나 아예 닿지 않았다
    (성공 에피소드의 joint_pos_target 을 그대로 재생해도 접촉이 0건이었던 이유).
    """
    from pxr import Sdf, UsdPhysics

    for name in joints:
        prim = stage.GetPrimAtPath(f"{robot_prim}/joints/{name}")
        joint = UsdPhysics.RevoluteJoint(prim)
        if not joint:
            raise RuntimeError(f"바퀴 RevoluteJoint 없음: {prim.GetPath()}")
        joint.CreateLowerLimitAttr().Set(0.0)
        joint.CreateUpperLimitAttr().Set(0.0)
        for key, value in (("position", 0.0), ("velocity", 0.0)):
            prim.CreateAttribute(f"state:angular:physics:{key}", Sdf.ValueTypeNames.Float).Set(float(value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "d8.yaml"))
    ap.add_argument("--lift", type=float, default=-0.25, choices=[-0.25, -0.45])
    ap.add_argument("--wrist", type=float, default=None, help="arm_r_joint7 stiffness 덮어쓰기")
    ap.add_argument("--wrist-damping", type=float, default=None)
    ap.add_argument("--door-start", type=float, default=0.0, help="시작 문 각도(도)")
    ap.add_argument("--phase", default="all", choices=["all", "pull", "push"])
    ap.add_argument("--dq", type=float, default=None,
                    help="한 스텝당 최대 관절 변화(rad). 미지정 시 config 의 motion.dq")
    ap.add_argument("--substeps", type=int, default=None,
                    help="목표 하나당 물리 스텝 수. 미지정 시 config 의 motion.substeps")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--keep-open", action="store_true", help="결과 확인 후 GUI 창을 닫을 때까지 유지")
    ap.add_argument("--capture-ui", default="", help="GUI 검증 화면 PNG 저장 경로")
    ap.add_argument("--damp-scale", type=float, default=None,
                    help="팔 관절 감쇠 배수. 1 이면 USD 원본(튜닝 전) 값 = 심한 부족감쇠")
    ap.add_argument("--close-handle", type=float, default=None,
                    help="gripper.close_handle 덮어쓰기. 낮추면 막대를 꽉 못 쥔다")
    ap.add_argument("--solver-pos", type=int, default=None, help="위치 솔버 반복 덮어쓰기")
    ap.add_argument("--solver-vel", type=int, default=None, help="속도 솔버 반복 덮어쓰기")
    ap.add_argument("--no-wheel-lock", action="store_true",
                    help="바퀴 잠금을 풀어 베이스 밀림 실패를 재현한다 (실패 에피소드 생성용)")
    ap.add_argument("--no-left-drive", action="store_true",
                    help="왼쪽 손가락 드라이브를 복사하지 않아 한쪽 파지 실패를 재현한다")
    ap.add_argument("--grasp-back", type=float, default=None,
                    help="gripper.grasp_back 덮어쓰기. 0 이면 손끝이 문짝을 민다")
    ap.add_argument("--no-grasp-gate", action="store_true",
                    help="파지 검사에 떨어져도 중단하지 않고 끝까지 간다. "
                         "실패 에피소드에 '당기다 놓치는' 구간을 남기려면 필요하다")
    ap.add_argument("--hdf5", default="",
                    help="에피소드를 HDF5 로 저장할 경로. kimm_record 스키마와 같아 to_act.py 가 먹는다. "
                         "카메라 이미지가 필요하므로 헤드리스에서도 렌더를 켠다")
    ap.add_argument("--hdf5-rate", type=float, default=30.0,
                    help="HDF5 샘플링 주기(Hz). 물리 60 Hz 의 약수여야 샘플 시각이 정확히 맞는다")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--save-camera", default="",
                    help="GUI 종료 시 현재 뷰포트 시점을 이 JSON 에 저장한다. "
                         "--keep-open 과 함께 쓰면 직접 돌려 맞춘 구도를 그대로 남길 수 있다")
    ap.add_argument("--telemetry", default="",
                    help="매 스텝 상세 기록 CSV 경로. 같은 이름의 .json 에 조건·결과를 남긴다")
    ap.add_argument("--telemetry-every", type=int, default=1,
                    help="몇 물리 스텝마다 한 줄 기록할지")
    ap.add_argument("--capture-video", default="",
                    help="프레임 PNG 를 쌓을 폴더. GUI 실행 전용. tools/make_video.sh 로 mp4 를 만든다")
    ap.add_argument("--capture-every", type=int, default=12,
                    help="--capture-video 사용 시 몇 물리 스텝마다 한 장 찍을지")
    ap.add_argument("--selftest", action="store_true", help="팔 궤적 실행 없이 몸통 고정만 120스텝 검사")
    ap.add_argument("--seed", type=int, default=None,
                    help="무작위 조건 seed. 같은 seed 는 같은 조건을 낸다")
    ap.add_argument("--rand-xy", type=float, default=None,
                    help="베이스 x,y 를 각각 +-이 값(m)만큼 균등 무작위. 0 이면 끔")
    ap.add_argument("--rand-door-deg", type=float, default=None,
                    help="문 초기 각도를 0~이 값(도)에서 균등 무작위")
    ap.add_argument("--base-tol-mm", type=float, default=None,
                    help="베이스 이동 허용치(mm). 미지정 시 config 의 guards.base_tol_mm")
    ap.add_argument("--lift-tol-mm", type=float, default=None,
                    help="몸통 높이 허용치(mm). 미지정 시 config 의 guards.lift_tol_mm")
    args = ap.parse_args()
    if (args.keep_open or args.capture_ui or args.capture_video) and args.headless:
        ap.error("--keep-open/--capture-ui/--capture-video는 GUI 실행에서만 사용할 수 있습니다")
    if args.capture_video and args.capture_every < 1:
        ap.error("--capture-every는 1 이상이어야 합니다")
    if not math.isfinite(args.door_start) or not 0 <= args.door_start <= 90:
        ap.error("--door-start는 0~90도여야 합니다")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg.get("motion") or {}
    if args.dq is None:
        args.dq = float(m.get("dq", 0.005))
    if args.substeps is None:
        args.substeps = int(m.get("substeps", 1))
    if not math.isfinite(args.dq) or args.dq <= 0 or args.substeps < 1:
        ap.error("--dq는 양수, --substeps는 1 이상이어야 합니다")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "window_width": 1400, "window_height": 900})
    exit_code = 1
    try:
        exit_code = run_sim(args, cfg, app)
        return exit_code
    except Exception as e:
        # app.close(exit_code=...) 는 fastShutdown 으로 프로세스를 즉시 죽인다. finally 안에서
        # 죽으므로 파이썬 기본 트레이스백이 찍히기 전에 끝난다. 여기서 먼저 찍어야 보인다.
        traceback.print_exc()
        print(f"[v2] 중단: {e}", flush=True)
        sys.stdout.flush(); sys.stderr.flush()
        return exit_code
    finally:
        app.close(exit_code=exit_code)


def run_sim(args, cfg, app):
    import omni.usd
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysicsSchemaTools, PhysxSchema

    scene = os.path.join(ROOT, cfg["scene"]["root"])
    print(f"[v2] 씬 {scene}", flush=True)
    if not omni.usd.get_context().open_stage(scene):
        raise RuntimeError(f"씬을 열 수 없습니다: {scene}")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    lock_lift(stage, cfg["scene"]["robot_prim"], cfg["robot"]["lift_joint"], args.lift)
    if args.no_wheel_lock:
        print("[v2] 바퀴 잠금 해제 (실패 모드)", flush=True)
    else:
        lock_wheels(stage, cfg["scene"]["robot_prim"], tuple(cfg["robot"].get("wheel_joints", WHEEL_JOINTS)))
    door_prim = stage.GetPrimAtPath(cfg["scene"]["door_prim"])
    PhysxSchema.PhysxContactReportAPI.Apply(door_prim).CreateThresholdAttr().Set(0.0)
    print(f"[v2] 몸통 시작 높이 {args.lift:+.2f} m, 리프트 관절 잠금, 바퀴 6관절 잠금", flush=True)

    # --- 1. 액추에이터 게인 덮어쓰기 (물리 시작 전) -----------------------------------
    if args.damp_scale is not None:
        # USD 원본 damping 에 배수를 적용한다. 1 이면 튜닝 전 값(감쇠비 ~0.06)으로 되돌린다.
        base = {"arm_r_joint1": 0.692229, "arm_r_joint2": 0.572235, "arm_r_joint3": 0.508447,
                "arm_r_joint4": 0.236879, "arm_r_joint5": 0.101985, "arm_r_joint6": 0.028322,
                "arm_r_joint7": 0.009684}
        cfg["actuators"] = {k: {"stiffness": None, "damping": v * args.damp_scale}
                            for k, v in base.items()}
        print(f"[v2] 감쇠 배수 {args.damp_scale} 적용 (실패 모드)", flush=True)
    if args.close_handle is not None:
        cfg["gripper"]["close_handle"] = float(args.close_handle)
        print(f"[v2] close_handle {args.close_handle} (실패 모드)", flush=True)
    if args.solver_pos is not None or args.solver_vel is not None:
        it = cfg.setdefault("solver", {})
        if args.solver_pos is not None: it["position_iterations"] = args.solver_pos
        if args.solver_vel is not None: it["velocity_iterations"] = args.solver_vel
    gains = dict(cfg.get("actuators") or {})
    if args.wrist is not None:
        gains.setdefault("arm_r_joint7", {})["stiffness"] = args.wrist
    if args.wrist_damping is not None:
        gains.setdefault("arm_r_joint7", {})["damping"] = args.wrist_damping
    for jn, g in gains.items():
        prim = stage.GetPrimAtPath(f"{cfg['scene']['robot_prim']}/joints/{jn}")
        if not prim.IsValid():
            print(f"[v2] 경고: {jn} prim 없음", flush=True)
            continue
        for key, attr in (("stiffness", "drive:angular:physics:stiffness"),
                          ("damping", "drive:angular:physics:damping")):
            if g.get(key) is None:
                continue
            a = prim.GetAttribute(attr)
            before = a.Get()
            a.Set(float(g[key]))
            print(f"[v2] {jn}.{key}: {before} -> {g[key]}", flush=True)

    # 접촉 솔버 반복. 기본값으로는 11 mm 막대를 물는 동안 접촉이 부정확해 파지가 쉽게 풀린다.
    it = cfg.get("solver") or {}
    art_api = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(cfg["scene"]["robot_prim"]))
    pos_it, vel_it = int(it.get("position_iterations", 64)), int(it.get("velocity_iterations", 16))
    art_api.CreateSolverPositionIterationCountAttr().Set(pos_it)
    art_api.CreateSolverVelocityIterationCountAttr().Set(vel_it)
    print(f"[v2] 솔버 반복: 위치 {pos_it}, 속도 {vel_it}", flush=True)

    mu = (cfg.get("friction") or {}).get("pad")
    if mu:
        bind_friction(stage, mu, [cfg["scene"]["robot_prim"], cfg["scene"]["door_prim"]])
        print(f"[v2] 마찰 머티리얼 바인딩: 정지 {mu}, 운동 {mu * 0.9:.2f} (결합 max)", flush=True)

    if cfg["gripper"].get("drive_left", True) and not args.no_left_drive:
        gains = drive_left_finger(stage, cfg["scene"]["robot_prim"])
        print(f"[v2] 왼쪽 손가락 드라이브 복사: {gains}", flush=True)

    sim = SimulationContext(physics_dt=1 / 60, rendering_dt=1 / 60)
    sim.initialize_physics()

    art = SingleArticulation(cfg["scene"]["robot_prim"]); art.initialize()
    door = SingleRigidPrim(cfg["scene"]["door_prim"]); door.initialize()

    # --- 무작위 조건 ---------------------------------------------------------------
    # 흔드는 것은 베이스 x,y 와 문 초기 각도뿐이다.
    # **베이스 yaw 는 흔들지 않는다.** HandlePullPlanner 가 w2r(..., yaw=0.0) 을 하드코딩한다
    # (core/kimm_door_traj.py:186,195). yaw 를 주면 당기기 계획이 로봇 기준계와 어긋난다.
    rnd = cfg.get("randomize") or {}
    rand_xy = float(args.rand_xy if args.rand_xy is not None else rnd.get("base_xy_m", 0.0))
    rand_door = float(args.rand_door_deg if args.rand_door_deg is not None
                      else rnd.get("door_start_deg", 0.0))
    base_xy = list(map(float, cfg["robot"]["base_xy"]))
    if rand_xy > 0 or rand_door > 0:
        rng = np.random.default_rng(args.seed)
        dx, dy = (rng.uniform(-rand_xy, rand_xy, 2) if rand_xy > 0 else (0.0, 0.0))
        base_xy = [base_xy[0] + float(dx), base_xy[1] + float(dy)]
        if rand_door > 0:
            args.door_start = float(rng.uniform(0.0, rand_door))
        p, o = art.get_world_pose()
        p = np.asarray(p, dtype=float).ravel()
        art.set_world_pose(position=np.array([p[0] + float(dx), p[1] + float(dy), p[2]]),
                           orientation=np.asarray(o, dtype=float).ravel())
        print(f"[v2] 무작위 조건 seed={args.seed}: 베이스 {dx * 1000:+.1f}, {dy * 1000:+.1f} mm "
              f"-> ({base_xy[0]:.4f}, {base_xy[1]:.4f}), 문 시작 {args.door_start:.2f} deg", flush=True)
    names = list(art.dof_names)
    ARM = np.array([names.index(n) for n in cfg["robot"]["arm_joints"]], dtype=np.int32)
    GRIP = np.array([names.index(cfg["robot"]["gripper_joint"])], dtype=np.int32)
    GRIP_L = (np.array([names.index(cfg["robot"]["gripper_joint_left"])], dtype=np.int32)
              if cfg["gripper"].get("drive_left", True) and not args.no_left_drive else None)
    LIFT = np.array([names.index(cfg["robot"]["lift_joint"])], dtype=np.int32)
    art.set_joint_positions(np.array([args.lift], np.float32), joint_indices=LIFT)
    art.set_joint_velocities(np.zeros(1, np.float32), joint_indices=LIFT)
    art.apply_action(ArticulationAction(joint_positions=np.array([args.lift], np.float32), joint_indices=LIFT))

    base0 = np.asarray(art.get_world_pose()[0], dtype=float).ravel()   # 바퀴 잠금 확인용 기준 위치
    p0, q0 = (np.asarray(v, dtype=float).ravel() for v in door.get_world_pose())

    def door_deg():
        q = np.asarray(door.get_world_pose()[1], dtype=float).ravel()
        return math.degrees(yaw_delta(q, q0))

    def handle_contacts():
        """문짝 접촉과 구분해 손잡이 막대에 닿은 그리퍼 링크만 읽는다."""
        bar = cfg["scene"]["door_prim"] + "/col_bar"
        hits = set()
        headers, _ = get_physx_simulation_interface().get_contact_report()
        for h in headers:
            if h.type == ContactEventType.CONTACT_LOST:
                continue
            a, b = (str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1))
            other = b if a == bar else (a if b == bar else "")
            if "/gripper_r_" in other:
                hits.add(other)
        return hits

    labels = {}
    # 카메라는 **녹화 여부와 무관하게** 항상 설정한다. 설정하지 않으면 Kit 기본 카메라
    # (5,5,5 -> 원점)가 그대로 쓰여 영상 구도가 의도와 달라진다 (2026-10-07 실제로 그랬다).
    if not args.headless:
        from isaacsim.core.utils.viewports import set_camera_view
        cam = cfg.get("camera") or {}
        eye = np.array(cam.get("eye", [-4.25, -4.25, 2.25]), dtype=float)
        tgt = np.array(cam.get("target", [-2.55, -3.10, 0.85]), dtype=float)
        set_camera_view(eye=eye, target=tgt)
        print(f"[v2] 카메라 고정: eye {list(eye)} -> target {list(tgt)}", flush=True)
        # 뷰포트 위젯이 작아지면 렌더 해상도가 320x240 까지 떨어지고, 그러면 DLSS 최소 입력(300)
        # 미만이라 뷰포트가 **검게** 나온다. 캡처할 때는 위젯 크기와 무관하게 고정한다.
        if args.capture_video or args.capture_ui:
            try:
                from omni.kit.viewport.utility import get_active_viewport
                get_active_viewport().resolution = (1280, 720)
                print("[v2] 뷰포트 렌더 해상도 고정: 1280x720", flush=True)
            except Exception as e:
                print(f"[v2] 경고: 뷰포트 해상도 고정 실패 ({e})", flush=True)
        if args.capture_video:
            # RTX 는 수렴에 몇 프레임이 걸린다. 바로 캡처하면 **영상 앞부분이 검게** 나온다.
            # 물리를 돌리기 전에 렌더만 충분히 돌려 장면을 띄워 둔다.
            for _ in range(90):
                app.update()
            print("[v2] 렌더러 예열 90 프레임", flush=True)

    # 녹화 중에는 검증 패널을 띄우지 않는다. capture_viewport_to_file 은 뷰포트만 담으므로
    # 패널은 어차피 영상에 들어가지 않는데, 패널이 떠 있으면 뷰포트가 검게 캡처된다
    # (2026-10-07, 패널 있는 캡처 5회 전부 검정 / 없는 캡처 전부 정상).
    # 수치는 tools/overlay_video.py 가 telemetry CSV 를 읽어 프레임에 입힌다.
    if not args.headless and not args.capture_video:
        import omni.ui as ui
        # 접혀 있으면 녹화 영상에 수치가 안 담긴다. 6.1 의 omni.ui.Window 에는 collapsed 속성이
        # 없으므로 창 플래그로 접기를 막는다. 플래그 이름이 없는 빌드에서도 죽지 않게 처리한다.
        # Kit 은 창의 접힘/닫힘 상태를 세션 간에 저장한다. 그대로 두면 이전 실행에서 접힌 상태가
        # 복원돼 녹화 영상에 수치가 안 담긴다. NO_SAVED_SETTINGS 로 매번 새로 띄운다.
        flags = (getattr(ui, "WINDOW_FLAGS_NO_COLLAPSE", 0)
                 | getattr(ui, "WINDOW_FLAGS_NO_SAVED_SETTINGS", 0))
        # 뷰포트(화면 왼쪽)를 가리지 않도록 오른쪽 Stage 패널 위에 띄운다.
        window = ui.Window("D8 Door Validation", width=405, height=430, flags=flags)
        window.position_x = 988
        window.position_y = 30
        window.visible = True
        with window.frame:
            with ui.VStack(spacing=6):
                labels["status"] = ui.Label("Starting", height=30, style={"font_size": 20})
                labels["phase"] = ui.Label("Phase: --", height=22)
                labels["door"] = ui.Label("Door: 0.00 deg / target > 80 deg", height=22)
                labels["grasp"] = ui.Label("Bar depth: -- / hand-bar: --", height=22)
                labels["jaw"] = ui.Label("Jaw half-gap: --", height=22)
                labels["force"] = ui.Label("Bar contact: --", height=22)
                labels["grip"] = ui.Label("Gripper: --", height=22)
                labels["lift"] = ui.Label(f"Torso locked at {args.lift:+.2f} m", height=22)
                labels["cond"] = ui.Label("Condition: --", height=22)
                labels["contact"] = ui.Label("Handle contact: --", height=22)

    max_lift_error = 0.0
    max_base_drift = 0.0
    # 안전 가드. 바퀴가 굴러 베이스가 밀리는 사고(2026-10-07 실측 40 mm)를 잡는 것이 목적이다.
    guards = cfg.get("guards") or {}
    lift_tol = float(args.lift_tol_mm if args.lift_tol_mm is not None
                     else guards.get("lift_tol_mm", 1.0)) / 1000.0
    base_tol = float(args.base_tol_mm if args.base_tol_mm is not None
                     else guards.get("base_tol_mm", 2.0)) / 1000.0

    def update_ui(status=None):
        nonlocal max_lift_error, max_base_drift
        joints = np.asarray(art.get_joint_positions()).ravel()
        lift = float(joints[LIFT[0]])
        max_lift_error = max(max_lift_error, abs(lift - args.lift))
        base = np.asarray(art.get_world_pose()[0], dtype=float).ravel()
        max_base_drift = max(max_base_drift, float(np.linalg.norm(base - base0)))
        if labels:
            if status is not None:
                labels["status"].text = status
            labels["door"].text = f"Door: {door_deg():.2f} deg / target > {cfg['door']['success_deg']:g} deg"
            labels["lift"].text = (f"Torso: {lift:+.4f} m / drift {max_lift_error * 1000:.3f} mm, "
                                   f"base {max_base_drift * 1000:.1f} mm")
            lo, hi = cfg["gripper"]["grasp_range"]
            gl = joints[GRIP_L[0]] if GRIP_L is not None else float("nan")
            labels["grip"].text = (f"Gripper: cmd {tel['tgt_g']:.3f} / act R {joints[GRIP[0]]:.3f} "
                                   f"L {gl:.3f} rad / grasp {lo:.2f}-{hi:.2f}")
            contacts = handle_contacts()
            jaws = sum(any(f"_rh_p12_rn_{side}" in path for path in contacts) for side in ("l", "r"))
            labels["contact"].text = f"Handle contact: {jaws}/2 jaws"
            labels["phase"].text = f"Phase: {tel['phase']}  step {tel['step']}"
            labels["cond"].text = (f"seed {args.seed} | base ({base_xy[0]:.3f}, {base_xy[1]:.3f})"
                                   f" | door0 {args.door_start:.2f}d | lift {args.lift:+.2f}")
            if tel["obj"] is not None:                 # CSV 와 같은 측정값을 화면에도 띄운다
                try:
                    t = tel["obj"]; t.xf.Clear()
                    bar = t._bar_now()
                    lr, ll = t._pad_local("r2", bar), t._pad_local("l2", bar)
                    n, force, others = t._contacts()
                    labels["grasp"].text = (f"Bar depth: {lr[2] * 1000:+.1f} mm   "
                                            f"hand-bar: {np.linalg.norm(lr) * 1000:.1f} mm   "
                                            f"jaw center err: "
                                            f"{(abs(lr[1]) - abs(ll[1])) * 500:+.1f} mm")
                    labels["jaw"].text = (f"Jaw half-gap: "
                                          f"{jaw_half(float(joints[GRIP[0]])) * 1000:.1f} mm")
                    labels["force"].text = (f"Bar contact: {n} pts, {force:.1f} N"
                                            + (f"   other: {others}" if others else ""))
                except Exception:
                    pass
        return lift

    def save_camera():
        """현재 뷰포트 시점을 config 에 넣을 수 있는 eye/target 으로 저장한다.

        Kit 은 뷰포트 카메라를 디스크에 남기지 않는다. 직접 돌려 맞춘 구도를 다음 실행에서
        다시 쓰려면 종료 시점에 이렇게 꺼내 두어야 한다.
        """
        if not args.save_camera or args.headless:
            return
        try:
            from omni.kit.viewport.utility import get_active_viewport
            from pxr import UsdGeom
            vp = get_active_viewport()
            prim = stage.GetPrimAtPath(str(vp.camera_path))
            M = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
            e = M[:3, 3]
            fwd = -M[:3, 2]                      # USD 카메라는 -Z 를 본다
            bar = np.array(cfg["door"]["handle_bar_world"], dtype=float)
            d = float(np.dot(bar - e, fwd))      # 손잡이까지의 거리로 target 을 잡아 구도를 보존
            t = e + fwd * max(d, 0.5)
            import json
            with open(args.save_camera, "w", encoding="utf-8") as f:
                json.dump({"eye": [round(float(v), 4) for v in e],
                           "target": [round(float(v), 4) for v in t]}, f, indent=2)
            print(f"[v2] 시점 저장: {args.save_camera}\n"
                  f"       eye:    [{e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}]\n"
                  f"       target: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]", flush=True)
        except Exception as ex:
            print(f"[v2] 경고: 시점 저장 실패 ({ex})", flush=True)

    def finish(status, code):
        update_ui(status)
        print(f"[v2] {status}; 몸통 최대 오차 {max_lift_error * 1000:.3f} mm, "
              f"베이스 최대 이동 {max_base_drift * 1000:.3f} mm", flush=True)
        if tel["fh"] is not None:
            tel["fh"].flush(); tel["fh"].close(); tel["fh"] = None
            import json
            meta = {
                "status": status, "exit_code": code, "success": code == 0,
                "final_door_deg": round(door_deg(), 4),
                "success_deg": cfg["door"]["success_deg"],
                "max_lift_error_mm": round(max_lift_error * 1000, 4),
                "max_base_drift_mm": round(max_base_drift * 1000, 4),
                "steps_logged": tel["step"], "telemetry_every": args.telemetry_every,
                "condition": {
                    "seed": args.seed, "rand_xy_m": rand_xy, "rand_door_deg": rand_door,
                    "base_xy": base_xy, "base_xy_nominal": list(cfg["robot"]["base_xy"]),
                    "door_start_deg": args.door_start, "lift": args.lift,
                },
                "motion": {"dq": args.dq, "substeps": args.substeps,
                           "physics_hz": 60.0, "push_stop_deg": cfg["door"].get("push_stop_deg")},
                "solver": cfg.get("solver"), "actuators": cfg.get("actuators"),
                "gripper": cfg.get("gripper"), "guards": cfg.get("guards"),
                "scene": cfg["scene"]["root"],
            }
            p = os.path.splitext(args.telemetry)[0] + ".json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[v2] 조건·결과: {p}", flush=True)
        if ep["w"] is not None:
            p = ep["w"].save(args.hdf5, code == 0, {
                "final_door_deg": round(door_deg(), 4),
                "success_deg": cfg["door"]["success_deg"],
                "status": status,
                "max_base_drift_mm": round(max_base_drift * 1000, 4),
                "max_lift_error_mm": round(max_lift_error * 1000, 4),
                "scene_usd": cfg["scene"]["root"],
                "seed": -1 if args.seed is None else int(args.seed),
                "rand_xy_m": rand_xy, "rand_door_deg": rand_door,
                "base_xy": base_xy, "door_start_deg": args.door_start, "lift": args.lift,
                "dq": args.dq, "substeps": args.substeps,
                "push_stop_deg": cfg["door"].get("push_stop_deg"),
                "gripper": cfg.get("gripper"), "actuators": cfg.get("actuators"),
                "solver": cfg.get("solver"), "guards": cfg.get("guards"),
            })
            print(f"[v2] HDF5 저장: {p} ({ep['n']} 샘플)", flush=True)
        if args.capture_video and frames_idx:
            # 프레임 n 이 telemetry 의 어느 step 인지. 오버레이가 이걸로 수치를 찾는다.
            with open(os.path.join(args.capture_video, "frames.csv"), "w", encoding="utf-8") as f:
                f.write("frame,step\n")
                for i, s in frames_idx:
                    f.write(f"{i},{s}\n")
        sim.pause()                    # 최종 자세를 유지하고 결과를 보여준다
        if args.capture_ui:
            capture_ui(args.capture_ui)
        if args.keep_open:
            print("[v2] GUI 결과 표시 중. 창을 닫으면 종료됩니다.", flush=True)
            while app.is_running():
                app.update()
                time.sleep(0.02)
        save_camera()
        sim.stop()
        return code

    def capture_ui(path):
        import omni.appwindow
        from omni.kit.renderer_capture import acquire_renderer_capture_interface

        for _ in range(5):
            app.update()
        interface = acquire_renderer_capture_interface()
        app_window = omni.appwindow.get_default_app_window()
        interface.capture_next_frame_swapchain(os.path.abspath(path), app_window)
        app.update()
        interface.wait_async_capture(app_window)
        print(f"[v2] GUI 캡처: {os.path.abspath(path)}", flush=True)

    def plan(fn, status):
        update_ui(status)
        if args.headless:
            return fn()
        # numpy IK 계산 중에도 GUI가 응답하도록, 물리는 멈추고 계산만 별도 스레드에서 한다.
        sim.pause()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(fn)
                while not future.done():
                    if not app.is_running():
                        raise RuntimeError("시뮬레이터가 종료되었습니다")
                    app.update()
                    time.sleep(0.02)
                return future.result()
        finally:
            if app.is_running():
                sim.play()

    # --- 영상 프레임 캡처 ----------------------------------------------------------
    # 데스크톱 스크린샷은 GNOME 이 막는다(org.gnome.Shell.Screenshot: not allowed). Kit 스왑체인을
    # 직접 떠서 저장하면 컴포지터를 거치지 않고 창에 보이는 그대로가 나온다.
    frame_n = [0]
    frames_idx = []                # (프레임 번호, 그때의 telemetry step) — 오버레이가 쓴다

    def grab_frame():
        """뷰포트(3D)만 직접 캡처한다.

        앱 창 스왑체인(capture_next_frame_swapchain)은 검증 패널이 펼쳐져 있으면 뷰포트가
        합성되기 전 프레임을 잡아 **3D 가 검게 나온다** (2026-10-07 4회 재현). 뷰포트
        렌더 프로덕트를 직접 뜨면 UI 상태와 무관하게 항상 장면이 들어온다.
        수치는 tools/overlay_video.py 가 telemetry CSV 를 읽어 프레임에 입힌다.
        """
        if not args.capture_video:
            return
        frame_n[0] += 1
        if frame_n[0] % args.capture_every:
            return
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        idx = frame_n[0] // args.capture_every
        path = os.path.join(args.capture_video, f"f{idx:06d}.png")
        capture_viewport_to_file(get_active_viewport(), os.path.abspath(path))
        frames_idx.append((idx, tel["step"]))
        app.update()

    # --- 상세 기록 -----------------------------------------------------------------
    tel = {"obj": None, "fh": None, "w": None, "step": 0, "phase": "init",
           "tgt_q": np.zeros(len(ARM)), "tgt_g": 0.0}
    # 측정기는 CSV 를 쓰지 않더라도 GUI 패널 수치와 HDF5 의 auxiliary 에 필요하다.
    # --hdf5 를 빼먹으면 헤드리스 배치에서 문 각도·막대 깊이가 통째로 빠진다.
    if args.telemetry or args.hdf5 or not args.headless:
        from telemetry import Telemetry
        tel["obj"] = Telemetry(stage, art, door, cfg, door_deg, base0, args.lift,
                               ARM, GRIP, GRIP_L, LIFT)
    if args.telemetry:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.telemetry)) or ".", exist_ok=True)
        tel["fh"] = open(args.telemetry, "w", newline="", encoding="utf-8")
        tel["w"] = csv.writer(tel["fh"])
        tel["w"].writerow(tel["obj"].header())
        print(f"[v2] 상세 기록: {args.telemetry} ({args.telemetry_every} 스텝마다)", flush=True)

    def log_row():
        tel["step"] += 1                   # 패널에도 쓰므로 CSV 없이도 센다
        if tel["w"] is None or tel["step"] % args.telemetry_every:
            return
        tel["w"].writerow(tel["obj"].row(tel["step"], float(sim.current_time),
                                         tel["phase"], tel["tgt_q"], tel["tgt_g"]))

    # --- HDF5 에피소드 ---------------------------------------------------------------
    ep = {"w": None, "every": 1, "n": 0, "tgt": None}
    if args.hdf5:
        from episode_writer import EpisodeWriter
        every = int(round(60.0 / args.hdf5_rate))
        if abs(60.0 / every - args.hdf5_rate) > 1e-6:
            print(f"[v2] 경고: {args.hdf5_rate} Hz 는 물리 60 Hz 의 약수가 아니다. "
                  f"{60.0 / every:.3f} Hz 로 기록한다", flush=True)
        ep["every"] = every
        ep["w"] = EpisodeWriter(cfg, art, jpeg_quality=args.jpeg_quality,
                                rate_hz=60.0 / every)
        ep["tgt"] = np.asarray(art.get_joint_positions(), np.float32).ravel().copy()
        print(f"[v2] HDF5 기록: {args.hdf5} ({60.0 / every:.1f} Hz, "
              f"카메라 {', '.join(ep['w'].camera_keys())})", flush=True)

    def step():
        if not app.is_running():
            raise RuntimeError("시뮬레이터가 종료되었습니다")
        # 물리는 60 Hz, 기록은 30 Hz 다. 매 스텝 렌더하면 **절반은 아무도 읽지 않고 버려진다**.
        # 카메라 렌더 프로덕트는 렌더가 돌아야 갱신되므로, 샘플을 뜨는 스텝에만 렌더한다.
        # 물리 계산은 렌더와 무관하므로 동작 결과는 바뀌지 않는다.
        need_img = False
        if ep["w"] is not None:
            tel["step_hdf5"] = tel.get("step_hdf5", 0) + 1
            need_img = tel["step_hdf5"] % ep["every"] == 0
        sim.step(render=(not args.headless) or need_img)
        log_row()
        grab_frame()
        if ep["w"] is not None:
            if need_img:
                aux = {}
                if tel["obj"] is not None:
                    r = tel["obj"].row(tel["step"], float(sim.current_time), tel["phase"],
                                       tel["tgt_q"], tel["tgt_g"])
                    h = tel["obj"].header()
                    for k in ("door_deg", "door_vel_deg_s", "bar_depth_mm", "jaw_center_err_mm",
                              "jaw_half_gap_mm", "hand_bar_dist_mm", "contact_n",
                              "contact_force_N", "base_drift_mm", "lift_err_mm"):
                        aux[k] = float(r[h.index(k)])
                if ep["w"].sample(float(sim.current_time), ep["tgt"], aux):
                    ep["n"] += 1
        actual = update_ui()
        if not math.isfinite(actual) or abs(actual - args.lift) > lift_tol:
            raise RuntimeError(f"몸통 고정 실패: lift={actual:+.5f} m, 목표={args.lift:+.2f} m "
                               f"(허용 {lift_tol * 1000:.1f} mm)")
        if max_base_drift > base_tol:
            raise RuntimeError(f"베이스가 움직였다: {max_base_drift * 1000:.1f} mm "
                               f"(허용 {base_tol * 1000:.1f} mm, 바퀴 잠금 확인)")

    if args.selftest:
        # 초기 팔 자세를 유지한 채 몸통이 처지지 않는지만 확인한다.
        art.apply_action(ArticulationAction(joint_positions=np.asarray(art.get_joint_positions(), np.float32)))
        for _ in range(120):
            step()
        print(f"[v2] 몸통·베이스 고정 검사 통과: {args.lift:+.2f} m, 120스텝, 리프트 오차 <= 1 mm, "
              f"베이스 이동 {max_base_drift * 1000:.3f} mm (<= 2 mm)", flush=True)
        return finish("PASS: torso lock", 0)

    if abs(args.door_start) > 1e-6:
        a = math.radians(args.door_start)
        h = np.array([*cfg["door"]["hinge_xy"], 0.0])
        r = p0 - h
        c, s_ = math.cos(a), math.sin(a)
        door.set_world_pose(position=h + np.array([c*r[0]-s_*r[1], s_*r[0]+c*r[1], r[2]]),
                            orientation=quat_mul(np.array([math.cos(a/2), 0, 0, math.sin(a/2)]), q0))
        door.set_linear_velocity(np.zeros(3)); door.set_angular_velocity(np.zeros(3))
        for _ in range(30):
            step()

    def send(q, grip):
        art.apply_action(ArticulationAction(joint_positions=np.asarray(q, np.float32), joint_indices=ARM))
        art.apply_action(ArticulationAction(joint_positions=np.array([grip], np.float32), joint_indices=GRIP))
        if GRIP_L is not None:            # 양 손가락을 같은 목표로 (한 모터 구동 재현)
            art.apply_action(ArticulationAction(joint_positions=np.array([grip], np.float32), joint_indices=GRIP_L))
        art.apply_action(ArticulationAction(joint_positions=np.array([args.lift], np.float32), joint_indices=LIFT))
        if ep["tgt"] is not None:         # HDF5 의 action = 실제로 보낸 관절 위치 목표
            ep["tgt"][ARM] = np.asarray(q, np.float32).ravel()
            ep["tgt"][GRIP] = grip
            if GRIP_L is not None:
                ep["tgt"][GRIP_L] = grip
            ep["tgt"][LIFT] = args.lift

    def run(name, qs, grips, close, stop_when=None):
        update_ui(f"Running: {name}")
        tel["phase"] = name
        # v1처럼 그리퍼도 같은 시간축으로 보간한다 (release/hold가 사라지지 않게).
        joined = np.column_stack((np.asarray(qs), np.asarray(grips) * close))
        rt = resample(joined, args.dq)
        last = rt[-1, :-1]
        for row in rt:
            tel["tgt_q"], tel["tgt_g"] = row[:-1], row[-1]
            send(row[:-1], row[-1])
            for _ in range(args.substeps):
                step()
            if stop_when is not None and stop_when():
                # 계획된 원호를 끝까지 따라가면 문이 힌지 limit(90도)에 닿은 뒤에도 계속 민다.
                # 그 반력이 베이스를 밀어낸다. 목표에 닿으면 남은 궤적을 버린다.
                last = row[:-1]
                print(f"[v2] {name} 조기 종료: 문 {door_deg():.2f} deg 도달, 남은 궤적 버림", flush=True)
                break
        for _ in range(10):                    # 구간 끝 정지
            step()
        gj = float(np.asarray(art.get_joint_positions()).ravel()[GRIP[0]])
        wj = float(np.asarray(art.get_joint_positions()).ravel()[ARM[6]])
        print(f"[v2] {name:<10} {len(rt):>4} step  문 {door_deg():6.2f} deg  "
              f"grip {gj:.3f}  손목 {wj:+.3f}", flush=True)
        return last

    # --- 2. 홈 자세 ----------------------------------------------------------------
    if args.capture_ui:
        sim.pause()
        base, ext = os.path.splitext(args.capture_ui)
        capture_ui(base + "_start" + (ext or ".png"))
        sim.play()
    home = np.asarray(cfg["robot"]["home_arm"], dtype=float)
    cur = np.asarray(art.get_joint_positions()).ravel()[ARM].astype(float)
    run("home", [cur, home], [0.0, 0.0], cfg["gripper"]["close_handle"])

    preset = cfg["lift_presets"][args.lift]
    xy = tuple(base_xy)            # 무작위 조건이 적용된 값. 로봇 프림도 같이 옮겨져 있다.
    from kimm_door_traj import HandlePullPlanner, RightArmDoorPlanner

    # --- 3. 1단계: 손잡이 잡고 당기기 ------------------------------------------------
    q = home
    if args.phase in ("all", "pull"):
        g = cfg["gripper"]
        traj, errs = plan(lambda: HandlePullPlanner(robot_xy=xy, lift=args.lift,
                                                    back_grasp=float(args.grasp_back if args.grasp_back is not None
                                                                     else g.get("grasp_back", 0.0)),
                                                    dz_grasp=float(g.get("grasp_dz", 0.0)),
                                                    pull_rep=int(g.get("pull_rep", 2))).plan(home, args.lift),
                          "Planning: handle pull")
        print(f"[v2] 1단계 계획 오차 {errs}", flush=True)
        for name, qs, grips in phases(traj):
            q = run(name, [q] + qs, [grips[0]] + grips, cfg["gripper"]["close_handle"])
            if name == "grasp":
                actual_grip = float(np.asarray(art.get_joint_positions()).ravel()[GRIP[0]])
                lo, hi = cfg["gripper"]["grasp_range"]
                contacts = handle_contacts()
                both_jaws = all(any(f"_rh_p12_rn_{side}" in path for path in contacts)
                                for side in ("l", "r"))
                print(f"[v2] 파지 검사: grip={actual_grip:.3f}, 범위=[{lo}, {hi}], "
                      f"손잡이 접촉={sorted(contacts)}", flush=True)
                if not both_jaws or not lo <= actual_grip <= hi:
                    if not args.headless:
                        set_camera_view(eye=np.array([-2.95, -3.65, 1.20]),
                                        target=np.array(cfg["door"]["handle_bar_world"]))
                    if args.no_grasp_gate:
                        print("[v2] 파지 검사 실패 — --no-grasp-gate 라 계속 진행한다", flush=True)
                    else:
                        return finish("FAIL: gripper did not grasp handle", 1)
        q = run("back", [q, home], [0.0, 0.0], cfg["gripper"]["close_handle"])

    # --- 4. 2단계: 틈에 넣어 밀기 ----------------------------------------------------
    if args.phase in ("all", "push"):
        a = door_deg()
        need = cfg["door"]["push_requires_open_deg"]
        if a < need:
            print(f"[v2] 실패: 문이 {a:.1f} deg 라 밀기 중단 (최소 {need} deg). "
                  "손잡이 당기기 결과를 먼저 확인하세요.", flush=True)
            return finish("FAIL: pull did not open door to 15 deg", 1)

        def plan_push():
            best = None
            for phi in [p for p in (a, a - 2, a - 4, 28, 24) if 15 <= p <= min(max(a, 15), 28.5)] or [15.0]:
                for r_pre in (0.60, 0.52, 0.48):
                    t, e = RightArmDoorPlanner(robot_xy=xy, lift_push=args.lift, r_pre=r_pre,
                                               z=preset["push_z"], k=preset["push_k"]).plan(q, args.lift, phi0=phi)
                    sc = e["descend"] + e["push"] + e["insert"] + sum(e["pre_ik"])
                    if best is None or sc < best[0]:
                        best = (sc, t, dict(e, phi=round(phi, 1), r_pre=r_pre))
                if best[0] < 10:
                    break
            return best

        best = plan(plan_push, "Planning: door push")
        print(f"[v2] 2단계 계획 (문 {a:.1f} deg) 오차 {best[2]}", flush=True)
        # 힌지 limit 은 90 도인데 플래너는 phi_end 90 도까지 원호를 만든다. 끝까지 따라가면
        # 문이 멈춘 뒤에도 계속 밀어 베이스가 밀려난다 (2026-10-07 영상에서 확인, 7.57 mm).
        stop_at = float(cfg["door"].get("push_stop_deg", cfg["door"]["success_deg"]))
        stopped = False
        for name, qs, grips in phases(best[1]):
            q = run(name, [q] + qs, [grips[0]] + grips, cfg["gripper"]["close_push"],
                    stop_when=(lambda: door_deg() >= stop_at))
            if door_deg() >= stop_at:
                stopped = True
                break
        if stopped:
            # 목표에 닿았으면 남은 구간을 실행하지 않고 **현재 자세 그대로** 붙잡는다.
            j = np.asarray(art.get_joint_positions()).ravel()
            hold_q, hold_g = j[ARM].astype(float), float(j[GRIP[0]])
            for _ in range(30):
                send(hold_q, hold_g)
                step()
            print(f"[v2] hold(정지)   30 step  문 {door_deg():6.2f} deg  "
                  f"grip {float(np.asarray(art.get_joint_positions()).ravel()[GRIP[0]]):.3f}", flush=True)

    final = door_deg()
    ok = final > cfg["door"]["success_deg"]
    print(f"[v2] === {'성공' if ok else '실패'}  최종 문 {final:.2f} deg "
          f"(기준 > {cfg['door']['success_deg']})", flush=True)
    return finish("PASS: door open" if ok else "FAIL: door below target", 0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
