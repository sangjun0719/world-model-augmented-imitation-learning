"""문 D8이 닫힘(0°)/1°/3° 에서 자유롭게 도는지: 각속도를 주고 각도 변화를 본다 (로봇 정지)."""
import argparse, math
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(ap); args = ap.parse_args()
app = AppLauncher(vars(args)).app
import gymnasium as gym, torch
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
import cyclo_lab  # noqa
from cyclo_lab.manager_based.manipulation.kimm_door.config.ffw_sg2 import mdp as dm
T = "Cyclo-KIMM-Door-FFW-SG2-v0"
cfg = parse_env_cfg(T, device=args.device, num_envs=1); cfg.init_action_cfg("record")
cfg.terminations.time_out = None
cfg.terminations.success = TerminationTermCfg(func=lambda env: torch.zeros(1, dtype=torch.bool, device=env.device))
cfg.recorders.dataset_export_dir_path = "/tmp/isaaclab_tools"; cfg.recorders.dataset_filename = "doorfree"
env = gym.make(T, cfg=cfg).unwrapped; robot = env.scene["robot"]; door = env.scene["door"]
terms, dims = env.action_manager.active_terms, env.action_manager.action_term_dim
tj = {t: env.action_manager.get_term(t)._joint_ids for t in terms}
dc = SceneEntityCfg("door")
for a0 in (0.0, 1.0, 3.0):
    env.reset()
    ids = torch.tensor([0], device=env.device)
    dm.reset_door_ajar(env, ids, dc, (-2.212, -3.463), (a0, a0))
    base = robot.data.joint_pos[0].clone()
    act = torch.cat([base[tj[t]].reshape(-1)[:d] for t, d in zip(terms, dims)]).unsqueeze(0)
    v = torch.zeros(1, 6, device=env.device); v[0, 5] = -0.5   # 경첩 둘레로 여는 방향(-z yaw?) 둘 다 시험
    for sgn in (1, -1):
        dm.reset_door_ajar(env, ids, dc, (-2.212, -3.463), (a0, a0))
        vv = v.clone(); vv[0, 5] *= sgn
        door.write_root_velocity_to_sim(vv)
        for _ in range(10): env.step(act)
        print(f"[free] start {a0} deg, yaw vel {vv[0,5].item():+.1f}: angle after 0.5s {math.degrees(float(dm.door_angle(env, dc)[0])):.2f} deg", flush=True)
env.close(); app.close()
