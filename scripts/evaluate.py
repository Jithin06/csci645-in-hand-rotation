"""Fixed-protocol evaluation of a trained hand-cube rotation checkpoint.

Runs the deterministic policy (action mean, no exploration noise) for exactly one
full episode (max 400 control steps = 20 s) in each of N parallel environments,
repeated for a fixed list of seeds, under one or more evaluation conditions:

  nominal           training domain-randomization ranges (in-distribution)
  heldout_mass      cube mass scale 1.5-2.0x  (training range: 0.7-1.4x)
  heldout_friction  hand/cube friction 0.35-0.55 (training range: 0.6-1.4)

Everything except the checkpoint (and the actor history length it needs) is
identical across checkpoints, so baseline and modifications are compared under
the same conditions. Observation noise and the reward curriculum are disabled,
as in scripts/play.py. Metrics do not depend on reward weights or the reward's
drift gate, so a reward modification cannot change how it is scored.

Per-episode metrics:
  survived            1 if the episode reached the 20 s time limit
  term_<name>         which termination ended the episode (one-hot)
  episode_len_s       episode duration in seconds
  rotation_rad        net cube yaw rotated in the task direction (radians)
  yaw_rate_rad_s      rotation_rad / time the rotation was measured over
  rotation_progress   mean per-step (yaw speed x pose stability), 0-1; same formula as
                      the training metric but anchored to this episode's true start pose
  position_error_cm   mean cube position drift from this episode's start pose
  tilt_error_rad      mean cube roll/pitch drift from this episode's start pose
  *_logged            the same three quantities as the training code's metric terms
                      compute them (these match the W&B Episode_Metrics curves)
  anchor_gap_cm       distance between the training metric's reset anchor and the
                      cube's true start pose (diagnostic; 0 if the anchor is fresh)

Why two versions: the metric terms record their reference pose inside the env's
reset, before MuJoCo recomputes body positions, so the reference is the cube pose
from *before* the reset. This script re-anchors after the reset so drift is
measured from where each episode actually starts, and reports both.
  fingertip_contact   mean fraction of fingertips touching the cube
  torque_l2           mean sum of squared actuator torques (Nm^2)
  mech_power_w        mean sum |torque * joint velocity| (W)
  work_j              total absolute mechanical work over the episode (J)

Usage (on a GPU node):
  uv run python scripts/evaluate.py --checkpoint path/to/model_4999.pt --label baseline
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

TASK_DEFAULT = "Mjlab-Leap-Left-HandCube-Rotate"
ACTOR_OBS_PER_STEP = 32  # actor proprioception per control step (320 / 10)

CONDITIONS = {
  "nominal": {},
  "heldout_mass": {"dr_cube_mass": {"mass_range": (1.5, 2.0)}},
  "heldout_friction": {"dr_shared_contact_friction": {"friction_range": (0.35, 0.55)}},
}

PER_EPISODE_METRICS = [
  "episode_len_s",
  "rotation_rad",
  "yaw_rate_rad_s",
  "rotation_progress",
  "position_error_cm",
  "tilt_error_rad",
  "rotation_progress_logged",
  "position_error_logged_cm",
  "tilt_error_logged_rad",
  "anchor_gap_cm",
  "fingertip_contact",
  "torque_l2",
  "mech_power_w",
  "work_j",
]


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True, type=Path)
  p.add_argument("--label", required=True, help="name for this policy, e.g. baseline")
  p.add_argument("--task", default=TASK_DEFAULT)
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
  p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
  p.add_argument("--history-length", type=int, default=None,
                 help="actor history length; default: inferred from the checkpoint")
  p.add_argument("--out-dir", type=Path, default=Path("eval_results"))
  p.add_argument("--device", default=None)
  p.add_argument("--stochastic", action="store_true",
                 help="diagnostic: sample actions like training instead of using the mean")
  p.add_argument("--obs-noise", action="store_true",
                 help="diagnostic: keep training observation noise on")
  return p.parse_args()


def actor_input_dim(ckpt_path: Path) -> int:
  sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["actor_state_dict"]
  for name, t in sd.items():
    if "mlp" in name and name.endswith("weight") and t.ndim == 2:
      return int(t.shape[1])
  raise RuntimeError(f"Could not find the actor's first layer in {ckpt_path}")


def build(task: str, cond: str, num_envs: int, history: int, seed: int, device: str,
          obs_noise: bool = False):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task)  # training config (20 s episodes, training DR), deep-copied
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  cfg.observations["actor"].history_length = history
  cfg.observations["actor"].enable_corruption = obs_noise
  cfg.curriculum = {}
  for event_name, overrides in CONDITIONS[cond].items():
    cfg.events[event_name].params.update(copy.deepcopy(overrides))
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  return env


def cube_yaw(u) -> torch.Tensor:
  from mjlab.utils.lab_api.math import euler_xyz_from_quat
  return euler_xyz_from_quat(u.scene["cube"].data.root_link_quat_w)[2]


def cube_pose(u):
  from mjlab.utils.lab_api.math import euler_xyz_from_quat
  d = u.scene["cube"].data
  roll, pitch, yaw = euler_xyz_from_quat(d.root_link_quat_w)
  return d.root_link_pos_w.clone(), roll.clone(), pitch.clone(), yaw.clone()


@torch.no_grad()
def rollout(wrapper, policy, stochastic: bool = False) -> dict[str, np.ndarray]:
  from mjlab.utils.lab_api.math import wrap_to_pi

  u = wrapper.unwrapped
  n, dev, dt = u.num_envs, u.device, u.step_dt
  max_len = int(u.max_episode_length)
  tm, mm = u.termination_manager, u.metrics_manager
  term_names = list(tm.active_terms)
  metric_names = list(mm.active_terms)
  # Prefer the physical cause when several terms fire on the same step.
  term_order = [t for t in term_names if t != "time_out"] + [t for t in term_names if t == "time_out"]

  obs, _ = wrapper.reset()
  alive = torch.ones(n, dtype=torch.bool, device=dev)
  steps = torch.zeros(n, device=dev)
  valid_steps = torch.zeros(n, device=dev)
  cause = torch.full((n,), -1, dtype=torch.long, device=dev)
  msum = torch.zeros((n, len(metric_names)), device=dev)
  rot = torch.zeros(n, device=dev)
  torque = torch.zeros(n, device=dev)
  power = torch.zeros(n, device=dev)
  # reset() ends with a forward pass, so this is each episode's true start pose.
  pos0, roll0, pitch0, prev_yaw = cube_pose(u)
  pos_err = torch.zeros(n, device=dev)
  tilt_err = torch.zeros(n, device=dev)
  progress = torch.zeros(n, device=dev)
  anchor_gap = torch.zeros(n, device=dev)
  for i, name in enumerate(metric_names):
    fn = mm._term_cfgs[i].func
    if name == "position_error" and hasattr(fn, "_init_pos_w"):
      anchor_gap = torch.linalg.vector_norm(fn._init_pos_w - pos0, dim=-1) * 100.0
  robot = u.scene["robot"]

  for _ in range(max_len + 5):
    act = policy(obs, stochastic_output=True) if stochastic else policy(obs)
    obs, _, dones, _ = wrapper.step(act)
    done = dones.bool()
    a = alive.float()
    # Metrics and terminations are computed before auto-reset, so they are valid
    # for every env on the step it finishes.
    msum += mm._step_values * a[:, None]
    steps += a
    # Cube/robot state read after step() is already reset for envs that just
    # finished, so those envs skip this step's state-based quantities.
    valid = alive & ~done
    v = valid.float()
    pos, roll, pitch, yaw = cube_pose(u)
    rot += -wrap_to_pi(yaw - prev_yaw) * v  # left hand: clockwise is the task direction
    prev_yaw = yaw
    pe = torch.linalg.vector_norm(pos - pos0, dim=-1)
    te = torch.linalg.vector_norm(
      torch.stack([wrap_to_pi(roll - roll0).abs(), wrap_to_pi(pitch - pitch0).abs()], -1), dim=-1)
    yaw_rate = -u.scene["cube"].data.root_link_ang_vel_w[:, 2]
    prog = ((yaw_rate / 0.20).clamp(0, 1) * (1 - pe / 0.02).clamp(0, 1)
            * (1 - te / 0.35).clamp(0, 1))
    pos_err += pe * v
    tilt_err += te * v
    progress += torch.nan_to_num(prog) * v
    tau, qd = robot.data.actuator_force, robot.data.joint_vel
    torque += (tau ** 2).sum(-1) * v
    if tau.shape == qd.shape:
      power += (tau * qd).abs().sum(-1) * v
    valid_steps += v
    newly = alive & done
    for idx_name in term_order:
      hit = newly & tm.get_term(idx_name).bool() & (cause < 0)
      cause[hit] = term_names.index(idx_name)
    alive &= ~done
    if not alive.any():
      break

  vs = valid_steps.clamp(min=1)
  out = {
    "episode_len_s": (steps * dt),
    "rotation_rad": rot,
    "yaw_rate_rad_s": rot / (vs * dt),
    "torque_l2": torque / vs,
    "mech_power_w": power / vs,
    "work_j": power * dt,  # total |mechanical work| over the episode
    "rotation_progress": progress / vs,
    "position_error_cm": pos_err / vs * 100.0,
    "tilt_error_rad": tilt_err / vs,
    "anchor_gap_cm": anchor_gap,
  }
  st = steps.clamp(min=1)
  lookup = {
    "rotation_progress": "rotation_progress_logged",
    "position_error": "position_error_logged_cm",
    "tilt_error": "tilt_error_logged_rad",
    "fingertip_contact_fraction": "fingertip_contact",
  }
  for i, name in enumerate(metric_names):
    if name in lookup:
      val = msum[:, i] / st
      out[lookup[name]] = val * 100.0 if name == "position_error" else val
  out = {k: v.cpu().numpy() for k, v in out.items()}
  cause_np = cause.cpu().numpy()
  for i, name in enumerate(term_names):
    out[f"term_{name}"] = (cause_np == i).astype(np.float64)
  out["survived"] = out.get("term_time_out", np.zeros(n))
  out["unfinished"] = (cause_np < 0).astype(np.float64)
  return out


def summarize(eps: dict[str, np.ndarray], per_seed: list[dict[str, np.ndarray]]) -> dict:
  s = {"episodes": int(len(eps["survived"]))}
  keys = ["survived"] + sorted(k for k in eps if k.startswith("term_")) + PER_EPISODE_METRICS
  for k in keys:
    if k not in eps:
      continue
    x = eps[k].astype(np.float64)
    seed_means = [float(np.mean(d[k])) for d in per_seed]
    s[k] = {
      "mean": float(np.mean(x)),
      "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
      "ci95": float(1.96 * np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0,
      "seed_means": seed_means,
    }
  return s


def main() -> None:
  args = parse_args()
  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  from rsl_rl.runners import OnPolicyRunner

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  ckpt = args.checkpoint.resolve()
  if not ckpt.exists():
    raise FileNotFoundError(ckpt)
  in_dim = actor_input_dim(ckpt)
  history = args.history_length or in_dim // ACTOR_OBS_PER_STEP
  print(f"[EVAL] checkpoint={ckpt}\n[EVAL] label={args.label} actor_input={in_dim} "
        f"history_length={history} num_envs={args.num_envs} seeds={args.seeds} device={device}")

  out_dir = args.out_dir / args.label
  out_dir.mkdir(parents=True, exist_ok=True)
  results = {
    "label": args.label, "checkpoint": str(ckpt), "task": args.task,
    "history_length": history, "num_envs": args.num_envs, "seeds": args.seeds,
    "policy": "stochastic (sampled)" if args.stochastic else "deterministic (action mean)",
    "obs_noise": args.obs_noise,
    "conditions": {c: CONDITIONS[c] for c in args.conditions}, "summary": {},
  }

  for cond in args.conditions:
    t0 = time.time()
    env = build(args.task, cond, args.num_envs, history, args.seeds[0], device, args.obs_noise)
    got = env.observation_manager.group_obs_dim["actor"]
    got = int(got[0]) if isinstance(got, tuple) else None
    if got != in_dim:
      raise RuntimeError(f"Env actor obs dim {got} != checkpoint input {in_dim}. "
                         f"Pass --history-length explicitly.")
    agent_cfg = load_rl_cfg(args.task)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    cfg_dict = asdict(agent_cfg)
    for key in ("actor", "critic"):
      if isinstance(cfg_dict.get(key), dict) and cfg_dict[key].get("class_name", "MLPModel") != "CNNModel":
        cfg_dict[key].pop("cnn_cfg", None)
    runner = (load_runner_cls(args.task) or OnPolicyRunner)(wrapper, cfg_dict, device=device)
    runner.load(str(ckpt), map_location=device)
    policy = runner.get_inference_policy(device=device)

    per_seed = []
    for seed in args.seeds:
      env.seed(seed)
      torch.manual_seed(seed)
      per_seed.append(rollout(wrapper, policy, args.stochastic))
      print(f"[EVAL] {cond} seed={seed}: survived={per_seed[-1]['survived'].mean():.3f} "
            f"rotation={per_seed[-1]['rotation_rad'].mean():.2f} rad "
            f"progress={per_seed[-1]['rotation_progress'].mean():.3f} "
            f"(logged {per_seed[-1]['rotation_progress_logged'].mean():.3f}) "
            f"anchor_gap={per_seed[-1]['anchor_gap_cm'].mean():.2f} cm")
    eps = {k: np.concatenate([d[k] for d in per_seed]) for k in per_seed[0]}
    with open(out_dir / f"episodes_{cond}.csv", "w", newline="") as f:
      w = csv.writer(f)
      keys = list(eps)
      w.writerow(["seed"] + keys)
      for s_i, d in zip(args.seeds, per_seed):
        for row in zip(*[d[k] for k in keys]):
          w.writerow([s_i] + [f"{x:.6g}" for x in row])
    results["summary"][cond] = summarize(eps, per_seed)
    env.close()
    del runner, policy, wrapper, env
    torch.cuda.empty_cache()
    print(f"[EVAL] {cond} done in {time.time() - t0:.0f}s")

  with open(out_dir / "summary.json", "w") as f:
    json.dump(results, f, indent=2)

  # Flat table, one row per condition; concatenate across labels to compare policies.
  rows = []
  for cond, s in results["summary"].items():
    row = {"label": args.label, "condition": cond, "episodes": s["episodes"]}
    for k, v in s.items():
      if isinstance(v, dict):
        row[k] = round(v["mean"], 4)
        row[k + "_ci95"] = round(v["ci95"], 4)
    rows.append(row)
  with open(out_dir / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

  print("\n[EVAL] summary (mean ± 95% CI over all episodes)")
  for cond, s in results["summary"].items():
    print(f"  == {cond}  ({s['episodes']} episodes)")
    for k, v in s.items():
      if isinstance(v, dict):
        print(f"     {k:22s} {v['mean']:10.4f} ± {v['ci95']:.4f}")
  print(f"[EVAL] wrote {out_dir}/summary.json, summary.csv, episodes_*.csv")


if __name__ == "__main__":
  main()
