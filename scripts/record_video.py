"""Record an MP4 of a trained checkpoint, headless (no display needed).

Uses the same environment settings as scripts/evaluate.py (training config,
20 s episodes, evaluation condition's domain randomization, no observation
noise, deterministic policy), with a single environment rendered offscreen
from the task's default palm-tracking camera. With the same --seed, every
checkpoint starts from the same grasp and cube physics, so videos of different
policies are directly comparable.

Usage (on a GPU node):
  uv run --with imageio-ffmpeg python scripts/record_video.py \
      --checkpoint path/to/model_4999.pt --label baseline --out-dir videos
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless OpenGL on GPU nodes

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ACTOR_OBS_PER_STEP, CONDITIONS, TASK_DEFAULT, actor_input_dim  # noqa: E402


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True, type=Path)
  p.add_argument("--label", required=True)
  p.add_argument("--condition", default="nominal", choices=list(CONDITIONS))
  p.add_argument("--episodes", type=int, default=3, help="episodes to record back to back")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--width", type=int, default=720)
  p.add_argument("--height", type=int, default=720)
  p.add_argument("--out-dir", type=Path, default=Path("videos"))
  p.add_argument("--history-length", type=int, default=None)
  args = p.parse_args()

  import copy

  import mediapy as media
  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  from rsl_rl.runners import OnPolicyRunner

  try:  # mediapy needs an ffmpeg binary; use the pip-provided one if present.
    import imageio_ffmpeg
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  except ImportError:
    pass

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  ckpt = args.checkpoint.resolve()
  in_dim = actor_input_dim(ckpt)
  history = args.history_length or in_dim // ACTOR_OBS_PER_STEP

  cfg = load_env_cfg(TASK_DEFAULT)
  cfg.seed = args.seed
  cfg.scene.num_envs = 1
  cfg.observations["actor"].history_length = history
  cfg.observations["actor"].enable_corruption = False
  cfg.curriculum = {}
  cfg.viewer.width, cfg.viewer.height = args.width, args.height
  for event_name, overrides in CONDITIONS[args.condition].items():
    cfg.events[event_name].params.update(copy.deepcopy(overrides))

  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  agent_cfg = load_rl_cfg(TASK_DEFAULT)
  wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  cfg_dict = asdict(agent_cfg)
  for key in ("actor", "critic"):
    if isinstance(cfg_dict.get(key), dict) and cfg_dict[key].get("class_name", "MLPModel") != "CNNModel":
      cfg_dict[key].pop("cnn_cfg", None)
  runner = (load_runner_cls(TASK_DEFAULT) or OnPolicyRunner)(wrapper, cfg_dict, device=device)
  runner.load(str(ckpt), map_location=device)
  policy = runner.get_inference_policy(device=device)

  print(f"[VIDEO] {args.label}: history={history} condition={args.condition} seed={args.seed}")
  env.seed(args.seed)
  obs, _ = wrapper.reset()
  frames = [env.render()]
  term_names = list(env.termination_manager.active_terms)
  done_eps, steps = 0, 0
  with torch.no_grad():
    while done_eps < args.episodes and steps < args.episodes * int(env.max_episode_length) + 10:
      obs, _, dones, _ = wrapper.step(policy(obs))
      steps += 1
      if bool(dones[0]):
        causes = [t for t in term_names if bool(env.termination_manager.get_term(t)[0])]
        done_eps += 1
        print(f"[VIDEO]   episode {done_eps}: {steps * env.step_dt:.1f} s elapsed, ended by {causes}")
      frames.append(env.render())

  args.out_dir.mkdir(parents=True, exist_ok=True)
  out = args.out_dir / f"{args.label}_{args.condition}_seed{args.seed}.mp4"
  fps = round(1.0 / env.step_dt)  # one frame per control step = real time
  media.write_video(str(out), np.stack(frames), fps=fps)
  print(f"[VIDEO] wrote {out}  ({len(frames)} frames, {len(frames) / fps:.1f} s at {fps} fps)")
  env.close()


if __name__ == "__main__":
  main()
