# Migration: mjlab v1.1.1 → v1.6.0

Branch `mjlab-latest` updates this repo from the pinned mjlab v1.1.1 to **mjlab v1.6.0**
(released 8 Aug 2026, the newest release before the assignment date of 27 Aug 2026).

## Dependency changes (`pyproject.toml`)

| Package | Before | After |
|---|---|---|
| mjlab | v1.1.1 (git tag) | v1.6.0 (git tag) |
| mujoco | 3.5.0 | ~=3.11.0 |
| mujoco-warp | git rev `fc91589` | ~=3.11.0 (PyPI) |
| warp-lang | 1.12.0.dev (NVIDIA index) | >=1.14.0 (resolved: 1.17.0) |
| rsl-rl-lib | 4.0.1 (via mjlab) | 5.4.2 (via mjlab) |

`uv.lock` must be regenerated on a machine that can reach the PyTorch/NVIDIA indexes:

```bash
uv lock && uv sync
```

## API changes handled

| mjlab change | Fix in this repo |
|---|---|
| `DelayedActuatorCfg` / `DelayedActuator` removed (1.3); delay is now inline on the actuator cfg, `delay_target` removed | `leap_right_constants.py`: `delay_*` fields moved onto `IdealPdActuatorCfg`; `events.set_actuator_effort_limits` no longer unwraps a delayed actuator; `sim2sim/native/actions.py` unwraps only if a legacy `base_cfg` exists |
| `mdp.sync_actuator_delays` removed (1.3) | Ported into `tasks/hand_cube/mdp/events.py` (one lag per env, applied through each actuator's `set_lags`) |
| `mdp.randomize_field` and `EventTermCfg(domain_randomization=...)` removed (1.3) | Replaced with typed `mdp.dr.*` functions: `body_com_offset`, `joint_friction`, `joint_damping`, `joint_armature`; `randomize_pd_gains` → `dr.pd_gains` |
| `RslRlModelCfg(stochastic=, init_noise_std=)` removed (1.3, rsl-rl 5) | `rl_cfg.py`: actor uses `distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.7, "std_type": "scalar"}`; critic `distribution_cfg=None` |
| rsl-rl 5 `MLPModel` rejects `None` optional fields | `train.py`, `play.py`, `evaluate.py`, `record_video.py` strip `distribution_cfg=None` and unused RNN fields before building the runner; `train.py` writes `params/*.yaml` before the runner mutates the cfg |
| `mjlab.utils.os.update_assets` removed (1.3, PR #873) | Vendored as `robots/leap_hand/_assets.py` so mesh embedding is unchanged |
| `TerrainImporterCfg` alias removed (1.3) | `TerrainEntityCfg` |
| `CommandTerm._update_command(env_ids)` (1.6) | `commands.py`: all command terms take `env_ids`; the rotation-tracking term scopes its yaw accumulation to `env_ids` on partial resets |

## Behaviour changes to be aware of (not code changes)

- **Command delay is shared per environment.** Since 1.5.1, identically configured
  ideal-PD actuators are fused and share one delay buffer, so all 16 joints get the same
  lag in an environment (previously each joint actuator sampled its own lag). The
  reset-time `sync_actuator_delays` event already enforced a shared lag, so this mostly
  matches the old intent.
- **Reset reference pose is fixed upstream.** With v1.1.1 the task's drift metrics were
  anchored to the pre-reset cube pose (evaluation measured an anchor gap of ~1.18 cm).
  Under v1.6.0, `scripts/evaluate.py` measures an anchor gap of exactly 0.00 cm, so the
  logged and re-anchored drift metrics now agree.
- MuJoCo / MuJoCo Warp 3.11 and the other reset/history fixes in mjlab 1.2–1.6 mean
  numbers are not bit-comparable with v1.1.1 runs; the experiments in the report were run
  on v1.1.1.

## Validation performed (CPU, no GPU training)

- All task configs load; `Mjlab-Leap-Left-HandCube-Rotate` builds with actor/critic
  observation sizes 320/83 (640 with `--env.observations.actor.history-length 20`),
  unchanged from v1.1.1.
- 40 environment steps with zero actions run without errors.
- `scripts/train.py`: 2 PPO iterations (8 envs) complete and save checkpoints.
- `scripts/evaluate.py`: runs all three conditions on that checkpoint and writes
  `summary.json` / CSVs.
- `scripts/record_video.py`: renders a 20 s clip headlessly (EGL).
- `--help` parses for every script in `scripts/`.

No full training run has been done on v1.6.0.
