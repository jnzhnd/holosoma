"""Eval callback that records per-step trajectory data to an NPZ file.

Records joint positions, velocities, torques, body poses, and root state
for later visualization with viser_eval_viewer.py.

Since eval_agent doesn't exit on its own (must be killed with timeout),
this callback registers an atexit/signal handler to save data even on
interruption.
"""

from __future__ import annotations

import atexit
import json
import signal
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from torch.nn import Module

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import RecordingCallbackConfig
from holosoma.utils.safe_torch_import import torch


class EvalRecordingCallback(RLEvalCallback):
    """Records per-step data during evaluation and saves to .npz on completion or interruption."""

    def __init__(
        self,
        config: RecordingCallbackConfig,
        training_loop: Any = None,
        output_path_override: str | None = None,
    ):
        # Skip RLEvalCallback.__init__ which accesses training_loop.device;
        # training_loop may be set later by _create_eval_callbacks.
        Module.__init__(self)
        self.training_loop = training_loop
        self.device = None
        self.env_id = config.env_id
        self.output_path = output_path_override or config.output_path or "eval_recording.npz"

        self._buffers: dict[str, list[np.ndarray]] = {}
        self._metadata: dict[str, Any] = {}
        self._step_count = 0
        self._saved = False

    def _get_env(self):
        """Get the unwrapped BaseTask environment."""
        return self.training_loop._unwrap_env()

    def _save(self) -> None:
        """Save recorded data to NPZ. Safe to call multiple times."""
        if self._saved or self._step_count == 0:
            return
        self._saved = True

        arrays: dict[str, np.ndarray] = {}
        for name, values in self._buffers.items():
            if values:
                arrays[name] = np.stack(values, axis=0)

        arrays["_metadata_json"] = np.array(json.dumps(self._metadata))

        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **arrays)

        channel_summary = ", ".join(
            f"{name}{list(arr.shape)}" for name, arr in arrays.items() if name != "_metadata_json"
        )
        logger.info(
            f"EvalRecordingCallback: saved {self._step_count} steps to {path}\n"
            f"  Channels: {channel_summary}"
        )

    def on_pre_evaluate_policy(self) -> None:
        env = self._get_env()
        sim = env.simulator

        self._metadata["dt"] = float(env.dt)
        self._metadata["fps"] = round(1.0 / float(env.dt))
        self._metadata["env_id"] = self.env_id
        if hasattr(sim, "dof_names"):
            self._metadata["dof_names"] = list(sim.dof_names)
        if hasattr(sim, "body_names"):
            self._metadata["body_names"] = list(sim.body_names)

        channel_names = [
            "dof_pos", "dof_vel", "torques", "actions",
            "root_pos", "root_quat_xyzw",
            "body_pos_w", "body_quat_xyzw",
            "commanded_velocity",
        ]
        for name in channel_names:
            self._buffers[name] = []

        # Register safety nets so data is saved even if killed by timeout
        atexit.register(self._save)

        def _sigterm_handler(signum, frame):
            logger.info("EvalRecordingCallback: caught SIGTERM, saving recording...")
            self._save()
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _sigterm_handler)

        logger.info(f"EvalRecordingCallback: recording env_id={self.env_id}, output={self.output_path}")

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        env = self._get_env()
        sim = env.simulator
        eid = self.env_id

        def _to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().copy()

        self._buffers["dof_pos"].append(_to_np(sim.dof_pos[eid]))
        self._buffers["dof_vel"].append(_to_np(sim.dof_vel[eid]))

        # robot_root_states: [num_envs, 13] = pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)
        root = sim.robot_root_states[eid]
        self._buffers["root_pos"].append(_to_np(root[:3]))
        self._buffers["root_quat_xyzw"].append(_to_np(root[3:7]))

        self._buffers["body_pos_w"].append(_to_np(sim._rigid_body_pos[eid]))
        self._buffers["body_quat_xyzw"].append(_to_np(sim._rigid_body_rot[eid]))

        torques = self._extract_torques(env, eid)
        if torques is not None:
            self._buffers["torques"].append(_to_np(torques))

        if "actions" in actor_state and actor_state["actions"] is not None:
            self._buffers["actions"].append(_to_np(actor_state["actions"][eid]))

        # Record commanded velocity [lin_vel_x, lin_vel_y, ang_vel_yaw]
        if hasattr(env, "command_manager") and env.command_manager is not None:
            try:
                self._buffers["commanded_velocity"].append(_to_np(env.command_manager.commands[eid]))
            except (AttributeError, IndexError):
                pass

        self._step_count += 1
        return actor_state

    def _extract_torques(self, env: Any, env_id: int) -> torch.Tensor | None:
        """Extract applied torques from the action manager's joint control term."""
        if not hasattr(env, "action_manager") or env.action_manager is None:
            return None
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "torques"):
                return term.torques[env_id]
        return None

    def on_post_evaluate_policy(self) -> None:
        self._save()
