"""
RobolabEnvironmentManager — wraps a RobolabEnv and InferenceClient, exposing
the same interface as LanguageTableEnvironmentManager for EnvServer.

The inner loop calls client.infer() for each active env at every step,
mirroring run_episode() in robolab/eval/episode.py, but packaged as an env
manager for the TCP server protocol.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from robolab.eval import InferenceClient
from robolab.core.environments.runtime import create_env

logger = logging.getLogger(__name__)


class RobolabEnvManager:
    """Env manager wrapping RobolabEnv with a policy InferenceClient inner loop.

    Parameters
    ----------
    envs : RobolabEnv
        Vectorized RoboLab environment (ManagerBasedRLEnv subclass).
    client : InferenceClient
        Policy inference client (pi05, dreamzero, etc.).
    max_inner_steps : int
        Inner-loop length: number of env steps executed per outer step() call.
    num_attempts : int
        Meta-RL attempts per episode (1 = no meta-RL).
    max_turns : int
        Outer-loop turns per attempt.
    do_reflection : bool
        Whether to include a reflection phase.
    """

    def __init__(
        self,
        task_envs: List[str],
        client: InferenceClient,
        max_inner_steps: int = 50,
        num_attempts: int = 1,
        max_turns: int = 1,
        do_reflection: bool = False,
        seed: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        instruction_type: str = "default",
        policy: str = "pi05",
    ):
        self.task_envs = task_envs
        self.envs = None
        self.client = client
        self.max_inner_steps = max_inner_steps
        self.num_attempts = num_attempts
        self.max_turns = max_turns
        self.do_reflection = do_reflection
        self._num_envs = num_envs
        self._device = device
        self._instruction_type = instruction_type
        self._policy = policy
        self._rng = np.random.default_rng(seed)
        self._current_task_env: Optional[str] = None
        # num_processes is set after the first env is created in reset()
        self.num_processes = num_envs

        self.curr_traj_idx = 0
        self.curr_turn_idx = 0

        self._last_obs: Any = None
        self._last_infos: List[Dict] = [{} for _ in range(self.num_processes)]
        self._instruction: str = ""

    def _get_instruction(self) -> str:
        """Pull task instruction from env config, fallback to empty string."""
        try:
            return str(self.envs.cfg.instruction)
        except AttributeError:
            return ""

    # ------------------------------------------------------------------
    # Core interface (must match _ALLOWED_METHODS in server.py)
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all envs and return initial observations.

        Samples a random task. If the task changed, closes the current env and
        creates a new one. Otherwise calls reset_eval_state() to unfreeze envs
        (same pattern as run_eval.py). Then double-resets: first seeds the
        simulation, second yields the clean initial observation.
        """
        task_env = self.task_envs[int(self._rng.integers(len(self.task_envs)))]

        if task_env != self._current_task_env:
            if self.envs is not None:
                self.envs.close()
            self.envs, _ = create_env(
                task_env,
                device=self._device,
                num_envs=self._num_envs,
                use_fabric=True,
                instruction_type=self._instruction_type,
                policy=self._policy,
            )
            self._current_task_env = task_env
            self.num_processes = self.envs.num_envs
        else:
            self.envs.reset_eval_state()

        self.envs.reset()
        obs, _ = self.envs.reset()

        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.client.reset()
        self._last_obs = obs
        self._instruction = self._get_instruction()

        observations = self._build_observations(obs)
        infos = self._build_reset_infos()
        return observations, infos

    def step(self, text_actions: List[str], phase: str = "play"):
        """Execute one outer-loop step.

        phase="play":    text_actions are per-env instruction strings;
                         runs the policy inner loop for max_inner_steps.
        phase="reflect": stores reflection text, returns dummy obs.
        """
        assert phase in ("play", "reflect")
        if phase == "reflect":
            return self._handle_reflect_step(text_actions)
        return self._handle_play_step(text_actions)

    def restart(self):
        """Restart envs for the next meta-RL attempt."""
        self.envs.reset_eval_state()
        self.envs.reset()
        obs, _ = self.envs.reset()

        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0
        self.client.reset()
        self._last_obs = obs

        observations = self._build_observations(obs)
        infos = self._build_reset_infos()
        return observations, infos

    def reflect(self):
        """Stub: reflection phase not used in single-attempt mode."""
        infos = [
            {"action_is_valid": True, "won": False}
            for _ in range(self.num_processes)
        ]
        observations = {
            "text": [""] * self.num_processes,
            "image": None,
            "anchor": ["reflection"] * self.num_processes,
        }
        return observations, infos

    def success_evaluator(self, **kwargs):
        """Compute success rates from accumulated episode infos.

        Called by the LaMer rollout runner after an episode. Iterates
        total_batch_list in reverse to find the last valid play step per
        attempt, then computes cumulative success across attempts.
        """
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        batch_size = len(total_batch_list)

        success = defaultdict(list)
        for bs in range(batch_size):
            wons = [False] * self.num_attempts
            best_reward = 0.0
            for i in reversed(range(len(total_batch_list[bs]))):
                batch_item = total_batch_list[bs][i]
                if batch_item["active_masks"]:
                    info = total_infos[bs][i]
                    traj_idx = batch_item["traj_idx"]
                    if batch_item["phase"] == "play":
                        wons[traj_idx] = wons[traj_idx] or info.get("won", False)
                        best_reward = max(best_reward, info.get("total_reward", 0.0))

            _won = False
            for traj_idx, won in enumerate(wons):
                _won = _won or won
                success[f"success_rate[{traj_idx}]"].append(_won)

        return {key: np.array(value) for key, value in success.items()}

    def build_text_obs(self, phase: str = "play") -> List[str]:
        """Return current text observations (task instruction repeated per env)."""
        return [self._instruction] * self.num_processes

    def render(self):
        """Return rendered frames from the environment."""
        return self.envs.render()

    def close(self):
        """Close environment and release client resources."""
        try:
            self.client.close()
        except Exception:
            logger.exception("Failed to close inference client")
        self.envs.close()

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def _handle_play_step(self, instructions: List[str]):
        """Run policy inner loop for max_inner_steps; return obs/rewards/dones/infos."""
        batch = self.num_processes
        action_dim = self._get_action_dim()
        total_rewards = np.zeros(batch, dtype=np.float32)

        # Reset per-env chunk state at the start of each new outer step so
        # the policy requests a fresh action chunk for the new goal/instruction.
        self.client.reset()

        for inner_step in range(self.max_inner_steps):
            active_ids = self.envs.active_env_ids
            if not active_ids:
                break

            actions = torch.zeros(batch, action_dim, device=self.envs.device)
            for env_id in active_ids:
                instruction = instructions[env_id] if instructions else self._instruction
                try:
                    ret = self.client.infer(self._last_obs, instruction, env_id=env_id)
                    actions[env_id] = torch.tensor(
                        ret["action"], dtype=torch.float32, device=self.envs.device
                    )
                except Exception:
                    logger.exception(
                        "client.infer failed at inner_step=%d env_id=%d", inner_step, env_id
                    )
                    raise

            try:
                obs, rewards, _term, _trunc, _ = self.envs.step(actions)
            except Exception:
                logger.exception(
                    "env.step failed at inner_step=%d active=%s", inner_step, active_ids
                )
                raise

            self._last_obs = obs

            # Accumulate reward only for envs that were active this step.
            rewards_np = self._to_numpy(rewards)
            active_mask = np.zeros(batch, dtype=bool)
            for env_id in active_ids:
                active_mask[env_id] = True
            total_rewards += rewards_np * active_mask

            if inner_step % 10 == 0:
                logger.debug("inner_step=%d active=%d/%d", inner_step, len(active_ids), batch)

            if self.envs.all_terminated:
                logger.info("All %d envs terminated at inner_step=%d", batch, inner_step)
                break

        # After the inner loop, read per-env success from the env's own tracker.
        env_results = self.envs.get_env_results()
        # An env is "done" if it was frozen (terminated or truncated).
        final_dones = np.array(
            [r.get("success") is not None for r in env_results], dtype=bool
        )

        self._last_infos = self._build_step_infos(
            rewards=total_rewards,
            env_results=env_results,
            instructions=instructions,
        )
        self.curr_turn_idx += 1

        observations = self._build_observations(self._last_obs)
        return observations, total_rewards, final_dones, self._last_infos

    def _handle_reflect_step(self, text_actions: List[str]):
        """Stub reflect step (not used in single-attempt mode)."""
        infos = [
            {"action_is_valid": True, "won": False, "is_action_valid": np.array(1.0)}
            for _ in range(self.num_processes)
        ]
        observations = {
            "text": [""] * self.num_processes,
            "image": None,
            "anchor": [""] * self.num_processes,
        }
        rewards = np.zeros(self.num_processes, dtype=np.float32)
        dones = np.array([False] * self.num_processes)
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_action_dim(self) -> int:
        """Pull action dim from env's action manager or gym action space."""
        action_dim = getattr(
            getattr(self.envs, "action_manager", None), "total_action_dim", None
        )
        if action_dim is None:
            action_dim = self.envs.action_space.shape[-1]
        return int(action_dim)

    def _to_numpy(self, x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _extract_images(self, obs) -> Optional[List[Optional[np.ndarray]]]:
        """Extract per-env combined images from the obs dict."""
        try:
            from robolab.core.observations.observation_utils import unpack_image_obs
            images = []
            for env_id in range(self.num_processes):
                img_dict = unpack_image_obs(obs, env_id=env_id)
                images.append(img_dict.get("combined_image"))
            return images
        except Exception:
            logger.debug("Could not extract images from obs", exc_info=True)
            return None

    def _build_observations(self, obs) -> Dict:
        text_obs = [self._instruction] * self.num_processes
        return {
            "text": text_obs,
            "image": self._extract_images(obs),
            "anchor": text_obs,
            # Raw Isaac tensor dict is omitted to reduce serialization overhead.
            # Populate "state" here if the consuming LaMer client requires it.
            "state": None,
        }

    def _build_reset_infos(self) -> List[Dict]:
        return [
            {"action_is_valid": True, "won": False, "is_action_valid": np.array(1.0)}
            for _ in range(self.num_processes)
        ]

    def _build_step_infos(
        self,
        rewards: np.ndarray,
        env_results: List[Dict],
        instructions: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Build per-env info dicts matching what success_evaluator expects."""
        infos = []
        for env_id in range(self.num_processes):
            instruction = instructions[env_id] if instructions else self._instruction
            result = env_results[env_id] if env_id < len(env_results) else {}
            # success=True → terminated by success predicate
            # success=False → truncated (max steps)
            # success=None  → still running (inner loop timed out)
            won = bool(result.get("success")) if result.get("success") is not None else False
            infos.append({
                "is_action_valid": np.array(1.0),
                "won": won,
                "total_reward": 1.0 if won else float(rewards[env_id]),
                "language_instruction": instruction,
            })
        return infos
