"""
LanguageTableEnvironmentManager — implements the same interface as LaMer's
EnvironmentManagerBase (same method signatures) without importing from LaMer.

Follows the two-loop VLA architecture from pybullet_vla/env_manager.py:
- Outer loop: LLM proposes goal strings
- Inner loop: VLA (or random) executes low-level actions in PyBullet
"""

import logging
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np

from .custom_task_provider import TaskProvider
from .frame_annotator import annotate_frames
from .state_to_text import batch_state_to_text, _decode_instruction

logger = logging.getLogger(__name__)

def clean_string(s: str) -> str:

    if s == "!":
        # NOTE: environment has done=True so first qwen token is passed ("!")
        return "inactive environment"

    cleaned = re.sub(r"[^a-z ,]", "", s.lower()).strip()
    
    if not cleaned:
        # NOTE: empty instruction after cleaning
        return "empty instruction"
    
    return cleaned


class LanguageTableEnvironmentManager:
    """Environment manager wrapping Language Table envs with a VLA inner loop.

    Parameters
    ----------
    envs : LanguageTableMultiProcessEnv
        Vectorised Language Table environment.
    num_attempts : int
        Number of meta-RL attempts per episode.
    max_turns : int
        Max outer-loop turns per attempt.
    do_reflection : bool
        Whether to include a reflection phase.
    max_inner_steps : int
        Number of inner-loop VLA steps per outer step.
    """

    def __init__(
        self,
        envs,
        num_attempts=1,
        max_turns=1,
        do_reflection=False,
        max_inner_steps=100,
        reflection_type="reflection_only",
        vla_policy=None,
        include_rgb=False,
        frame_subsample=5,
        n_steps=1,
        custom_task_provider: Optional[TaskProvider] = None,
        chunk_size: int = 1,
        max_seeds: Optional[int] = None,
    ):
        self.envs = envs
        self.num_processes = envs.num_processes
        self.num_attempts = num_attempts
        self.max_turns = max_turns
        self.do_reflection = do_reflection
        self.reflection_type = reflection_type
        self.max_inner_steps = max_inner_steps
        self.vla = vla_policy
        self._include_rgb = include_rgb
        self._frame_subsample = max(1, frame_subsample)
        self.n_steps = n_steps
        self._provider = custom_task_provider
        self.chunk_size = max(1, chunk_size)
        self.max_seeds = max_seeds
        if max_seeds is not None:
            self.envs.max_seeds = max_seeds

        # Action space bounds (Language Table uses [-0.1, 0.1]^2)
        self._action_low = -0.1
        self._action_high = 0.1

        # Meta-RL state
        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections: List[Dict] = [{} for _ in range(self.num_processes)]

        # Cached text observations
        self._init_text_obs: List[str] = [""] * self.num_processes
        self._last_text_obs: List[str] = [""] * self.num_processes
        self._last_infos: List[Dict] = [{} for _ in range(self.num_processes)]
        # Raw obs dicts cached for VLA (needs RGB at the start of each inner loop)
        self._last_obs_list: List[Dict] = [{} for _ in range(self.num_processes)]
        self._last_goal_strings: List[str] = [""] * self.num_processes
        self._task_strings: List[str] = [""] * self.num_processes

        # Custom-task state: cached per-env instruction strings when a
        # `custom_task_provider` is active. Generated on reset() only and
        # reused across restart() / subsequent turns so meta-RL attempts
        # share the same task within an episode.
        self._custom_instructions: List[str] = [""] * self.num_processes

    def _log_step_failure(
        self,
        inner_step: int,
        goal_strings: List[str],
        actions: List[np.ndarray],
        active_mask: np.ndarray,
    ) -> None:
        """Emit the last step context before re-raising an env failure."""
        active_indices = np.flatnonzero(active_mask).tolist()
        sample_context = []
        for env_idx in active_indices[:5]:
            action = np.asarray(actions[env_idx], dtype=np.float32)
            sample_context.append({
                "env_idx": env_idx,
                "goal": goal_strings[env_idx][:160],
                "action": action.tolist(),
                "action_norm": float(np.linalg.norm(action)),
            })

        logger.exception(
            "Language Table play step failed: inner_step=%d active=%d/%d "
            "active_indices_sample=%s context_sample=%s",
            inner_step,
            int(active_mask.sum()),
            len(active_mask),
            active_indices[:10],
            sample_context,
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def _extract_task_strings(self, obs_list):
        """Decode the environment task instruction from each observation."""
        tasks = []
        for obs in obs_list:
            instr = obs.get("instruction")
            tasks.append(_decode_instruction(instr) if instr is not None else "")
        return tasks

    def _extract_images(self, obs_list):
        """Extract RGB images from obs dicts for the client response.

        Only includes images if include_rgb=True (i.e. the outer LLM client
        needs images). The VLA accesses RGB directly from the obs dicts
        without going through this method.
        """
        if self._include_rgb and obs_list and "rgb" in obs_list[0]:
            return [obs["rgb"] for obs in obs_list]
        return None

    def _apply_custom_instructions(self, text_obs: List[str]) -> List[str]:
        """Prepend cached custom-task instructions to each env's text obs.

        Used when a `custom_task_provider` is active: the env emits no
        `instruction` field (reward_factory_cls=None) so `state_to_text`
        does not produce a "Task: ..." line. We inject it here so
        downstream prompts still see the task.
        """
        if self._provider is None:
            return text_obs
        return [
            f"Task: {self._custom_instructions[i]}\n{text_obs[i]}"
            if self._custom_instructions[i]
            else text_obs[i]
            for i in range(len(text_obs))
        ]

    def _ensure_rgb_for_vla(self, obs_list, indices):
        """Render once for cached VLA observations that skipped RGB."""
        if self.vla is None:
            return obs_list

        for i in indices:
            if isinstance(obs_list[i], dict) and "rgb" not in obs_list[i]:
                obs = dict(obs_list[i])
                obs["rgb"] = self.envs.render(i)
                obs_list[i] = obs
        return obs_list

    def reset(self):
        """Reset all envs and return text observations."""
        obs_list, infos = self.envs.reset()

        self.curr_traj_idx = 0
        self.curr_turn_idx = 0
        self.reflections = [{} for _ in range(self.num_processes)]

        if self.vla is not None:
            self.vla.reset(num_envs=self.num_processes)

        self._last_obs_list = obs_list
        text_obs = batch_state_to_text(obs_list)

        # Custom task provider: generate per-env instructions once per
        # episode (reset only) and cache. restart() will reuse them.
        if self._provider is not None:
            self._custom_instructions = [
                str(self._provider.instruction(
                    text_obs[i],
                    obs_list[i].get("rgb"),
                    i,
                ))
                for i in range(self.num_processes)
            ]
            text_obs = self._apply_custom_instructions(text_obs)

        self._init_text_obs = text_obs
        self._last_text_obs = text_obs
        self._last_infos = infos
        self._task_strings = self._extract_task_strings(obs_list)

        observations = {
            "text": text_obs,
            "image": self._extract_images(obs_list),
            "anchor": text_obs,
            "state": obs_list,
        }
        return observations, infos

    def step(self, text_actions: List[str], phase: str = "play"):
        """Execute a step in the meta-RL loop.

        phase="play": text_actions are goal strings → run VLA inner loop.
        phase="reflect": text_actions are reflections → store and return dummy obs.
        """
        assert phase in ("play", "reflect")

        if phase == "reflect":
            return self._handle_reflect_step(text_actions)

        return self._handle_play_step(text_actions)

    def restart(self):
        """Restart envs for the next meta-RL attempt."""
        obs_list, infos = self.envs.restart()

        self.curr_traj_idx += 1 if self.do_reflection else 0
        self.curr_turn_idx = 0

        if self.vla is not None:
            self.vla.reset(num_envs=self.num_processes)

        self._last_obs_list = obs_list
        text_obs = batch_state_to_text(obs_list)
        # Reuse the per-episode custom instructions (generated on reset()).
        text_obs = self._apply_custom_instructions(text_obs)
        self._last_text_obs = text_obs
        self._last_infos = infos
        self._task_strings = self._extract_task_strings(obs_list)

        observations = {
            "text": text_obs,
            "image": self._extract_images(obs_list),
            "anchor": text_obs,
            "state": obs_list,
        }
        return observations, infos

    def reflect(self):
        """Return prompts for the reflection phase."""
        infos = [
            {"action_is_valid": True, "won": False}
            for _ in range(self.num_processes)
        ]

        reflect_obs = self._build_reflect_prompt()

        observations = {
            "text": reflect_obs,
            "image": None,
            "anchor": ["reflection"] * self.num_processes,
        }
        return observations, infos

    def success_evaluator(self, **kwargs):
        """Evaluate episode success."""
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        batch_size = len(total_batch_list)

        success = defaultdict(list)
        for bs in range(batch_size):
            wons = [False for _ in range(self.num_attempts)]
            best_reward = 0.0
            for i in reversed(range(len(total_batch_list[bs]))):
                batch_item = total_batch_list[bs][i]
                if batch_item["active_masks"]:
                    info = total_infos[bs][i]
                    traj_idx = batch_item["traj_idx"]
                    if batch_item["phase"] == "play":
                        wons[traj_idx] = wons[traj_idx] or info.get("won", False)
                        best_reward = max(best_reward,
                                          info.get("total_reward", 0.0))

            _won = False
            for traj_idx, won in enumerate(wons):
                _won = _won or won
                success[f"success_rate[{traj_idx}]"].append(_won)

            if self.n_steps > 1:
                completion = min(best_reward / 100.0, 1.0)
                success["partial_completion"].append(completion)

        return {key: np.array(value) for key, value in success.items()}

    def build_text_obs(self, phase: str = "play") -> List[str]:
        """Build text observations."""
        if phase == "reflect":
            return self._build_reflect_prompt()
        return self._last_text_obs

    def render(self):
        """RGB frames from each worker (``env.render(mode='rgb_array')``)."""
        return self.envs.render()

    def get_object_positions(self):
        """Return gripper and block positions for every environment."""
        return self.envs.get_object_positions()

    def close(self):
        """Close all envs."""
        self.envs.close()

    # ------------------------------------------------------------------
    # VLA inner loop
    # ------------------------------------------------------------------

    def _handle_play_step(self, goal_strings: List[str]):
        """Run the VLA inner loop for a full episode.

        When a VLA policy is provided, it receives the LLM's goal strings as
        language conditioning and produces actions from RGB observations.
        Otherwise, falls back to random actions.

        Action chunking: each active env maintains a FIFO of depth chunk_size.
        When the FIFO is empty the policy is queried via predict_chunk, which
        returns >= chunk_size actions; only the first chunk_size are kept.
        On steps where the FIFO is still non-empty after the pop, rendering is
        skipped (skip_rgb=True) and no video frame is captured.
        """
        raw_strings = list(goal_strings)
        for i, raw in enumerate(raw_strings):
            cleaned = clean_string(raw)
            goal_strings[i] = cleaned

        batch = self.num_processes
        active_mask = np.ones(batch, dtype=bool)
        total_rewards = np.zeros(batch, dtype=np.float32)
        final_dones = np.zeros(batch, dtype=bool)
        last_obs = None
        last_infos = [{} for _ in range(batch)]
        self._last_goal_strings = list(goal_strings)

        # Per-env action FIFOs (owned by the manager, not the policy).
        action_fifos: List[deque] = [deque() for _ in range(batch)]

        all_frames = [[] for _ in range(batch)]
        for i in range(batch):
            if "rgb" in self._last_obs_list[i]:
                all_frames[i].append(self._last_obs_list[i]["rgb"].copy())

        # Seed the VLA with the obs cached from reset()/restart().
        obs_list_for_vla = self._last_obs_list if self.vla is not None else None

        # logger.info(f"goal_strings: {goal_strings}")
        # logger.info(f"obs_list_for_vla: {self.max_inner_steps}")
        for inner_step in range(self.max_inner_steps):
            if self.vla is not None:
                # Refill FIFOs for envs that are active and have no buffered actions.
                refill_mask = np.array(
                    [active_mask[i] and len(action_fifos[i]) == 0
                     for i in range(batch)],
                    dtype=bool,
                )
                if refill_mask.any():
                    chunks = self.vla.predict_chunk(
                        goal_strings, obs_list_for_vla, refill_mask,
                    )
                    for i in range(batch):
                        if not refill_mask[i]:
                            continue
                        chunk = chunks[i]  # (N_i, 2)
                        if len(chunk) < self.chunk_size:
                            raise AssertionError(
                                f"predict_chunk returned {len(chunk)} actions for env {i} "
                                f"but chunk_size={self.chunk_size} required"
                            )
                        for a in chunk[: self.chunk_size]:
                            action_fifos[i].append(np.asarray(a, dtype=np.float32))

                # Pop one action per env from the FIFO.
                actions = [
                    action_fifos[i].popleft() if active_mask[i] else np.zeros(2, dtype=np.float32)
                    for i in range(batch)
                ]

                # Skip rendering for envs that still have buffered actions.
                # The next predict_chunk for those envs won't use obs['rgb'],
                # so we save a camera render per skip step.
                skip_rgb_mask = np.array(
                    [active_mask[i] and len(action_fifos[i]) > 0
                     for i in range(batch)],
                    dtype=bool,
                )
            else:
                # Random actions fallback (no VLA loaded)
                if inner_step == 0:
                    logger.warning(
                        "No VLA policy loaded — using random actions for the "
                        "inner loop. Set --vla_checkpoint to use the LAVA policy."
                    )
                actions = [
                    np.random.uniform(self._action_low, self._action_high, size=(2,)).astype(np.float32)
                    if active_mask[i] else np.zeros(2, dtype=np.float32)
                    for i in range(batch)
                ]
                skip_rgb_mask = np.zeros(batch, dtype=bool)

            action_array = np.asarray(actions, dtype=np.float32)
            invalid_action_mask = ~np.isfinite(action_array).all(axis=1)
            if invalid_action_mask.any():
                bad_indices = np.flatnonzero(invalid_action_mask).tolist()
                logger.error(
                    "Invalid action batch at inner_step=%d bad_envs=%s goals=%s actions=%s",
                    inner_step,
                    bad_indices,
                    [goal_strings[i][:160] for i in bad_indices[:5]],
                    [action_array[i].tolist() for i in bad_indices[:5]],
                )

            cached_obs = (
                obs_list_for_vla
                if obs_list_for_vla is not None
                else self._last_obs_list
            )
            try:
                obs_list, rewards, dones, infos = self.envs.step(
                    actions,
                    active_mask=active_mask,
                    cached_obs=cached_obs,
                    cached_infos=last_infos,
                    skip_rgb_mask=skip_rgb_mask,
                )
            except Exception:
                self._log_step_failure(inner_step, goal_strings, actions, active_mask)
                raise

            # Feed fresh observations back to the VLA for the next refill.
            # On skip steps obs['rgb'] is absent; that's fine because the next
            # predict_chunk won't be called until the FIFO empties (at which
            # point a fresh render will have occurred).
            if self.vla is not None:
                obs_list_for_vla = obs_list

            # Capture video frames only on steps where RGB was actually rendered.
            if (inner_step + 1) % self._frame_subsample == 0:
                for i in range(batch):
                    if active_mask[i] and not skip_rgb_mask[i] and "rgb" in obs_list[i]:
                        all_frames[i].append(obs_list[i]["rgb"].copy())

            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=bool)

            total_rewards += rewards * active_mask

            newly_done = dones & active_mask
            final_dones |= newly_done
            active_mask &= ~dones
            if newly_done.any():
                obs_list = self._ensure_rgb_for_vla(
                    obs_list, np.flatnonzero(newly_done)
                )
            # Clear FIFOs for newly-done envs so they don't interfere with
            # subsequent attempts/restarts.
            for i in np.flatnonzero(newly_done):
                action_fifos[i].clear()

            if newly_done.any():
                logger.info(
                    "Language Table inner step %d completed envs=%s remaining_active=%d",
                    inner_step,
                    np.flatnonzero(newly_done).tolist(),
                    int(active_mask.sum()),
                )

            last_obs = obs_list
            for i in range(batch):
                if newly_done[i] or inner_step == self.max_inner_steps - 1:
                    last_infos[i] = infos[i]

            if not active_mask.any():
                break

        timed_out = active_mask.copy()
        # if timed_out.any():
        #     logger.info(
        #         "Language Table inner loop timed out after %d steps: "
        #         "envs=%s (these did NOT win)",
        #         self.max_inner_steps,
        #         np.flatnonzero(timed_out).tolist(),
        #     )

        final_obs = last_obs if last_obs is not None else [{}] * batch
        final_obs = self._ensure_rgb_for_vla(list(final_obs), range(batch))
        text_obs = batch_state_to_text(final_obs)

        # Custom reward: replace the env's accumulated per-step rewards
        # for this turn with the provider's scalar, computed from the
        # final (text, image) observation. The env is running with
        # reward_factory_cls=None so total_rewards is zero anyway.
        if self._provider is not None:
            if hasattr(self._provider, "precompute_rewards"):
                image_batch = [
                    final_obs[i].get("rgb") if isinstance(final_obs[i], dict) else None
                    for i in range(batch)
                ]
                self._provider.precompute_rewards(text_obs, image_batch)
            for i in range(batch):
                r = float(self._provider.reward(
                    text_obs[i],
                    final_obs[i].get("rgb") if isinstance(final_obs[i], dict) else None,
                    i,
                ))
                total_rewards[i] = r
            text_obs = self._apply_custom_instructions(text_obs)

        self._last_text_obs = text_obs
        self._last_obs_list = final_obs
        self._last_infos = last_infos

        observations = {
            "text": text_obs,
            "image": self._extract_images(final_obs),
            "anchor": text_obs,
            "state": final_obs,
        }

        # When a custom provider exposes a `success_threshold`, the env's
        # built-in done signal is irrelevant (the VLM is the validator):
        # an episode is a win iff the provider's reward meets the threshold.
        success_threshold = (
            getattr(self._provider, "success_threshold", None)
            if self._provider is not None
            else None
        )

        for i, info in enumerate(last_infos):
            info["is_action_valid"] = np.array(1.0)
            if success_threshold is not None:
                info["won"] = bool(total_rewards[i] >= success_threshold)
            else:
                info["won"] = bool(final_dones[i])
            info["total_reward"] = float(total_rewards[i])
            # When a custom task provider is active the env emits no
            # `instruction` field, so `_task_strings[i]` is empty. Fall back
            # to the provider's cached instruction so the video overlay keeps
            # showing the real task.
            task_overlay = self._task_strings[i]
            if self._provider is not None and self._custom_instructions[i]:
                task_overlay = self._custom_instructions[i]
            info["frames"] = annotate_frames(
                all_frames[i],
                traj_idx=self.curr_traj_idx,
                turn_idx=self.curr_turn_idx,
                instruction=goal_strings[i],
                task=task_overlay,
                reward=float(total_rewards[i]),
            )
            info["language_instruction"] = goal_strings[i]

        self.curr_turn_idx += 1

        return observations, total_rewards / 100.0, final_dones, last_infos

    def _handle_reflect_step(self, text_actions: List[str]):
        """Store reflections from the LLM."""
        for i, reflection in enumerate(text_actions):
            self.reflections[i][self.curr_traj_idx] = reflection

        infos = [
            {"action_is_valid": True, "won": False, "is_action_valid": np.array(1.0)}
            for _ in range(self.num_processes)
        ]
        observations = {
            "text": "",
            "image": None,
            "anchor": "",
        }
        rewards = np.zeros(self.num_processes, dtype=np.float32)
        dones = np.array([False] * self.num_processes)
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reflect_prompt(self) -> List[str]:
        """Build text prompts for the reflection phase."""
        prompts = []
        for i in range(self.num_processes):
            parts = [f"Initial state:\n{self._init_text_obs[i]}"]
            parts.append(f"\nEpisode outcome:\n{self._last_text_obs[i]}")

            for traj_idx in sorted(self.reflections[i]):
                parts.append(
                    f"\nReflection (attempt {traj_idx}):\n"
                    f"{self.reflections[i][traj_idx]}"
                )

            parts.append(
                "\nBased on the episode outcome, reflect on what went wrong "
                "and propose a new goal for the next attempt."
            )
            prompts.append("\n".join(parts))
        return prompts
