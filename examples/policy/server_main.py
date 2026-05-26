# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Run policy evaluation across multiple tasks.

This script runs policy evaluation on multiple registered tasks, supporting various
policy backends (pi0, etc.) with options for subtask tracking and result logging.

Supports multi-env: each "run" spawns num_envs parallel episodes.
Total episodes = num_runs * num_envs.

Usage:
    Run on all registered tasks:
    $ python run_eval.py

    Run on specific tasks:
    $ python run_eval.py --task BananaInBowlTask RubiksCubeTask

    Run on a tag:
    $ python run_eval.py --tag spatial

    Use specific policy:
    $ python run_eval.py --policy pi05

    Run multiple episodes with 2 parallel envs:
    $ python run_eval.py --num-runs 2 --num_envs 4

Output:
    Results are saved to: output/<output_folder_name>/
"""

import argparse
import cv2 # Must import this before isaaclab. Do not remove
import os
import traceback
import sys
from isaaclab.app import AppLauncher
from robolab.constants import get_timestamp, DEFAULT_TASK_SUBFOLDERS # noqa

# add argparse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default=None,
                       help="List of tasks to evaluate on ")
parser.add_argument("--task-dirs", nargs='+', default=DEFAULT_TASK_SUBFOLDERS,
                       help="List of task directories to evaluate on")
parser.add_argument("--policy",
                    choices=["pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05", "gr00t", "dreamzero", "molmo", "openvla", "openvla_oft"], default="pi05",
                       help="Action-prediction backend to use (default: pi05)")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1,
                       help="Number of sequential runs per task (default: 1). Total episodes = num_runs * num_envs. Prefer increasing --num_envs for more episodes. Only increase --num-runs if you run out of GPU memory with the desired num_envs.")
parser.add_argument("--enable-subtask", "--enable_subtask", action="store_true",
                       help="Enable subtask progress checking (default: False)")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                       help="Enable proprio image data recording (default: False)")
parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                       help="Output folder name under /robolab/output. Default is <timestamp>_<policy>. If you provide the output folder name for a previous run, the script will skip the tasks and episodes that have already been run.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                       help="Verbose output (default: False)")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                       help="Debug output (default: False)")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                       help="Remote host for policy server (default: localhost)")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                       help="Remote port for policy server (default: 8000)")
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None,
                       help="Full WebSocket URI for policy server, e.g. wss://host.lepton.run. "
                            "Overrides --remote-host and --remote-port when set.")
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                       help="Number of actions to execute from each predicted chunk before requesting a new one. "
                            "If omitted, each inference client uses its own default. "
                            "Must match the model's action_horizon for best performance.")
# DreamZero-specific flags (silently ignored by other backends via create_client kwarg filtering)
parser.add_argument("--dz-binarize-gripper", "--dz_binarize_gripper", action="store_true",
                    help="[DreamZero] Re-enable gripper binarization at 0.5 threshold (ablation; default: off)")
parser.add_argument("--dz-resize", "--dz_resize", type=str, default="area", choices=["area", "linear", "pad"],
                    help="[DreamZero] Image resize method: 'area' (default, INTER_AREA), 'linear' (INTER_LINEAR), "
                         "or 'pad' (aspect-preserving letterbox). Note: 'area'/'linear' change aspect ratio if "
                         "source differs from 180x320 target.")
parser.add_argument("--server-port", "--server_port", type=int, default=50051,
                    help="Port for robolab env server (default: 50051)")
parser.add_argument("--server-host", "--server_host", type=str, default="0.0.0.0",
                    help="Host for robolab env server (default: 0.0.0.0)")
parser.add_argument("--num-inner-steps", "--num_inner_steps", type=int, default=50,
                    help="Number of inner steps for robolab env server (default: 50)")
parser.add_argument("--remote-token", "--remote_token", type=str, default=None,
                    help="Bearer token for authenticated endpoints (e.g. Lepton). "
                         "Falls back to DREAMZERO_API_TOKEN env var.")
parser.add_argument("--dz-cam2", "--dz_cam2", type=str, default="black",
                    choices=["black", "right", "head", "duplicate"],
                    help="[DreamZero] Second exterior camera: 'black' (default, matches training dropout), "
                         "'right' (over-shoulder), 'head' (front overhead), 'duplicate' (copy of left)")
parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default",
                       help="Which instruction variant to use when a task defines multiple (default, vague, specific, etc.)")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"],
                    help="Which videos to save: 'all' (sensor + viewport), 'viewport' only, 'sensor' only, or 'none' (default: all)")
# parse the arguments
args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = args_cli.video_mode != "none"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import PACKAGE_DIR, set_output_dir # noqa
from robolab.core.environments.runtime import create_env # noqa
from robolab.eval import create_client, run_episode, summarize_run # noqa
from robolab.core.environments.factory import get_envs # noqa
from robolab.core.utils.print_utils import print_experiment_summary # noqa
from robolab.core.logging.results import check_all_episodes_complete, check_run_complete # noqa
from robolab.core.logging.results import init_experiment, summarize_experiment_results # noqa
import robolab.constants # noqa

# additional imports
from robolab.lamer.server import EnvServer
from robolab.lamer.env_manager import RobolabEnvManager

# Update robolab.constants module settings from command line arguments
robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

# Run automatic factory generation before main
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs # noqa
if args_cli.policy == "dreamzero" and args_cli.dz_cam2 in ("right", "head"):
    from robolab.registrations.droid_jointpos.camera_presets import WRIST_LEFT_RIGHT_HEAD # noqa
    auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task, cameras=WRIST_LEFT_RIGHT_HEAD)
else:
    auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)

def main():
    """Main function."""
    if args_cli.output_folder_name is None:
        args_cli.output_folder_name = get_timestamp() + f"_{args_cli.policy}"
        if args_cli.instruction_type != "default":
            args_cli.output_folder_name += f"_{args_cli.instruction_type}"

    output_dir = os.path.join(PACKAGE_DIR, "output", args_cli.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    task_envs = get_envs(task=args_cli.task)
    filter_str = f"task: {args_cli.task}"

    num_envs = args_cli.num_envs
    num_runs = args_cli.num_runs
    total_episodes = num_runs * num_envs

    print_experiment_summary(
        task_envs=task_envs,
        filter_str=filter_str,
        num_envs=num_envs,
        num_episodes=total_episodes,
        policy=args_cli.policy,
        instruction_type=args_cli.instruction_type,
        output_dir=output_dir,
    )

    env, env_cfg = create_env(task_envs[0],
            device=args_cli.device,
            num_envs=num_envs,
            use_fabric=True,
            instruction_type=args_cli.instruction_type,
            policy=args_cli.policy)
    
    client = create_client(
            args_cli.policy,
            remote_host=args_cli.remote_host,
            remote_port=args_cli.remote_port,
            remote_uri=args_cli.remote_uri,
            open_loop_horizon=args_cli.open_loop_horizon,
            api_token=args_cli.remote_token,
            binarize_gripper=args_cli.dz_binarize_gripper,
            resize=args_cli.dz_resize,
            cam2_source=args_cli.dz_cam2,
        )

    robolab_env_manager = RobolabEnvManager(env, client, max_inner_steps=args_cli.num_inner_steps)
    env_server = EnvServer(robolab_env_manager, host=args_cli.server_host, port=args_cli.server_port)

    env_server.serve()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
