# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard teleoperation of the microsurgical robot — no external hardware required.

Loads the pre-built microsurgical scene USD and controls arm1 via differential IK,
driven by keyboard input. Use this to confirm the scene and IK pipeline work before
connecting a Geomagic Touch.

Key bindings
------------
    I / K       Move end-effector along +X / -X
    J / H       Move end-effector along +Y / -Y
    U / O       Move end-effector along +Z / -Z
    M           Reset robot to default joint pose and re-centre IK target

Usage
-----
    ./isaaclab.sh -p scripts/demos/keyboard_microsurgical.py

    # Print joint / body names and exit (useful for debugging)
    ./isaaclab.sh -p scripts/demos/keyboard_microsurgical.py --list_info

    # Tune motion scale (default 0.005 keeps movements surgical-scale)
    ./isaaclab.sh -p scripts/demos/keyboard_microsurgical.py --pos_sensitivity 0.005
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleoperation of the microsurgical robot.")
parser.add_argument(
    "--list_info",
    action="store_true",
    help="Print joint names and body names then exit.",
)
parser.add_argument(
    "--pos_sensitivity",
    type=float,
    default=0.005,
    help="Metres moved per physics step per held key (default 0.005 m for surgical precision).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import carb
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Custom keyboard with non-conflicting bindings
# (W/E/Q are Isaac Sim transform-tool shortcuts; this uses I/K/J/H/U/O instead)
# ---------------------------------------------------------------------------

class MicrosurgicalKeyboard(Se3Keyboard):
    """Se3Keyboard subclass with Isaac-Sim-safe key bindings for position-only control."""

    def _create_key_bindings(self):
        self._INPUT_KEY_MAPPING = {
            "I": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,   # +X
            "K": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,  # -X
            "J": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,   # +Y
            "H": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,  # -Y
            "U": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,   # +Z
            "O": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,  # -Z
        }

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "M":
                self.reset()
            elif event.input.name in self._INPUT_KEY_MAPPING:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()
        return True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MICROBOT_USD_PATH = "/home/comptmad/Downloads/IssacSim_envs/flattened_microbot.usd"
MICROBOT_PRIM_PATH = "/World/Microrobot/body_link"

ARM1_JOINT_NAMES = [
    "RevoluteJoint1",
    "RevoluteJoint2",
    "RevoluteJoint3",
    "RevoluteJoint4",
    "RevoluteJoint5",
]

EE_BODY_NAME = "arm1_link5"


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

@configclass
class MicrosurgicalSceneCfg(InteractiveSceneCfg):

    robot: Articulation = ArticulationCfg(
        prim_path=MICROBOT_PRIM_PATH,
        spawn=None,
        actuators={
            "arm1": ImplicitActuatorCfg(
                joint_names_expr=ARM1_JOINT_NAMES,
                effort_limit=100.0,
                effort_limit_sim=100.0,
                velocity_limit_sim=1.0,
                stiffness=600.0,
                damping=50.0,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scene_usd(usd_path: str) -> None:
    """Add the scene USD as a sublayer so all prims appear directly under /World."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().subLayerPaths.append(usd_path)
    print(f"[INFO] Loaded scene USD: {usd_path}")


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    keyboard: Se3Keyboard,
) -> None:
    """Main teleoperation loop."""
    sim_dt = sim.get_physics_dt()
    count = 1

    robot: Articulation = scene["robot"]

    arm1_joint_indices = [robot.joint_names.index(n) for n in ARM1_JOINT_NAMES]
    ee_body_idx = robot.body_names.index(EE_BODY_NAME)

    # ---------------------------------------------------------------------- #
    # Diagnostic: print joint / body names and exit
    # ---------------------------------------------------------------------- #
    if args_cli.list_info:
        print("\n=== Joint names ===")
        for i, name in enumerate(robot.joint_names):
            print(f"  [{i:2d}] {name}")
        print("\n=== Body names ===")
        for i, name in enumerate(robot.body_names):
            print(f"  [{i:2d}] {name}")
        return

    # ---------------------------------------------------------------------- #
    # Warm-up: hold the USD home pose so the arm settles before recording the
    # IK reference frame.
    # Values are drive:angular:physics:targetPosition (degrees) → radians.
    # ---------------------------------------------------------------------- #
    default_joint_pos = torch.tensor(
        [[-0.209476, 0.025148, -0.486721, 0.000025, -0.000040]], device=sim.device
    )
    robot.write_joint_state_to_sim(default_joint_pos, robot.data.default_joint_vel.clone())
    for _ in range(50):
        robot.set_joint_position_target(default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # Absolute EE target starts at the settled position
    target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.1},
    )
    ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=sim.device)
    ee_quat = robot.data.body_quat_w[:, ee_body_idx]
    ik_controller.set_command(command=target_pos.unsqueeze(0), ee_quat=ee_quat)

    def _on_reset():
        nonlocal target_pos
        robot.write_joint_state_to_sim(default_joint_pos, robot.data.default_joint_vel.clone())
        scene.reset()
        ik_controller.reset()
        for _ in range(10):
            robot.set_joint_position_target(default_joint_pos)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
        print("[INFO] Reset complete. Target re-centred at:", target_pos.tolist())

    keyboard.add_callback("M", _on_reset)

    print("\n[INFO] Keyboard teleoperation ready!")
    print(f"  pos_sensitivity : {args_cli.pos_sensitivity} m/step")
    print(f"  EE start pos    : {target_pos.tolist()}")
    print("  I / K           : +X / -X")
    print("  J / H           : +Y / -Y")
    print("  U / O           : +Z / -Z")
    print("  M               : Reset robot\n")

    while simulation_app.is_running():
        if count % 10000 == 0:
            count = 1
            robot.write_joint_state_to_sim(default_joint_pos, robot.data.default_joint_vel.clone())
            scene.reset()
            keyboard.reset()
            ik_controller.reset()
            target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
            print("[INFO] Periodic reset complete.")

        kb_cmd = keyboard.advance()
        delta_pos = kb_cmd[:3].to(sim.device)
        target_pos = target_pos + delta_pos

        current_joint_pos = robot.data.joint_pos[:, arm1_joint_indices]
        ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_body_idx, :, arm1_joint_indices]

        ik_controller.set_command(command=target_pos.unsqueeze(0), ee_quat=ee_quat_w)
        joint_pos_des = ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, current_joint_pos)

        joint_pos_target = robot.data.joint_pos[0].clone()
        joint_pos_target[arm1_joint_indices] = joint_pos_des[0]
        robot.set_joint_position_target(joint_pos_target.unsqueeze(0))

        # joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
        # joint_pos_target = joint_pos_target.clamp_(
        #     robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
        #     )
        #     # apply action to the robot
        # robot.set_joint_position_target(joint_pos_target)

        for _ in range(5):
            scene.write_data_to_sim()
            sim.step()

        scene.update(sim_dt)
        count += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1 / 200)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.2])

    load_scene_usd(MICROBOT_USD_PATH)

    scene_cfg = MicrosurgicalSceneCfg(num_envs=1, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)

    kb_cfg = Se3KeyboardCfg(
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=0.0,
        gripper_term=False,
        sim_device=args_cli.device,
    )
    keyboard = MicrosurgicalKeyboard(cfg=kb_cfg)

    print(keyboard)

    sim.reset()
    run_simulator(sim, scene, keyboard)


if __name__ == "__main__":
    main()
    simulation_app.close()
