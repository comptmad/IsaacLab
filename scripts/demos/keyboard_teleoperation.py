# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard teleoperation of a Franka Panda arm — no external hardware required.

Uses the Isaac Lab Se3Keyboard device (Omniverse keyboard events) to drive
differential IK on the Franka Panda. This is useful for confirming the scene
and IK pipeline work before connecting a physical haptic device.

Key bindings
------------
    I / K       Move end-effector along +X / -X
    J / H       Move end-effector along +Y / -Y
    U / O       Move end-effector along +Z / -Z
    N           Toggle gripper open / close
    M           Reset robot to default pose and re-centre target

Usage
-----
    ./isaaclab.sh -p scripts/demos/keyboard_teleoperation.py

    # Tune how fast the EE moves per key-press
    ./isaaclab.sh -p scripts/demos/keyboard_teleoperation.py --pos_sensitivity 0.02
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleoperation of Franka Panda.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--pos_sensitivity",
    type=float,
    default=0.01,
    help="Metres moved per physics step per held key (default 0.01 m).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import carb
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# ---------------------------------------------------------------------------
# Custom keyboard with non-conflicting bindings
# (W/E/Q are Isaac Sim transform-tool shortcuts; this uses I/K/J/H/U/O instead)
# ---------------------------------------------------------------------------

class TeleopKeyboard(Se3Keyboard):
    """Se3Keyboard subclass with Isaac-Sim-safe key bindings."""

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
            elif event.input.name == "N":
                self._close_gripper = not self._close_gripper
            elif event.input.name in self._INPUT_KEY_MAPPING:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()
        return True

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

WORKSPACE_LIMITS = {
    "x": (0.1, 0.9),
    "y": (-0.50, 0.50),
    "z": (1.05, 1.85),
}

# Default comfortable starting joint configuration for Franka Panda
FRANKA_DEFAULT_JOINT_POS = [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741]


@configclass
class FrankaKeyboardSceneCfg(InteractiveSceneCfg):
    """Franka Panda scene with a table, a cube, and finger contact sensors."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.50, 0.0, 1.05), rot=(0.707, 0, 0, 0.707)),
    )

    robot: Articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (-0.02, 0.0, 1.05)
    robot.spawn.activate_contact_sensors = True

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.60, 0.00, 1.15)),
    )

    left_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        track_pose=True,
    )

    right_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        track_pose=True,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, keyboard: Se3Keyboard) -> None:
    """Main simulation loop driven by keyboard input."""
    sim_dt = sim.get_physics_dt()
    count = 1

    robot: Articulation = scene["robot"]
    cube: RigidObject = scene["cube"]
    left_sensor: ContactSensor = scene["left_finger_contact_sensor"]
    right_sensor: ContactSensor = scene["right_finger_contact_sensor"]

    ee_body_name = "panda_hand"
    ee_body_idx = robot.body_names.index(ee_body_name)

    arm_joint_names = [f"panda_joint{i}" for i in range(1, 7)]
    arm_joint_indices = [robot.joint_names.index(n) for n in arm_joint_names]

    def reset_robot():
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[0, :7] = torch.tensor(FRANKA_DEFAULT_JOINT_POS, device=robot.device)
        robot.write_joint_state_to_sim(joint_pos, robot.data.default_joint_vel.clone())

    # Warm-up: settle robot at starting pose
    reset_robot()
    for _ in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # EE absolute target starts at current EE position after warm-up
    target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=scene.num_envs, device=sim.device)
    initial_ee_quat = robot.data.body_quat_w[:, ee_body_idx]
    ik_controller.set_command(command=target_pos.unsqueeze(0), ee_quat=initial_ee_quat)

    gripper_open = True
    gripper_target = 0.04
    ee_rotation_angle = robot.data.joint_pos[0, 6].item()

    # Wire the L key to reset (Se3Keyboard already handles L → reset() internally,
    # but we also need to re-centre target_pos and ik_controller here).
    def _on_reset():
        nonlocal target_pos, gripper_open, gripper_target, ee_rotation_angle
        reset_robot()
        scene.reset()
        ik_controller.reset()
        for _ in range(5):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
        gripper_open = True
        gripper_target = 0.04
        ee_rotation_angle = robot.data.joint_pos[0, 6].item()
        print("[INFO] Reset complete. Target re-centred at:", target_pos.tolist())

    keyboard.add_callback("L", _on_reset)

    print("\n[INFO] Keyboard teleoperation ready!")
    print(f"  pos_sensitivity : {args_cli.pos_sensitivity} m/step")
    print("  I / K           : +X / -X")
    print("  J / H           : +Y / -Y")
    print("  U / O           : +Z / -Z")
    print("  N               : Toggle gripper")
    print("  M               : Reset robot\n")

    prev_gripper_cmd = 1.0  # +1 = open

    while simulation_app.is_running():
        # Periodic full reset every 10 000 steps
        if count % 10000 == 0:
            count = 1
            reset_robot()

            cube_state = cube.data.default_root_state.clone()
            cube_state[:, :3] += scene.env_origins
            cube.write_root_pose_to_sim(cube_state[:, :7])
            cube.write_root_velocity_to_sim(cube_state[:, 7:])

            scene.reset()
            keyboard.reset()
            ik_controller.reset()

            for _ in range(5):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)

            target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
            print("[INFO] Periodic reset complete.")

        # ------------------------------------------------------------------ #
        # Read keyboard
        # Se3Keyboard.advance() returns [dx, dy, dz, rx, ry, rz, gripper]
        # where the first 6 are *deltas* scaled by pos/rot_sensitivity.
        # We ignore rotation here and accumulate position deltas.
        # ------------------------------------------------------------------ #
        kb_cmd = keyboard.advance()
        delta_pos = kb_cmd[:3].to(sim.device)  # [dx, dy, dz] in m
        gripper_cmd = kb_cmd[6].item()          # +1 open, -1 close

        # Gripper toggle on falling edge (open → close or vice versa)
        if gripper_cmd != prev_gripper_cmd:
            gripper_open = gripper_cmd > 0
            gripper_target = 0.04 if gripper_open else 0.0
        prev_gripper_cmd = gripper_cmd

        # Accumulate position target and clamp to workspace
        target_pos = target_pos + delta_pos
        target_pos[0] = target_pos[0].clamp(WORKSPACE_LIMITS["x"][0], WORKSPACE_LIMITS["x"][1])
        target_pos[1] = target_pos[1].clamp(WORKSPACE_LIMITS["y"][0], WORKSPACE_LIMITS["y"][1])
        target_pos[2] = target_pos[2].clamp(WORKSPACE_LIMITS["z"][0], WORKSPACE_LIMITS["z"][1])

        # ------------------------------------------------------------------ #
        # Differential IK
        # ------------------------------------------------------------------ #
        current_joint_pos = robot.data.joint_pos[:, arm_joint_indices]
        ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_body_idx, :, arm_joint_indices]

        ik_controller.set_command(command=target_pos.unsqueeze(0), ee_quat=ee_quat_w)
        joint_pos_des = ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, current_joint_pos)

        joint_pos_target = robot.data.joint_pos[0].clone()
        joint_pos_target[arm_joint_indices] = joint_pos_des[0]
        joint_pos_target[6] = ee_rotation_angle
        joint_pos_target[[-2, -1]] = gripper_target

        robot.set_joint_position_target(joint_pos_target.unsqueeze(0))

        for _ in range(5):
            scene.write_data_to_sim()
            sim.step()

        scene.update(sim_dt)
        count += 1


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1 / 200)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, 1.0, 1.70], [0.4, 0.0, 1.0])

    scene_cfg = FrankaKeyboardSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Se3Keyboard: pos_sensitivity is the metres per keypress event.
    # We use a small value since we accumulate every step; tune with --pos_sensitivity.
    kb_cfg = Se3KeyboardCfg(
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=0.0,   # rotation not used in this demo
        gripper_term=True,
        sim_device=args_cli.device,
    )
    keyboard = TeleopKeyboard(cfg=kb_cfg)

    print(keyboard)

    sim.reset()
    run_simulator(sim, scene, keyboard)


if __name__ == "__main__":
    main()
    simulation_app.close()
