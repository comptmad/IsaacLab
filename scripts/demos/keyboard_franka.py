# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard teleoperation of a Franka Panda arm using franka_test.usd.

Key bindings
------------
    I / K       Move end-effector along +X / -X
    J / H       Move end-effector along +Y / -Y
    U / O       Move end-effector along +Z / -Z
    N           Toggle gripper open / close
    M / L       Reset robot to default pose

Usage
-----
    ./isaaclab.sh -p scripts/demos/keyboard_franka.py

    # Tune how fast the EE moves per key-press
    ./isaaclab.sh -p scripts/demos/keyboard_franka.py --pos_sensitivity 0.02
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleoperation of Franka Panda (franka_test.usd).")
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
import omni.usd
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from pathlib import Path

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

FRANKA_TEST_USD = str(Path(__file__).resolve().parents[2] / "assets" / "franka_test.usd")

WORKSPACE_LIMITS = {
    "x": (-0.5, 0.5),
    "y": (-0.8, 0.2),
    "z": (0.3, 1.4),
}



class TeleopKeyboard(Se3Keyboard):
    """Se3Keyboard subclass with Isaac-Sim-safe key bindings."""

    def _create_key_bindings(self):
        self._INPUT_KEY_MAPPING = {
            "I": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "K": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "J": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "H": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            "U": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "O": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
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


@configclass
class FrankaKeyboardSceneCfg(InteractiveSceneCfg):
    """Scene with franka_test.usd and finger contact sensors."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: Articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.usd_path = FRANKA_TEST_USD
    robot.spawn.activate_contact_sensors = True
    robot.init_state.pos = (0.0, 0.0, 1.4648)
    robot.init_state.joint_pos = {
        "panda_joint1": 0.058822,
        "panda_joint2": -0.162536,
        "panda_joint3": -0.810561,
        "panda_joint4": -1.674450,
        "panda_joint5": 0.205468,
        "panda_joint6": 0.0,
        "panda_joint7": 0.799991,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }

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

    ee_body_name = "panda_hand"
    ee_body_idx = robot.body_names.index(ee_body_name)

    arm_joint_names = [f"panda_joint{i}" for i in range(1, 7)]
    arm_joint_indices = [robot.joint_names.index(n) for n in arm_joint_names]

    def reset_robot():
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())

    reset_robot()
    for _ in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

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
    print("  M / L           : Reset robot\n")

    prev_gripper_cmd = 1.0

    while simulation_app.is_running():
        if count % 10000 == 0:
            count = 1
            reset_robot()

            scene.reset()
            keyboard.reset()
            ik_controller.reset()

            for _ in range(5):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)

            target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
            print("[INFO] Periodic reset complete.")

        kb_cmd = keyboard.advance()
        delta_pos = kb_cmd[:3].to(sim.device)
        gripper_cmd = kb_cmd[6].item()

        if gripper_cmd != prev_gripper_cmd:
            gripper_open = gripper_cmd > 0
            gripper_target = 0.04 if gripper_open else 0.0
        prev_gripper_cmd = gripper_cmd

        target_pos = target_pos + delta_pos
        target_pos[0] = target_pos[0].clamp(WORKSPACE_LIMITS["x"][0], WORKSPACE_LIMITS["x"][1])
        target_pos[1] = target_pos[1].clamp(WORKSPACE_LIMITS["y"][0], WORKSPACE_LIMITS["y"][1])
        target_pos[2] = target_pos[2].clamp(WORKSPACE_LIMITS["z"][0], WORKSPACE_LIMITS["z"][1])

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


def _bake_base_rotation(num_envs: int) -> None:
    """Rotate the robot root prim 180° around Y directly on the stage."""
    stage = omni.usd.get_context().get_stage()
    for i in range(num_envs):
        prim_path = f"/World/envs/env_{i}/Robot"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[WARN] Robot prim not found at {prim_path}")
            continue
        xformable = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
        if "xformOp:orient" in ops:
            ops["xformOp:orient"].Set(Gf.Quatf(0.0, 0.0, 1.0, 0.0))
        elif "xformOp:rotateXYZ" in ops:
            ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 180.0, 0.0))
        else:
            xformable.AddOrientOp().Set(Gf.Quatf(0.0, 0.0, 1.0, 0.0))
    print("[INFO] Robot base rotation baked onto stage.")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1 / 200)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([-1.2, -0.8, 1.2], [0.0, -0.5, 0.7])

    scene_cfg = FrankaKeyboardSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    _bake_base_rotation(args_cli.num_envs)

    kb_cfg = Se3KeyboardCfg(
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=0.0,
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
