# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demonstration of Geomagic Touch device teleoperation with a Franka Panda arm.

Requires the Geomagic_Touch_ROS2 driver to be running:

.. code-block:: bash

    ros2 launch omni_common omni_state.launch.py

Then launch this script:

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/geomagic_teleoperation.py

    # With custom ROS namespace (must match omni_name in driver launch)
    ./isaaclab.sh -p scripts/demos/geomagic_teleoperation.py --ros_namespace Geomagic

    # With sensitivity adjustment
    ./isaaclab.sh -p scripts/demos/geomagic_teleoperation.py --pos_sensitivity 2.0

Button mapping:
    Grey  button: Toggle gripper open / close
    White button: Rotate end-effector 60 degrees
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Geomagic Touch teleoperation demo with Franka Panda.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--ros_namespace",
    type=str,
    default="Geomagic",
    help="ROS2 topic namespace matching omni_name in the driver launch file.",
)
parser.add_argument(
    "--pos_sensitivity",
    type=float,
    default=1.0,
    help="Position sensitivity scaling factor.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import GeomagicDevice, GeomagicDeviceCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from pathlib import Path

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

WORKSPACE_LIMITS = {
    "x": (0.1, 0.9),
    "y": (-0.50, 0.50),
    "z": (0.0, 0.8),
}


def apply_geomagic_to_robot_mapping(
    geomagic_pos: np.ndarray | torch.Tensor,
    geomagic_initial_pos: np.ndarray,
    robot_initial_pos: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Map Geomagic Touch workspace to Franka end-effector world frame.

    Geomagic axes (stylus tip, device sitting on table):
        X: left / right
        Y: up / down
        Z: toward / away from user

    Mapped to robot frame as:
        Robot X  <-  Geomagic -Z  (push forward -> robot reaches forward)
        Robot Y  <-  Geomagic  X  (move right   -> robot moves right)
        Robot Z  <-  Geomagic  Y  (lift up       -> robot lifts up)

    Tune ``--pos_sensitivity`` if the motion range feels too small or large.
    """
    if isinstance(geomagic_pos, torch.Tensor):
        geomagic_pos = geomagic_pos.cpu().numpy()
    if isinstance(robot_initial_pos, torch.Tensor):
        robot_initial_pos = robot_initial_pos.cpu().numpy()

    delta = geomagic_pos - geomagic_initial_pos

    robot_offset = np.array([-delta[2], delta[0], delta[1]])
    robot_pos = robot_initial_pos + robot_offset

    robot_pos[0] = np.clip(robot_pos[0], WORKSPACE_LIMITS["x"][0], WORKSPACE_LIMITS["x"][1])
    robot_pos[1] = np.clip(robot_pos[1], WORKSPACE_LIMITS["y"][0], WORKSPACE_LIMITS["y"][1])
    robot_pos[2] = np.clip(robot_pos[2], WORKSPACE_LIMITS["z"][0], WORKSPACE_LIMITS["z"][1])

    return robot_pos


@configclass
class FrankaGeomagicSceneCfg(InteractiveSceneCfg):
    """Scene with Franka Panda, a table, a cube, and finger contact sensors."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: Articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.usd_path = str(Path(__file__).resolve().parents[4] / "assets" / "franka_test.usd")
    robot.spawn.activate_contact_sensors = True
    robot.init_state.joint_pos = {}

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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, device: GeomagicDevice):
    """Main simulation loop with Geomagic teleoperation and force feedback."""
    sim_dt = sim.get_physics_dt()
    count = 1

    robot: Articulation = scene["robot"]
    cube: RigidObject = scene["cube"]
    left_sensor: ContactSensor = scene["left_finger_contact_sensor"]
    right_sensor: ContactSensor = scene["right_finger_contact_sensor"]

    ee_body_name = "panda_hand"
    ee_body_idx = robot.body_names.index(ee_body_name)

    for _ in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    robot_initial_pos = robot.data.body_pos_w[0, ee_body_idx].cpu().numpy()
    geomagic_initial_pos = device.advance()[:3].cpu().numpy()

    arm_joint_names = [f"panda_joint{i}" for i in range(1, 7)]
    arm_joint_indices = [robot.joint_names.index(n) for n in arm_joint_names]

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=scene.num_envs, device=sim.device)
    initial_ee_quat = robot.data.body_quat_w[:, ee_body_idx]
    ik_controller.set_command(command=torch.zeros(scene.num_envs, 3, device=sim.device), ee_quat=initial_ee_quat)

    prev_grey = False
    prev_white = False
    gripper_open = True
    gripper_target = 0.04
    ee_rotation_angle = robot.data.joint_pos[0, 6].item()
    rotation_step = np.pi / 3

    print("\n[INFO] Teleoperation ready!")
    print("  Move stylus : Control end-effector position")
    print("  Grey  button: Toggle gripper open / close")
    print("  White button: Rotate end-effector 60 degrees\n")

    while simulation_app.is_running():
        # Periodic full reset
        if count % 10000 == 0:
            count = 1
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            )

            cube_state = cube.data.default_root_state.clone()
            cube_state[:, :3] += scene.env_origins
            cube.write_root_pose_to_sim(cube_state[:, :7])
            cube.write_root_velocity_to_sim(cube_state[:, 7:])

            scene.reset()
            device.reset()
            ik_controller.reset()

            robot_initial_pos = robot.data.body_pos_w[0, ee_body_idx].cpu().numpy()
            geomagic_initial_pos = device.advance()[:3].cpu().numpy()
            print("[INFO] Resetting robot state...")

        # Read device
        data = device.advance()
        geomagic_pos = data[:3]
        btn_grey = data[7].item() > 0.5
        btn_white = data[8].item() > 0.5

        # Grey button: toggle gripper on rising edge
        if btn_grey and not prev_grey:
            gripper_open = not gripper_open
            gripper_target = 0.04 if gripper_open else 0.0

        # White button: rotate EE on rising edge
        if btn_white and not prev_white:
            joint_7_limit = 3.0
            ee_rotation_angle += rotation_step
            if ee_rotation_angle > joint_7_limit:
                ee_rotation_angle = -joint_7_limit + (ee_rotation_angle - joint_7_limit)

        prev_grey = btn_grey
        prev_white = btn_white

        # IK position target
        target_pos = apply_geomagic_to_robot_mapping(geomagic_pos, geomagic_initial_pos, robot_initial_pos)
        target_pos_tensor = torch.tensor(target_pos, dtype=torch.float32, device=sim.device).unsqueeze(0)

        current_joint_pos = robot.data.joint_pos[:, arm_joint_indices]
        ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_body_idx, :, arm_joint_indices]

        ik_controller.set_command(command=target_pos_tensor, ee_quat=ee_quat_w)
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

        # Force feedback from finger contact sensors
        left_forces = left_sensor.data.net_forces_w[0, 0]
        right_forces = right_sensor.data.net_forces_w[0, 0]
        total_force = (left_forces + right_forces) * 0.5
        device.push_force(forces=total_force.unsqueeze(0), position=torch.tensor([0]))


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1 / 200)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, 1.0, 0.8], [0.4, 0.0, 0.0])

    scene_cfg = FrankaGeomagicSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    geomagic_cfg = GeomagicDeviceCfg(
        ros_namespace=args_cli.ros_namespace,
        pos_sensitivity=args_cli.pos_sensitivity,
        sim_device=args_cli.device,
        limit_force=2.0,
    )
    geomagic_device = GeomagicDevice(cfg=geomagic_cfg)
    print(f"[INFO] Geomagic connected: /{args_cli.ros_namespace}")

    sim.reset()
    run_simulator(sim, scene, geomagic_device)


if __name__ == "__main__":
    main()
    simulation_app.close()
