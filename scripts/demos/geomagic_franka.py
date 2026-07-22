# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Geomagic Touch teleoperation of a Franka Panda arm using franka_test.usd.

Prerequisites
-------------
    ros2 launch omni_common omni_state.launch.py

Usage
-----
    ./isaaclab.sh -p scripts/demos/geomagic_franka.py

    # With custom ROS namespace
    ./isaaclab.sh -p scripts/demos/geomagic_franka.py --ros_namespace Geomagic

    # Tune sensitivity
    ./isaaclab.sh -p scripts/demos/geomagic_franka.py --pos_sensitivity 2.0

Button mapping
--------------
    Grey  button: Toggle gripper open / close
    White button: Rotate end-effector 60 degrees
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Geomagic Touch teleoperation of Franka Panda (franka_test.usd).")
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

import omni.usd
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import GeomagicDevice, GeomagicDeviceCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

FRANKA_TEST_USD = "/home/comptmad/Downloads/IssacSim_envs/franka_test.usd"

WORKSPACE_LIMITS = {
    "x": (-0.5, 0.5),
    "y": (-0.8, 0.2),
    "z": (0.3, 1.4),
}


def apply_geomagic_to_robot_mapping(
    geomagic_pos: np.ndarray | torch.Tensor,
    geomagic_initial_pos: np.ndarray,
    robot_initial_pos: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Map Geomagic Touch workspace to Franka end-effector world frame."""
    if isinstance(geomagic_pos, torch.Tensor):
        geomagic_pos = geomagic_pos.cpu().numpy()
    if isinstance(robot_initial_pos, torch.Tensor):
        robot_initial_pos = robot_initial_pos.cpu().numpy()

    delta = geomagic_pos - geomagic_initial_pos
    robot_offset = np.array([-delta[2], delta[0], delta[1]]) * args_cli.pos_sensitivity
    robot_pos = robot_initial_pos + robot_offset

    robot_pos[0] = np.clip(robot_pos[0], WORKSPACE_LIMITS["x"][0], WORKSPACE_LIMITS["x"][1])
    robot_pos[1] = np.clip(robot_pos[1], WORKSPACE_LIMITS["y"][0], WORKSPACE_LIMITS["y"][1])
    robot_pos[2] = np.clip(robot_pos[2], WORKSPACE_LIMITS["z"][0], WORKSPACE_LIMITS["z"][1])

    return robot_pos


@configclass
class FrankaGeomagicSceneCfg(InteractiveSceneCfg):
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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, device: GeomagicDevice) -> None:
    """Main simulation loop with Geomagic teleoperation and force feedback."""
    sim_dt = sim.get_physics_dt()
    count = 1

    robot: Articulation = scene["robot"]
    left_sensor: ContactSensor = scene["left_finger_contact_sensor"]
    right_sensor: ContactSensor = scene["right_finger_contact_sensor"]

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

    robot_initial_pos = robot.data.body_pos_w[0, ee_body_idx].cpu().numpy()
    geomagic_initial_pos = device.advance()[:3].cpu().numpy()

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=scene.num_envs, device=sim.device)
    initial_ee_quat = robot.data.body_quat_w[:, ee_body_idx]
    ik_controller.set_command(command=torch.tensor(robot_initial_pos, device=sim.device).unsqueeze(0), ee_quat=initial_ee_quat)

    prev_grey = False
    prev_white = False
    gripper_open = True
    gripper_target = 0.04
    ee_rotation_angle = robot.data.joint_pos[0, 6].item()
    rotation_step = np.pi / 3

    print("\n[INFO] Geomagic teleoperation ready!")
    print(f"  EE start pos   : {robot_initial_pos.tolist()}")
    print(f"  Pos sensitivity: {args_cli.pos_sensitivity}")
    print("  Move stylus    : Control end-effector position")
    print("  Grey  button   : Toggle gripper open / close")
    print("  White button   : Rotate end-effector 60 degrees\n")

    while simulation_app.is_running():
        if count % 10000 == 0:
            count = 1
            reset_robot()
            scene.reset()
            device.reset()
            ik_controller.reset()
            for _ in range(5):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
            robot_initial_pos = robot.data.body_pos_w[0, ee_body_idx].cpu().numpy()
            geomagic_initial_pos = device.advance()[:3].cpu().numpy()
            print("[INFO] Periodic reset complete.")

        data = device.advance()
        geomagic_pos = data[:3]
        btn_grey  = data[7].item() > 0.5
        btn_white = data[8].item() > 0.5

        if btn_grey and not prev_grey:
            gripper_open = not gripper_open
            gripper_target = 0.04 if gripper_open else 0.0

        if btn_white and not prev_white:
            joint_7_limit = 3.0
            ee_rotation_angle += rotation_step
            if ee_rotation_angle > joint_7_limit:
                ee_rotation_angle = -joint_7_limit + (ee_rotation_angle - joint_7_limit)

        prev_grey  = btn_grey
        prev_white = btn_white

        target_pos = apply_geomagic_to_robot_mapping(geomagic_pos, geomagic_initial_pos, robot_initial_pos)
        target_pos_tensor = torch.tensor(target_pos, dtype=torch.float32, device=sim.device).unsqueeze(0)

        current_joint_pos = robot.data.joint_pos[:, arm_joint_indices]
        ee_pos_w   = robot.data.body_pos_w[:, ee_body_idx]
        ee_quat_w  = robot.data.body_quat_w[:, ee_body_idx]
        jacobian   = robot.root_physx_view.get_jacobians()[:, ee_body_idx, :, arm_joint_indices]

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

        left_forces  = left_sensor.data.net_forces_w[0, 0]
        right_forces = right_sensor.data.net_forces_w[0, 0]
        total_force  = (left_forces + right_forces) * 0.5
        device.push_force(forces=total_force.unsqueeze(0), position=torch.tensor([0]))


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

    scene_cfg = FrankaGeomagicSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    _bake_base_rotation(args_cli.num_envs)

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
