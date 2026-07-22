# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Drag-target teleoperation of the microsurgical robot — viewport grab-and-drag IK test.

A small sphere prim (/World/IK_Target) is spawned at the end-effector position after
warm-up.  Drag it in the Isaac Sim viewport (select it, press W for the translate
gizmo, then drag) and the arm will follow via differential IK.

This script is intended to verify that the IK pipeline is working correctly before
connecting physical hardware.

Usage
-----
    ./isaaclab.sh -p scripts/demos/drag_microsurgical.py

    # Print joint / body names and exit
    ./isaaclab.sh -p scripts/demos/drag_microsurgical.py --list_info

    # Reset the arm and re-centre the target sphere
    Press M in the viewport.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drag-target IK test for the microsurgical robot.")
parser.add_argument(
    "--list_info",
    action="store_true",
    help="Print joint names and body names then exit.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import torch

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from pxr import Gf, Usd, UsdGeom


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
IK_TARGET_PRIM_PATH = "/World/IK_Target"


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

@configclass
class MicrosurgicalSceneCfg(InteractiveSceneCfg):
    """Scene wrapping the pre-built microsurgical USD."""

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
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().subLayerPaths.append(usd_path)
    print(f"[INFO] Loaded scene USD: {usd_path}")


def create_target_sphere(position: torch.Tensor, radius: float = 0.005) -> None:
    """Spawn a small visible sphere at *position* (world frame, metres).

    The sphere is selectable and draggable in the viewport via the translate
    gizmo (W key).  Its world position is read back each sim step and used as
    the IK target.
    """
    stage = omni.usd.get_context().get_stage()
    sphere = UsdGeom.Sphere.Define(stage, IK_TARGET_PRIM_PATH)
    sphere.GetRadiusAttr().Set(radius)

    # Make it a bright colour so it is easy to see
    sphere.GetDisplayColorAttr().Set([(1.0, 0.2, 0.2)])  # red

    # Position it at the current EE location
    xform = UsdGeom.Xformable(sphere)
    xform.ClearXformOpOrder()
    op = xform.AddTranslateOp()
    pos = position.cpu().tolist()
    op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))

    print(f"[INFO] IK target sphere created at {pos}  (path: {IK_TARGET_PRIM_PATH})")
    print("[INFO] Select the sphere in the viewport, press W for the translate gizmo, then drag.")


def get_target_position(device: str) -> torch.Tensor:
    """Return the world-space position of the IK target sphere as a (3,) tensor."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(IK_TARGET_PRIM_PATH)
    xform = UsdGeom.Xformable(prim)
    transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = transform.ExtractTranslation()
    return torch.tensor([translation[0], translation[1], translation[2]],
                        dtype=torch.float32, device=device)


def move_target_sphere(position: torch.Tensor) -> None:
    """Teleport the IK target sphere to *position* (used on reset)."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(IK_TARGET_PRIM_PATH)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    op = xform.AddTranslateOp()
    pos = position.cpu().tolist()
    op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, keyboard: Se3Keyboard) -> None:
    """Main drag-target IK loop."""
    sim_dt = sim.get_physics_dt()
    count = 1

    robot: Articulation = scene["robot"]
    arm1_joint_indices = [robot.joint_names.index(n) for n in ARM1_JOINT_NAMES]
    ee_body_idx = robot.body_names.index(EE_BODY_NAME)

    if args_cli.list_info:
        print("\n=== Joint names ===")
        for i, name in enumerate(robot.joint_names):
            print(f"  [{i:2d}] {name}")
        print("\n=== Body names ===")
        for i, name in enumerate(robot.body_names):
            print(f"  [{i:2d}] {name}")
        return

    # ---------------------------------------------------------------------- #
    # Warm-up: hold the USD home pose so the arm settles
    # ---------------------------------------------------------------------- #
    home_deg = [66.5, -10.7, 40.4, 53.4, 7.8]
    default_joint_pos = torch.tensor(
        [[math.radians(d) for d in home_deg]], device=sim.device
    )
    initial_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    initial_joint_vel[0, 0] = 0.6  # joint 0 (base rotation)
    initial_joint_vel[0, 1] = 0.6  # joint 1
    initial_joint_vel[0, 2] = 0.6  # joint 2
    initial_joint_vel[0, 3] = 0.6  # joint 3
    initial_joint_vel[0, 4] = 0.6  # joint 4 (wrist)
    robot.write_joint_state_to_sim(default_joint_pos, initial_joint_vel)
    for _ in range(50):
        robot.set_joint_position_target(default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # EE settled position becomes both the IK start and the target sphere origin
    target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
    create_target_sphere(target_pos)

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
        initial_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
        initial_joint_vel[0, 0] = 0.6  # joint 0 (base rotation)
        initial_joint_vel[0, 1] = 0.6  # joint 1
        initial_joint_vel[0, 2] = 0.6  # joint 2
        initial_joint_vel[0, 3] = 0.6  # joint 3
        initial_joint_vel[0, 4] = 0.6  # joint 4 (wrist)
        robot.write_joint_state_to_sim(default_joint_pos, initial_joint_vel)
        scene.reset()
        ik_controller.reset()
        for _ in range(10):
            robot.set_joint_position_target(default_joint_pos)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
        move_target_sphere(target_pos)
        print("[INFO] Reset complete. Target sphere re-centred at:", target_pos.tolist())

    keyboard.add_callback("M", _on_reset)

    print("\n[INFO] Drag-target IK ready!")
    print(f"  IK target prim : {IK_TARGET_PRIM_PATH}")
    print("  Select the red sphere in the viewport, press W, then drag.")
    print("  M              : Reset robot and re-centre target sphere\n")

    while simulation_app.is_running():
        if count % 10000 == 0:
            count = 1
            initial_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
            initial_joint_vel[0, 0] = 0.6  # joint 0 (base rotation)
            initial_joint_vel[0, 1] = 0.6  # joint 1
            initial_joint_vel[0, 2] = 0.6  # joint 2
            initial_joint_vel[0, 3] = 0.6  # joint 3
            initial_joint_vel[0, 4] = 0.6  # joint 4 (wrist)
            robot.write_joint_state_to_sim(default_joint_pos, initial_joint_vel)
            scene.reset()
            ik_controller.reset()
            target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
            move_target_sphere(target_pos)
            print("[INFO] Periodic reset complete.")

        # Pump keyboard event queue (needed for M-key reset callback)
        keyboard.advance()

        # Read target position directly from the draggable USD sphere
        target_pos = get_target_position(sim.device)

        current_joint_pos = robot.data.joint_pos[:, arm1_joint_indices]
        ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_body_idx, :, arm1_joint_indices]

        ik_controller.set_command(command=target_pos.unsqueeze(0), ee_quat=ee_quat_w)
        joint_pos_des = ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, current_joint_pos)

        joint_pos_target = robot.data.joint_pos[0].clone()
        joint_pos_target[arm1_joint_indices] = joint_pos_des[0]
        robot.set_joint_position_target(joint_pos_target.unsqueeze(0))

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
        pos_sensitivity=0.0,
        rot_sensitivity=0.0,
        gripper_term=False,
        sim_device=args_cli.device,
    )
    keyboard = Se3Keyboard(cfg=kb_cfg)

    sim.reset()
    run_simulator(sim, scene, keyboard)


if __name__ == "__main__":
    main()
    simulation_app.close()
