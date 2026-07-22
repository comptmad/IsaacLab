# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Geomagic Touch teleoperation of the microsurgical robot.

Loads the pre-built microsurgical scene USD and controls arm1 via
differential IK, driven by the Geomagic Touch stylus position.

Prerequisites:
    ros2 launch omni_common omni_state.launch.py

Usage:
    ./isaaclab.sh -p scripts/demos/geomagic_microsurgical.py

    # Print joint / body names and exit (useful for debugging)
    ./isaaclab.sh -p scripts/demos/geomagic_microsurgical.py --list_info

    # Tune motion scale (default 0.1 keeps movements surgical-scale)
    ./isaaclab.sh -p scripts/demos/geomagic_microsurgical.py --pos_sensitivity 0.05

Button mapping:
    Grey  button : Reset robot to default joint pose
    White button : (reserved — gripper to be added later)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Geomagic Touch teleoperation of the microsurgical robot.")
parser.add_argument(
    "--list_info",
    action="store_true",
    help="Print joint names and body names then exit.",
)
parser.add_argument(
    "--pos_sensitivity",
    type=float,
    default=0.1,
    help="Scale applied to Geomagic delta position (default 0.1 for surgical precision).",
)
parser.add_argument(
    "--ros_namespace",
    type=str,
    default="Geomagic",
    help="ROS2 topic namespace matching omni_name in the driver launch file.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import GeomagicDevice, GeomagicDeviceCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MICROBOT_USD_PATH = str(Path(__file__).resolve().parents[4] / "assets" / "flattened_microbot.usd")
MICROBOT_PRIM_PATH = "/World/Microrobot"

ARM1_JOINT_NAMES = [
    "RevoluteJoint1",
    "RevoluteJoint2",
    "RevoluteJoint3",
    "RevoluteJoint4",
    "RevoluteJoint5",
]

# arm1_link5 is the last rigid body; Tool_sim_intoolkit lives inside it as
# an Xform — use the link itself as the IK end-effector body.
EE_BODY_NAME = "arm1_link5"


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

@configclass
class MicrosurgicalSceneCfg(InteractiveSceneCfg):
    """Scene wrapping the pre-built microsurgical USD.

    The USD is loaded as a sublayer before this scene is initialised, so all
    prims (including /World/Microrobot) already exist — no spawning needed.
    """

    robot: Articulation = ArticulationCfg(
        prim_path=MICROBOT_PRIM_PATH,
        spawn=None,  # prim already exists from the USD sublayer
        actuators={
            "arm1": ImplicitActuatorCfg(
                joint_names_expr=ARM1_JOINT_NAMES,
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
                stiffness=5000.0,
                damping=5000.0,
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
    geomagic: GeomagicDevice,
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
        geomagic.shutdown()
        return

    # ---------------------------------------------------------------------- #
    # Hold at the USD home pose during warm-up so the arm doesn't fall before
    # we record the initial EE position as the teleoperation reference frame.
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

    # Record starting poses as the teleoperation reference frame
    robot_initial_ee_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
    geomagic_initial_pos = geomagic.advance()[:3].clone()

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.1},
    )
    ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=sim.device)
    ee_quat = robot.data.body_quat_w[:, ee_body_idx]
    ik_controller.set_command(command=robot_initial_ee_pos.unsqueeze(0), ee_quat=ee_quat)

    prev_grey = False

    print("\n[INFO] Teleoperation ready!")
    print(f"  EE start pos   : {robot_initial_ee_pos.tolist()}")
    print(f"  Pos sensitivity: {args_cli.pos_sensitivity}")
    print("  Move stylus    : Control arm1 end-effector position")
    print("  Grey  button   : Reset robot to default pose")
    print("  White button   : (reserved)\n")

    while simulation_app.is_running():
        # ------------------------------------------------------------------ #
        # Periodic full reset every 10 000 steps
        # ------------------------------------------------------------------ #
        if count % 10000 == 0:
            count = 1
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            scene.reset()
            geomagic.reset()
            ik_controller.reset()
            robot_initial_ee_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
            geomagic_initial_pos = geomagic.advance()[:3].clone()
            print("[INFO] Periodic reset complete.")

        # ------------------------------------------------------------------ #
        # Read Geomagic device
        # ------------------------------------------------------------------ #
        data = geomagic.advance()
        geomagic_pos = data[:3].to(sim.device)
        btn_grey = data[7].item() > 0.5
        # data[8] = white button — reserved for gripper

        # Grey button rising edge: reset to default pose
        if btn_grey and not prev_grey:
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            scene.reset()
            ik_controller.reset()
            robot_initial_ee_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
            geomagic_initial_pos = geomagic_pos.clone()
            print("[INFO] Manual reset triggered.")

        prev_grey = btn_grey

        # ------------------------------------------------------------------ #
        # Map Geomagic delta → robot EE target
        #
        # Geomagic axes (stylus tip, device on table):
        #   X: left / right
        #   Y: up / down
        #   Z: toward / away from user
        #
        # Mapped to robot world frame as:
        #   robot ΔX ← geomagic -ΔZ  (push forward → robot reaches forward)
        #   robot ΔY ← geomagic  ΔX  (right        → robot moves right)
        #   robot ΔZ ← geomagic  ΔY  (lift          → robot lifts)
        #
        # Tune --pos_sensitivity if motion range feels too large / small.
        # Swap / negate axes here if directions feel inverted.
        # ------------------------------------------------------------------ #
        delta = (geomagic_pos - geomagic_initial_pos) * args_cli.pos_sensitivity
        mapped_delta = torch.stack([-delta[2], delta[0], delta[1]])
        target_pos = robot_initial_ee_pos.to(sim.device) + mapped_delta

        # ------------------------------------------------------------------ #
        # Differential IK
        # ------------------------------------------------------------------ #
        current_joint_pos = robot.data.joint_pos[:, arm1_joint_indices]
        ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_body_idx, :, arm1_joint_indices]

        ik_controller.set_command(command=target_pos.unsqueeze(0), ee_quat=ee_quat_w)
        joint_pos_des = ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, current_joint_pos)

        joint_pos_target = robot.data.joint_pos[0].clone()
        joint_pos_target[arm1_joint_indices] = joint_pos_des[0]
        robot.set_joint_position_target(joint_pos_target.unsqueeze(0))

        # ------------------------------------------------------------------ #
        # Step simulation (same cadence as geomagic_teleoperation.py)
        # ------------------------------------------------------------------ #
        for _ in range(5):
            scene.write_data_to_sim()
            sim.step()

        scene.update(sim_dt)
        count += 1

        # TODO: add force feedback once contact sensors are configured on the tool tip


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1 / 200)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.2])

    # Load the full scene (ground, table, phantom, microrobot) from the USD.
    # Must happen before InteractiveScene is created so the prims exist.
    load_scene_usd(MICROBOT_USD_PATH)

    scene_cfg = MicrosurgicalSceneCfg(num_envs=1, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)

    # Connect Geomagic device (requires ROS2 driver to be running)
    geomagic_cfg = GeomagicDeviceCfg(
        ros_namespace=args_cli.ros_namespace,
        pos_sensitivity=1.0,  # raw sensitivity; further scaled in the mapping above
        sim_device=args_cli.device,
        limit_force=2.0,
    )
    geomagic = GeomagicDevice(cfg=geomagic_cfg)
    print(f"[INFO] Geomagic connected: /{args_cli.ros_namespace}")

    sim.reset()

    run_simulator(sim, scene, geomagic)
    geomagic.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
