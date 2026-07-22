# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Full ROS2 pipeline teleoperation of the microsurgical robot via OmniGraph.

Architecture
────────────
  Keyboard
     ↓
  Python DifferentialIK  (computes desired joint positions)
     ↓ rclpy publish
  /microbot/joint_commands  (sensor_msgs/JointState)
     ↓ OmniGraph ROS2SubscribeJointState
  IsaacArticulationController  ──drives──►  robot joints
     ↑ (reads current state)
  IsaacArticulationState / PublishJointState
     ↓ OmniGraph ROS2PublishJointState
  /microbot/joint_states   → RViz, rosbag, external monitors
  /clock                   → sim time for external nodes
  /tf                      → link poses

Isaac Sim is the physics server.  The IK result travels over a real ROS2
topic before it reaches the robot — so any external ROS2 node can intercept,
record, or replace the joint commands without touching this script.

Key bindings
────────────
  I / K   +X / -X       U / O   +Z / -Z
  J / H   +Y / -Y       M       reset

Usage
─────
  ./isaaclab.sh -p scripts/demos/omnigraph_microsurgical.py

  # Check topics in another terminal
  ros2 topic list
  ros2 topic echo /microbot/joint_states
  ros2 topic hz  /microbot/joint_commands
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OmniGraph full-pipeline teleoperation of the microsurgical robot.")
parser.add_argument("--list_info", action="store_true", help="Print joint/body names then exit.")
parser.add_argument("--pos_sensitivity", type=float, default=0.005,
                    help="Metres per physics step per held key (default 0.005).")
parser.add_argument("--ros_domain_id", type=int, default=0,
                    help="ROS2 DDS domain ID (default 0).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Enable ROS2 bridge — must happen before importing rclpy or og.Controller
# ---------------------------------------------------------------------------
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import math

import numpy as np
import torch

import carb
import omni.graph.core as og
import usdrt.Sdf
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Custom keyboard (non-conflicting bindings)
# ---------------------------------------------------------------------------

class MicrosurgicalKeyboard(Se3Keyboard):
    def _create_key_bindings(self):
        self._INPUT_KEY_MAPPING = {
            "I": np.asarray([1.0,  0.0,  0.0]) * self.pos_sensitivity,
            "K": np.asarray([-1.0, 0.0,  0.0]) * self.pos_sensitivity,
            "J": np.asarray([0.0,  1.0,  0.0]) * self.pos_sensitivity,
            "H": np.asarray([0.0, -1.0,  0.0]) * self.pos_sensitivity,
            "U": np.asarray([0.0,  0.0,  1.0]) * self.pos_sensitivity,
            "O": np.asarray([0.0,  0.0, -1.0]) * self.pos_sensitivity,
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

MICROBOT_USD_PATH = str(Path(__file__).resolve().parents[2] / "assets" / "flattened_microbot.usd")
MICROBOT_PRIM_PATH = "/World/Microrobot/body_link"

ARM1_JOINT_NAMES = [
    "RevoluteJoint1",
    "RevoluteJoint2",
    "RevoluteJoint3",
    "RevoluteJoint4",
    "RevoluteJoint5",
]

EE_BODY_NAME = "arm1_link5"

TOPIC_JOINT_COMMANDS = "microbot/joint_commands"
TOPIC_JOINT_STATES   = "microbot/joint_states"
TOPIC_CLOCK          = "clock"
TOPIC_TF             = "tf"

GRAPH_PATH = "/ROS2ActionGraph"


# ---------------------------------------------------------------------------
# Scene
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
# OmniGraph
# ---------------------------------------------------------------------------

def setup_ros2_omnigraph(robot_prim_path: str, domain_id: int) -> None:
    """Build the full bidirectional ROS2 action graph.

    Data flow
    ─────────
    OnImpulseEvent
      ├─execOut──► ROS2PublishJointState  → /microbot/joint_states
      ├─execOut──► ROS2PublishClock       → /clock
      ├─execOut──► ROS2PublishTransformTree → /tf
      └─execOut──► ROS2SubscribeJointState (reads /microbot/joint_commands)
                        └─execOut (fires only when msg received)
                              └──► IsaacArticulationController → robot joints

    The graph fires once per IK cycle — triggered manually from the Python
    loop via og.Controller.set(enableImpulse).  The Python loop:
      1. Computes IK
      2. Publishes result to /microbot/joint_commands via rclpy
      3. Triggers this graph — SubscribeJointState reads the just-published
         message and ArticulationController applies the joint positions
      4. Steps the simulation
    """
    prim_sdf = usdrt.Sdf.Path(robot_prim_path)

    try:
        og.Controller.edit(
            {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    # Manual trigger — fired once per IK cycle from Python
                    ("OnTick",              "omni.graph.action.OnImpulseEvent"),
                    # Utilities
                    ("ReadSimTime",         "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context",             "isaacsim.ros2.bridge.ROS2Context"),
                    # Publish side (state out)
                    ("PublishJointState",   "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("PublishClock",        "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ("PublishTF",           "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    # Subscribe side (commands in)
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    # Actuator: applies subscribed joint positions to the robot
                    ("ArtController",       "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                og.Controller.Keys.CONNECT: [
                    # ── Publish chain ──────────────────────────────────────
                    ("OnTick.outputs:execOut",                    "PublishJointState.inputs:execIn"),
                    ("OnTick.outputs:execOut",                    "PublishClock.inputs:execIn"),
                    ("OnTick.outputs:execOut",                    "PublishTF.inputs:execIn"),
                    # ROS2 context
                    ("Context.outputs:context",                   "PublishJointState.inputs:context"),
                    ("Context.outputs:context",                   "PublishClock.inputs:context"),
                    ("Context.outputs:context",                   "PublishTF.inputs:context"),
                    ("Context.outputs:context",                   "SubscribeJointState.inputs:context"),
                    # Timestamps
                    ("ReadSimTime.outputs:simulationTime",        "PublishJointState.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime",        "PublishClock.inputs:timeStamp"),

                    # ── Subscribe → control chain ──────────────────────────
                    # OnTick triggers the subscriber
                    ("OnTick.outputs:execOut",                    "SubscribeJointState.inputs:execIn"),
                    # SubscribeJointState.outputs:execOut fires ONLY when a
                    # new message is received — gates the articulation controller
                    ("SubscribeJointState.outputs:execOut",       "ArtController.inputs:execIn"),
                    # Data: subscribed joint names + positions → controller
                    ("SubscribeJointState.outputs:jointNames",    "ArtController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand","ArtController.inputs:positionCommand"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    # DDS domain
                    ("Context.inputs:domain_id",                  domain_id),
                    # Publish topics
                    ("PublishJointState.inputs:topicName",        TOPIC_JOINT_STATES),
                    ("PublishJointState.inputs:targetPrim",       [prim_sdf]),
                    ("PublishClock.inputs:topicName",             TOPIC_CLOCK),
                    ("PublishTF.inputs:topicName",                TOPIC_TF),
                    ("PublishTF.inputs:targetPrims",              [prim_sdf]),
                    # Subscribe topic
                    ("SubscribeJointState.inputs:topicName",      TOPIC_JOINT_COMMANDS),
                    # Articulation root for the controller
                    ("ArtController.inputs:robotPath",            robot_prim_path),
                ],
            },
        )
        print(f"[INFO] OmniGraph ready at {GRAPH_PATH}  (domain_id={domain_id})")
        print(f"[INFO]   IK publishes to   → /{TOPIC_JOINT_COMMANDS}")
        print(f"[INFO]   State visible on  → /{TOPIC_JOINT_STATES}")
        print(f"[INFO]   Clock on          → /{TOPIC_CLOCK}")
        print(f"[INFO]   TF on             → /{TOPIC_TF}")
    except Exception as exc:
        print(f"[ERROR] OmniGraph setup failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scene_usd(usd_path: str) -> None:
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
    ros_domain_id: int = 0,
) -> None:
    """Full pipeline: keyboard → Python IK → ROS2 → OmniGraph → robot."""
    # Import Isaac Sim's bundled rclpy (available after ros2.bridge is enabled)
    import rclpy
    from sensor_msgs.msg import JointState as JointStateMsg

    sim_dt = sim.get_physics_dt()
    count  = 1

    robot: Articulation = scene["robot"]
    arm1_joint_indices  = [robot.joint_names.index(n) for n in ARM1_JOINT_NAMES]
    ee_body_idx         = robot.body_names.index(EE_BODY_NAME)

    # Diagnostic
    if args_cli.list_info:
        print("\n=== Joint names ===")
        for i, n in enumerate(robot.joint_names):
            print(f"  [{i:2d}] {n}")
        print("\n=== Body names ===")
        for i, n in enumerate(robot.body_names):
            print(f"  [{i:2d}] {n}")
        return

    # ---------------------------------------------------------------------- #
    # Warm-up: settle the arm at the USD home pose.                           #
    # Uses Python-direct control (set_joint_position_target).                 #
    # OmniGraph is NOT set up yet, so no ROS2 publishing during warm-up.     #
    # ---------------------------------------------------------------------- #
    home_deg = [66.5, -10.7, 40.4, 53.4, 7.8]
    default_joint_pos = torch.tensor(
        [[math.radians(d) for d in home_deg]], device=sim.device
    )
    robot.write_joint_state_to_sim(default_joint_pos, robot.data.default_joint_vel.clone())
    for _ in range(50):
        robot.set_joint_position_target(default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # ---------------------------------------------------------------------- #
    # Build OmniGraph now — after warm-up so PhysX tensors are initialised.  #
    # ---------------------------------------------------------------------- #
    setup_ros2_omnigraph(MICROBOT_PRIM_PATH, domain_id=ros_domain_id)

    # ---------------------------------------------------------------------- #
    # IK setup                                                                #
    # ---------------------------------------------------------------------- #
    target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.1},
    )
    ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=sim.device)
    ik_controller.set_command(
        command=target_pos.unsqueeze(0),
        ee_quat=robot.data.body_quat_w[:, ee_body_idx],
    )

    # ---------------------------------------------------------------------- #
    # ROS2 publisher — IK results go out as JointState on /microbot/joint_commands
    # OmniGraph SubscribeJointState picks them up and feeds ArticulationController.
    # ---------------------------------------------------------------------- #
    rclpy.init()
    ik_node = rclpy.create_node("microbot_ik_controller")
    cmd_pub = ik_node.create_publisher(JointStateMsg, TOPIC_JOINT_COMMANDS, 10)

    # Convenience: attribute path to manually trigger the OmniGraph each cycle
    impulse_attr = og.Controller.attribute(f"{GRAPH_PATH}/OnTick.state:enableImpulse")

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

    print("\n[INFO] Full ROS2 pipeline active.")
    print(f"  EE start pos : {target_pos.tolist()}")
    print(f"  I/K J/H U/O  : move EE in X/Y/Z     M : reset\n")

    while simulation_app.is_running():

        # Periodic full reset
        if count % 10000 == 0:
            count = 1
            robot.write_joint_state_to_sim(default_joint_pos, robot.data.default_joint_vel.clone())
            scene.reset()
            keyboard.reset()
            ik_controller.reset()
            target_pos = robot.data.body_pos_w[0, ee_body_idx].clone()
            print("[INFO] Periodic reset.")

        # ── 1. Keyboard → EE target ───────────────────────────────────────
        delta_pos  = keyboard.advance()[:3].to(sim.device)
        target_pos = target_pos + delta_pos

        # ── 2. Differential IK ────────────────────────────────────────────
        current_joint_pos = robot.data.joint_pos[:, arm1_joint_indices]
        ee_pos_w          = robot.data.body_pos_w[:, ee_body_idx]
        ee_quat_w         = robot.data.body_quat_w[:, ee_body_idx]
        jacobian          = robot.root_physx_view.get_jacobians()[
            :, ee_body_idx, :, arm1_joint_indices
        ]
        ik_controller.set_command(command=target_pos.unsqueeze(0), ee_quat=ee_quat_w)
        joint_pos_des = ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, current_joint_pos)

        # ── 3. Publish IK result over ROS2 ───────────────────────────────
        # Only arm1 joints are commanded; the OmniGraph ArticulationController
        # will move only those joints (others keep their current drive targets).
        msg          = JointStateMsg()
        sim_time     = sim.current_time
        msg.header.stamp.sec     = int(sim_time)
        msg.header.stamp.nanosec = int((sim_time % 1.0) * 1e9)
        msg.name     = list(ARM1_JOINT_NAMES)
        msg.position = joint_pos_des[0].tolist()
        cmd_pub.publish(msg)

        # ── 4. Trigger OmniGraph ──────────────────────────────────────────
        # SubscribeJointState reads the just-published message from DDS.
        # Its execOut fires → IsaacArticulationController drives the robot.
        # PublishJointState / PublishClock / PublishTF also fire this tick.
        og.Controller.set(impulse_attr, True)

        # ── 5. Step simulation ────────────────────────────────────────────
        # scene.write_data_to_sim() is intentionally omitted here: we are not
        # using robot.set_joint_position_target(), so there is nothing in
        # IsaacLab's buffer to write.  The robot is driven purely by the
        # OmniGraph ArticulationController above.
        for _ in range(5):
            sim.step()

        scene.update(sim_dt)
        count += 1

    rclpy.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1 / 200)
    sim     = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.2])

    load_scene_usd(MICROBOT_USD_PATH)

    scene_cfg = MicrosurgicalSceneCfg(num_envs=1, env_spacing=0.0)
    scene     = InteractiveScene(scene_cfg)

    kb_cfg   = Se3KeyboardCfg(
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=0.0,
        gripper_term=False,
        sim_device=args_cli.device,
    )
    keyboard = MicrosurgicalKeyboard(cfg=kb_cfg)

    sim.reset()
    run_simulator(sim, scene, keyboard, ros_domain_id=args_cli.ros_domain_id)


if __name__ == "__main__":
    main()
    simulation_app.close()
