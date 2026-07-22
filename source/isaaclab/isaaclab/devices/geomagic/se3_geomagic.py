"""Geomagic Touch device controller for SE3 control with force feedback."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

try:
    import rclpy
    import rclpy.node
    from geometry_msgs.msg import PoseStamped
    from omni_msgs.msg import OmniButtonEvent, OmniFeedback

    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False

from ..device_base import DeviceBase, DeviceCfg
from ..retargeter_base import RetargeterBase


class GeomagicDevice(DeviceBase):
    """A Geomagic Touch device controller for sending SE(3) commands with force feedback.

    Communicates via ROS2 using the Geomagic_Touch_ROS2 driver. Requires
    ``ros2 launch omni_common omni_state.launch.py`` to be running.

    The device provides raw data as a 9-element tensor:
    ``[x, y, z, qx, qy, qz, qw, btn_grey, btn_white]``
    """

    def __init__(self, cfg: GeomagicDeviceCfg, retargeters: list[RetargeterBase] | None = None):
        """Initialize the Geomagic Touch device interface.

        Args:
            cfg: Configuration object for Geomagic device settings.
            retargeters: Optional list of retargeting components.

        Raises:
            ImportError: If rclpy or omni_msgs are not available.
            RuntimeError: If no pose message is received within the timeout.
        """
        super().__init__(retargeters)

        if not RCLPY_AVAILABLE:
            raise ImportError(
                "rclpy and omni_msgs are required for GeomagicDevice. "
                "Source your Geomagic_Touch_ROS2 workspace before launching IsaacLab."
            )

        # Store configuration
        self._ros_namespace = cfg.ros_namespace
        self.pos_sensitivity = cfg.pos_sensitivity
        self._sim_device = cfg.sim_device
        self.limit_force = cfg.limit_force

        # Thread-safe data cache
        self.cached_data = {
            "position": np.zeros(3, dtype=np.float32),
            "quaternion": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "buttons": {"grey": False, "white": False},
            "connected": False,
        }
        self.data_lock = threading.Lock()
        self.force_lock = threading.Lock()

        self._additional_callbacks: dict[str, Callable] = {}
        self._prev_buttons = {"grey": False, "white": False}

        # ROS2 node, subscriptions, and force publisher
        self._rclpy_owner = not rclpy.ok()
        if self._rclpy_owner:
            rclpy.init()

        self._node = rclpy.node.Node("geomagic_device_node")

        ns = self._ros_namespace
        self._pose_sub = self._node.create_subscription(
            PoseStamped, f"/{ns}/pose", self._pose_cb, 1
        )
        self._button_sub = self._node.create_subscription(
            OmniButtonEvent, f"/{ns}/button", self._button_cb, 1
        )
        self._force_pub = self._node.create_publisher(OmniFeedback, f"/{ns}/force_feedback", 1)

        # Spin in a background thread so it doesn't block the sim
        self._stop_spin = threading.Event()
        self._ros_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._ros_thread.start()

        # Wait for first pose message (confirms driver is running)
        timeout = 5.0
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            with self.data_lock:
                if self.cached_data["connected"]:
                    break
            time.sleep(0.1)

        with self.data_lock:
            if not self.cached_data["connected"]:
                raise RuntimeError(
                    f"No pose received from /{ns}/pose within {timeout}s. "
                    "Is 'ros2 launch omni_common omni_state.launch.py' running?"
                )

    def __del__(self):
        """Shutdown the ROS2 node on deletion."""
        self.shutdown()

    def shutdown(self):
        """Stop the spin thread, destroy the node, and (if we own rclpy) shut it down."""
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True

        if hasattr(self, "_stop_spin"):
            self._stop_spin.set()
        if hasattr(self, "_ros_thread") and self._ros_thread.is_alive():
            self._ros_thread.join(timeout=2.0)
        if hasattr(self, "_node") and rclpy.ok():
            self._node.destroy_node()
        if getattr(self, "_rclpy_owner", False) and rclpy.ok():
            rclpy.shutdown()

    def _spin_loop(self):
        """Background thread: spin rclpy until stop is requested."""
        while rclpy.ok() and not self._stop_spin.is_set():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def __str__(self) -> str:
        """Returns: A string containing the information of the device."""
        msg = f"Geomagic Touch Device Controller: {self.__class__.__name__}\n"
        msg += f"\tROS namespace: /{self._ros_namespace}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tOutput: [x, y, z, qx, qy, qz, qw, btn_grey, btn_white]\n"
        return msg

    def reset(self):
        """Reset the device internal state."""
        with self.force_lock:
            msg = OmniFeedback()
            self._force_pub.publish(msg)  # zero force

        self._prev_buttons = {"grey": False, "white": False}

    def add_callback(self, key: str, func: Callable):
        """Add a function to be called on button rising-edge press.

        Args:
            key: Button name. Valid values are ``"grey"`` and ``"white"``.
            func: Zero-argument callable to invoke on press.
        """
        if key not in ("grey", "white"):
            raise ValueError(f"Invalid button key: '{key}'. Valid keys are 'grey' and 'white'.")
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Read the current device state and fire any button callbacks.

        Returns:
            torch.Tensor: 9-element tensor ``[x, y, z, qx, qy, qz, qw, btn_grey, btn_white]``.

        Raises:
            RuntimeError: If the device is not connected.
        """
        with self.data_lock:
            if not self.cached_data["connected"]:
                raise RuntimeError("Geomagic device not connected.")

            position = self.cached_data["position"].copy() * self.pos_sensitivity
            quaternion = self.cached_data["quaternion"].copy()
            btn_grey = self.cached_data["buttons"]["grey"]
            btn_white = self.cached_data["buttons"]["white"]

        # Rising-edge callbacks — executed outside lock to prevent deadlock
        for key, current in (("grey", btn_grey), ("white", btn_white)):
            if current and not self._prev_buttons[key]:
                if key in self._additional_callbacks:
                    self._additional_callbacks[key]()
            self._prev_buttons[key] = current

        button_states = np.array(
            [1.0 if btn_grey else 0.0, 1.0 if btn_white else 0.0], dtype=np.float32
        )

        command = np.concatenate([position, quaternion, button_states])
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def push_force(self, forces: torch.Tensor, position: torch.Tensor) -> None:
        """Publish force feedback to the Geomagic Touch device.

        Forces are clipped to ``[-limit_force, limit_force]`` for safety.

        Args:
            forces: Tensor of shape ``(N, 3)`` with forces ``[fx, fy, fz]``.
            position: Tensor of indices selecting which row(s) of ``forces`` to use.
        """
        if forces.shape[0] == 0:
            raise ValueError("No forces provided.")

        selected = forces[position] if position.ndim > 0 else forces[position].unsqueeze(0)
        force = selected.sum(dim=0)
        force = force.cpu().numpy() if force.is_cuda else force.numpy()

        fx = float(np.clip(force[0], -self.limit_force, self.limit_force))
        fy = float(np.clip(force[1], -self.limit_force, self.limit_force))
        fz = float(np.clip(force[2], -self.limit_force, self.limit_force))

        msg = OmniFeedback()
        msg.force.x = fx
        msg.force.y = fy
        msg.force.z = fz
        self._force_pub.publish(msg)

    # ------------------------------------------------------------------
    # ROS2 callbacks (run in the spin thread)
    # ------------------------------------------------------------------

    def _pose_cb(self, msg: PoseStamped) -> None:
        """Update cached position and quaternion from /phantom/pose."""
        p = msg.pose.position
        q = msg.pose.orientation
        with self.data_lock:
            self.cached_data["position"] = np.array([p.x, p.y, p.z], dtype=np.float32)
            self.cached_data["quaternion"] = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
            self.cached_data["connected"] = True

    def _button_cb(self, msg: OmniButtonEvent) -> None:
        """Update cached button states from /phantom/button."""
        with self.data_lock:
            self.cached_data["buttons"]["grey"] = bool(msg.grey_button)
            self.cached_data["buttons"]["white"] = bool(msg.white_button)


@dataclass
class GeomagicDeviceCfg(DeviceCfg):
    """Configuration for the Geomagic Touch device.

    Attributes:
        ros_namespace: ROS2 topic namespace (matches ``omni_name`` in the driver launch file).
        pos_sensitivity: Scale factor applied to the raw position.
        limit_force: Maximum force magnitude in Newtons (safety clip).
    """

    ros_namespace: str = "Geomagic"
    pos_sensitivity: float = 1.0
    limit_force: float = 2.0
    class_type: type[DeviceBase] = GeomagicDevice
