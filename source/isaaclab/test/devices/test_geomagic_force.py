import time

import torch

from isaaclab.devices import GeomagicDevice, GeomagicDeviceCfg

dev = GeomagicDevice(GeomagicDeviceCfg(limit_force=2.0))
print(dev)

print("Sending 1N force in +X for 1 second...")
forces = torch.tensor([[1.0, 0.0, 0.0]])
dev.push_force(forces, position=torch.tensor([0]))
time.sleep(1.0)

print("Zeroing force...")
dev.reset()

print("Done.")
dev.shutdown()
