import time

from isaaclab.devices import GeomagicDevice, GeomagicDeviceCfg

dev = GeomagicDevice(GeomagicDeviceCfg())
print(dev)
print("Move the stylus and press buttons. Ctrl+C to stop.\n")

try:
    while True:
        data = dev.advance()
        pos = data[:3].tolist()
        quat = data[3:7].tolist()
        btns = data[7:].tolist()
        print(
            f"pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
            f"quat=[{quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f}]  "
            f"grey={int(btns[0])}  white={int(btns[1])}",
            end="\r",
        )
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    dev.shutdown()
