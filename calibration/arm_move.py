import os
import yaml
import time
from pydobotplus import Dobot


def get_dobot_port():
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]

if __name__ == "__main__":
    port = get_dobot_port()
    device = Dobot(port=port)
    print("Homing the robotic arm...")
    device.home()
    print("Arm homed. Starting demo moves to three points.")

    # Define three points
    points = [
        (200, 0, 50),
        (250, 50, 50),
        (200, -50, 50),
        (250, 0, 50)
    ]

    for i, (x, y, z) in enumerate(points, 1):
        print(f"Moving to point {i}: ({x}, {y}, {z})")
        device.move_to(x, y, z)
        time.sleep(2)  # Wait 2 seconds between moves

    device.close()