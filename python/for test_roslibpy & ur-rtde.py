# import roslibpy
# import time

# ros = roslibpy.Ros(host='172.16.10.171', port=9090)
# ros.run()

# def tf_callback(message):
#     # """Called when a new TF message arrives"""
#     for tf in message['transforms']:
#         parent = tf["header"]["frame_id"]
#         child = tf["child_frame_id"]
#         trans = tf["transform"]["translation"]
#         print(f"--- Transform: {parent} -> {child} ---")
#         print(f"Translation: x={trans['x']:.3f}, y={trans['y']:.3f}, z={trans['z']:.3f}\n")

# listener = roslibpy.Topic(ros, '/tf_static', 'tf2_msgs/msg/TFMessage')
# listener.subscribe(tf_callback)

# print("Listening to /tf_static... Press Ctrl+C to stop.")

# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     ros.terminate()


#!/usr/bin/env python3

import time
from datetime import datetime

import rtde_receive

# ==========================================================
# USER SETTINGS
# ==========================================================
ROBOT_IP = "192.168.56.101"    # <-- Change your robot IP here
UPDATE_RATE = 10             # Hz
# ==========================================================


def main():
    print(f"Connecting to UR10e at {ROBOT_IP}...")

    try:
        receiver = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Connected. Press Ctrl+C to stop.\n")

    period = 1.0 / UPDATE_RATE

    try:
        while True:
            start = time.time()

            # Read robot status
            joint_pos = receiver.getActualQ()
            joint_vel = receiver.getActualQd()
            tcp_pose = receiver.getActualTCPPose()
            tcp_force = receiver.getActualTCPForce()

            robot_mode = receiver.getRobotMode()
            safety_mode = receiver.getSafetyMode()
            runtime_state = receiver.getRuntimeState()

            speed_scaling = receiver.getSpeedScaling()

            print("=" * 70)
            print(datetime.now().strftime("%H:%M:%S"))
            print(f"Robot Mode   : {robot_mode}")
            print(f"Safety Mode  : {safety_mode}")
            print(f"Runtime State: {runtime_state}")
            print(f"Speed Scaling: {speed_scaling:.2f}")

            print("\nJoint Positions (rad)")
            for i, q in enumerate(joint_pos):
                print(f"  Joint {i+1}: {q:.4f}")

            print("\nJoint Velocities (rad/s)")
            for i, qd in enumerate(joint_vel):
                print(f"  Joint {i+1}: {qd:.4f}")

            print("\nTCP Pose [x, y, z, rx, ry, rz]")
            print([round(v, 4) for v in tcp_pose])

            print("\nTCP Force/Torque")
            print([round(v, 2) for v in tcp_force])

            elapsed = time.time() - start
            time.sleep(max(0, period - elapsed))

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        receiver.disconnect()


if __name__ == "__main__":
    main()