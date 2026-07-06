#!/usr/bin/env python3
"""
Connect to a Mech-Eye Industrial 3D Camera, capture, and save:
  - Textured point cloud (PLY)
  - 2D image (PNG)
  - Depth map (raw 16bit-png) and rendered/colored depth map (png)

All outputs are saved into the active session folders managed by helpers/session_manager:
  - Initial point clouds
  - Initial images
  - Initial depth images
"""

import os
import json

import cv2
import numpy as np

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import print_camera_info, show_error

# import the session helpers (directory management only)
from helpers.session_manager import load_session, init_session


class CaptureTexturedPointCloud:
    def __init__(self):
        self.camera = Camera()
        self.frame_all_2d_3d = Frame2DAnd3D()

        # === Directories resolved per session ===
        # We set these later in main() after choosing/creating the session.
        self.output_pc_dir = None          # -> "Initial point clouds"
        self.output_img_dir = None         # -> "Initial images"
        self.output_depth_dir = None       # -> "Initial depth images"
        self.output_intrinsics_dir = None  # -> "Camera intrinsics"

    def discover_cameras(self):
        print("Discovering all available cameras...")
        camera_infos = Camera.discover_cameras()
        if len(camera_infos) == 0:
            print("No cameras found.")
            return []

        # Display available cameras
        for i, info in enumerate(camera_infos):
            print(f"\nCamera index : {i}")
            print_camera_info(info)
        return camera_infos

    def choose_camera_index(self, camera_infos):
        while True:
            index_str = input(
                f"\nEnter the index of the camera to connect (0–{len(camera_infos) - 1}, default 0): "
            ).strip()
            if index_str == "":
                return 0
            if index_str.isdigit() and 0 <= int(index_str) < len(camera_infos):
                return int(index_str)
            print("Invalid input. Please enter a valid index.")

    def ask_index(self, prompt: str, default: int = 1, lo: int = 1, hi: int = 99) -> int:
        """Ask for a capture index so files line up across pointcloud / image / depth."""
        while True:
            idx_str = input(f"\n{prompt} [{lo}..{hi}] (default {default}): ").strip()
            if idx_str == "":
                return default
            if idx_str.isdigit() and lo <= int(idx_str) <= hi:
                return int(idx_str)
            print(f"Invalid input. Please enter a number between {lo} and {hi}.")

    def capture_textured_point_cloud(self, idx: int):
        """Capture a textured point cloud and save it with an index-based filename."""
        filename = f"point_cloud_{idx:02d}.ply"
        output_file = os.path.join(self.output_pc_dir, filename)

        print("\nCapturing 2D + 3D frame...")
        status = self.camera.capture_2d_and_3d(self.frame_all_2d_3d)
        if not status.is_ok():
            show_error(status)
            return False

        print(f"Saving textured point cloud to: {output_file}")
        status = self.frame_all_2d_3d.save_textured_point_cloud(FileFormat_PLY, output_file)
        if not status.is_ok():
            show_error(status)
            return False

        print(f"Textured point cloud saved as: {output_file}")
        return True

    def capture_2d_image(self, idx: int):
        """Capture a 2D image and save it with an index-based filename."""
        frame_2d = Frame2D()

        print("\nCapturing 2D frame...")
        status = self.camera.capture_2d(frame_2d)
        if not status.is_ok():
            show_error(status)
            return False

        # Handle mono vs color camera output
        if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
            image2d = frame_2d.get_gray_scale_image()
        else:
            image2d = frame_2d.get_color_image()

        print(f"[debug] RGB shape = {image2d.data().shape}")


        filename = f"image_{idx:02d}.png"
        output_file = os.path.join(self.output_img_dir, filename)

        cv2.imwrite(output_file, image2d.data())
        print(f"2D image saved as: {output_file}")
        return True

    def render_depth_data(self, depth: np.ndarray) -> np.ndarray:
        """
        Render the depth map with a jet colormap.
        Produces an 8-bit color visualization (not a metric-depth png).
        """
        if depth is None or depth.size == 0:
            return np.array([])

        mask = np.isfinite(depth).astype(np.uint8)
        minv, maxv, _, _ = cv2.minMaxLoc(depth, mask)

        if np.isclose(maxv - minv, 0):
            depth8u = depth.astype(np.uint8)
        else:
            depth8u = cv2.convertScaleAbs(
                depth,
                alpha=(255.0 / (minv - maxv)),
                beta=((maxv * 255.0) / (maxv - minv) + 1),
            )

        if depth8u.size == 0:
            return np.array([])

        colored = cv2.applyColorMap(depth8u, cv2.COLORMAP_JET)
        colored[depth8u == 0] = [0, 0, 0]
        return colored

    def capture_depth_maps(self, idx: int, save_raw: bool = True, save_rendered: bool = True):
        """
        Capture the 3D frame, then save:
        - Raw depth map (16-bit PNG) -> depth_XX.png   (millimeters if float input)
        - Rendered depth map (PNG)   -> depth_rendered_XX.png
        """
        frame3d = Frame3D()

        print("\nCapturing 3D frame for depth...")
        status = self.camera.capture_3d(frame3d)
        if not status.is_ok():
            show_error(status)
            return False

        depth_map = frame3d.get_depth_map()
        depth_np = depth_map.data()

        print(f"[debug] Depth shape = {depth_np.shape}")

        # use it for debug only
        # print(f"[debug] depth dtype={depth_np.dtype}, shape={depth_np.shape}")

        # finite = depth_np[np.isfinite(depth_np)]
        # if finite.size:
        #     print(f"[debug] depth min={finite.min():.3f}, max={finite.max():.3f}")


        # --- Convert raw depth to 16-bit PNG safely ---
        # Robust handling of depth data that may be in float (meters or millimeters) or already in discrete units, with invalid values set to 0.
        # The SDK returns float32 depth.
        # For mechmind pro s camera, depth values are already in millimeters
        #
        # To keep this robust in case of future camera/config changes:
        #   - If values look like meters (e.g., <= 50), convert to mm.
        #   - Otherwise assume they are already mm.
        #
        # All invalid (NaN / Inf) values are set to 0 before conversion.

        if depth_np.dtype == np.float32 or depth_np.dtype == np.float64:
            depth_f = depth_np.copy()

            # Replace invalid values
            depth_f[~np.isfinite(depth_f)] = 0

            # Heuristic unit check
            finite = depth_f[depth_f > 0]
            if finite.size and finite.max() <= 50.0:
                # Values likely in meters -> convert to millimeters
                depth_mm = depth_f * 1000.0
            else:
                # Values already in millimeters
                depth_mm = depth_f

            depth_u16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
            print(f"[debug] nonzero={np.count_nonzero(depth_u16)}  max={depth_u16.max()}  mean_nonzero={depth_u16[depth_u16>0].mean():.1f}")

        else:
            # Integer-like depth (already discrete units)
            depth_i = depth_np.copy()

            if np.issubdtype(depth_i.dtype, np.floating):
                depth_i[~np.isfinite(depth_i)] = 0

            depth_u16 = np.clip(depth_i, 0, 65535).astype(np.uint16)
            print(f"[debug] nonzero={np.count_nonzero(depth_u16)}  max={depth_u16.max()}  mean_nonzero={depth_u16[depth_u16>0].mean():.1f}")

        ok = True
                
                
        # # --- Convert raw depth to 16-bit PNG safely ---
        # # simplified version based on current camera behavior (float32 in millimeters), but still robust to invalid values
        # # Depth from this camera is float32 in millimeters.
        # depth_mm = depth_np.copy()
        # depth_mm[~np.isfinite(depth_mm)] = 0
        # depth_u16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        # print(f"[debug] nonzero={np.count_nonzero(depth_u16)}  max={depth_u16.max()}  mean_nonzero={depth_u16[depth_u16>0].mean():.1f}")

        # ok = True


        # Save raw depth (16-bit PNG)
        if save_raw:
            raw_file = os.path.join(self.output_depth_dir, f"depth_{idx:02d}.png")
            cv2.imwrite(raw_file, depth_u16)
            print(f"Raw depth map (16-bit PNG) saved as: {raw_file}")

        # Save rendered/colored visualization (8-bit)
        if save_rendered:
            rendered = self.render_depth_data(depth_np)
            if rendered.size == 0:
                print("[warn] Rendered depth map is empty.")
                ok = False
            else:
                rendered_file = os.path.join(self.output_depth_dir, f"depth_rendered_{idx:02d}.png")
                cv2.imwrite(rendered_file, rendered)
                print(f"Rendered depth map saved as: {rendered_file}")

        return ok

    def save_camera_intrinsics(self):
        """
        Get and save camera intrinsic and extrinsic parameters once per session.
        Saves into the per-session "Camera intrinsics" folder.
        """
        if not self.output_intrinsics_dir:
            return False

        intrinsics_file = os.path.join(self.output_intrinsics_dir, "camera_intrinsics.json")
        if os.path.exists(intrinsics_file):
            print(f"[calib] camera intrinsics already exist: {intrinsics_file}")
            return True

        intr = CameraIntrinsics()
        status = self.camera.get_camera_intrinsics(intr)
        if not status.is_ok():
            show_error(status)
            return False

        data = {
            "texture": {
                "camera_matrix": {
                    "fx": intr.texture.camera_matrix.fx,
                    "fy": intr.texture.camera_matrix.fy,
                    "cx": intr.texture.camera_matrix.cx,
                    "cy": intr.texture.camera_matrix.cy,
                },
                "distortion": {
                    "k1": intr.texture.camera_distortion.k1,
                    "k2": intr.texture.camera_distortion.k2,
                    "p1": intr.texture.camera_distortion.p1,
                    "p2": intr.texture.camera_distortion.p2,
                    "k3": intr.texture.camera_distortion.k3,
                },
            },
            "depth": {
                "camera_matrix": {
                    "fx": intr.depth.camera_matrix.fx,
                    "fy": intr.depth.camera_matrix.fy,
                    "cx": intr.depth.camera_matrix.cx,
                    "cy": intr.depth.camera_matrix.cy,
                },
                "distortion": {
                    "k1": intr.depth.camera_distortion.k1,
                    "k2": intr.depth.camera_distortion.k2,
                    "p1": intr.depth.camera_distortion.p1,
                    "p2": intr.depth.camera_distortion.p2,
                    "k3": intr.depth.camera_distortion.k3,
                },
            },
            "depth_to_texture": {
                "rotation": [
                    [intr.depth_to_texture.rotation[i][j] for j in range(3)]
                    for i in range(3)
                ],
                # translation is in millimeters
                "translation_mm": [
                    intr.depth_to_texture.translation[0],
                    intr.depth_to_texture.translation[1],
                    intr.depth_to_texture.translation[2],
                ],
            },
        }

        os.makedirs(self.output_intrinsics_dir, exist_ok=True)
        with open(intrinsics_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"[calib] camera intrinsics saved: {intrinsics_file}")
        return True
    
    def run_capture_from_config(self, config: dict):
        """
        Non-interactive capture entry point for Python Agent / Grasshopper.
        Keeps main() unchanged for manual CLI workflow.
        """

        session_name = config.get("session") or "Test 01"
        camera_index = int(config.get("camera_index", 0))
        capture_index = int(config.get("capture_index", 1))
        mode = config.get("mode", "all")

        try:
            paths = load_session(".", session_name)
        except FileNotFoundError:
            paths = init_session(".", session_name)
            print(f"[init] created session: {paths.session_name}")

        self.output_pc_dir = str(paths.initial_point_clouds)
        self.output_img_dir = str(paths.initial_images)
        self.output_depth_dir = str(paths.initial_depth_images)
        self.output_intrinsics_dir = str(paths.camera_intrinsics)

        for d in (
            self.output_pc_dir,
            self.output_img_dir,
            self.output_depth_dir,
            self.output_intrinsics_dir,
        ):
            os.makedirs(d, exist_ok=True)

        camera_infos = self.discover_cameras()
        if not camera_infos:
            return {
                "status": "error",
                "message": "No cameras found."
            }

        if camera_index < 0 or camera_index >= len(camera_infos):
            return {
                "status": "error",
                "message": f"Invalid camera_index {camera_index}."
            }

        print(f"\nConnecting to camera index {camera_index}...")
        status = self.camera.connect(camera_infos[camera_index])
        if not status.is_ok():
            show_error(status)
            return {
                "status": "error",
                "message": "Failed to connect to camera."
            }

        try:
            print("Connected to the camera successfully.")

            self.save_camera_intrinsics()

            ok_pc = None
            ok_img = None
            ok_depth = None

            if mode == "pointcloud":
                ok_pc = self.capture_textured_point_cloud(capture_index)

            elif mode == "image":
                ok_img = self.capture_2d_image(capture_index)

            elif mode == "depth":
                ok_depth = self.capture_depth_maps(
                    capture_index,
                    save_raw=True,
                    save_rendered=True
                )

            elif mode == "all":
                ok_pc = self.capture_textured_point_cloud(capture_index)
                ok_img = self.capture_2d_image(capture_index)
                ok_depth = self.capture_depth_maps(
                    capture_index,
                    save_raw=True,
                    save_rendered=True
                )

            else:
                return {
                    "status": "error",
                    "message": f"Unknown capture mode: {mode}"
                }

            return {
                "status": "ok",
                "session": paths.session_name,
                "capture_index": capture_index,
                "mode": mode,
                "files": {
                    "pointcloud": os.path.join(
                        self.output_pc_dir,
                        f"point_cloud_{capture_index:02d}.ply"
                    ),
                    "image": os.path.join(
                        self.output_img_dir,
                        f"image_{capture_index:02d}.png"
                    ),
                    "depth": os.path.join(
                        self.output_depth_dir,
                        f"depth_{capture_index:02d}.png"
                    ),
                    "depth_rendered": os.path.join(
                        self.output_depth_dir,
                        f"depth_rendered_{capture_index:02d}.png"
                    ),
                    "intrinsics": os.path.join(
                        self.output_intrinsics_dir,
                        "camera_intrinsics.json"
                    ),
                },
                "results": {
                    "pointcloud": ok_pc,
                    "image": ok_img,
                    "depth": ok_depth,
                }
            }

        finally:
            self.camera.disconnect()
            print("Disconnected from the camera successfully.")


    def main(self):
        # >>> Added: pick or create a session, then set output dirs to session folders
        session_name = input("Session name (leave blank to reuse last or create 'Test 01'): ").strip()
        try:
            paths = load_session(".", session_name or None)
        except FileNotFoundError:
            # If no session exists, create one (named by user or default)
            paths = init_session(".", session_name or "Test 01")
            print(f"[init] created session: {paths.session_name}")

        # Match your folder intent:
        #   - point clouds -> "Initial point clouds"
        #   - 2D images    -> "Initial images"
        #   - depth maps   -> "Initial depth images"
        self.output_pc_dir = str(paths.initial_point_clouds)
        self.output_img_dir = str(paths.initial_images)
        self.output_depth_dir = str(paths.initial_depth_images)
        self.output_intrinsics_dir = str(paths.camera_intrinsics)

        for d in (self.output_pc_dir, self.output_img_dir, self.output_depth_dir, self.output_intrinsics_dir):
            os.makedirs(d, exist_ok=True)

        print(f"[capture] Point clouds dir: {self.output_pc_dir}")
        print(f"[capture] Images dir:      {self.output_img_dir}")
        print(f"[capture] Depth dir:       {self.output_depth_dir}")
        print(f"[capture] Intrinsics dir:  {self.output_intrinsics_dir}")

        # Discover available cameras
        camera_infos = self.discover_cameras()
        if not camera_infos:
            return

        # Choose camera
        cam_index = self.choose_camera_index(camera_infos)

        # Connect to selected camera
        print(f"\nConnecting to camera index {cam_index}...")
        status = self.camera.connect(camera_infos[cam_index])
        if not status.is_ok():
            show_error(status)
            return

        print("Connected to the camera successfully.")

        # >>> Added: save camera intrinsics once per session
        self.save_camera_intrinsics()

        # Ask one index so files line up (point_cloud_XX / image_XX / depth_XX)
        idx = self.ask_index("Enter capture index", default=1)

        # Simple capture menu (keeps your one-run flow)
        print("\nWhat do you want to capture?")
        print("1) Textured point cloud (PLY)")
        print("2) 2D image (PNG)")
        print("3) Depth map (raw + rendered)")
        print("4) All (point cloud + 2D + depth)")
        choice = input("Choose [1-4] (default 4): ").strip() or "4"

        if choice == "1":
            self.capture_textured_point_cloud(idx)
        elif choice == "2":
            self.capture_2d_image(idx)
        elif choice == "3":
            self.capture_depth_maps(idx, save_raw=True, save_rendered=True)
        else:
            self.capture_textured_point_cloud(idx)
            self.capture_2d_image(idx)
            self.capture_depth_maps(idx, save_raw=True, save_rendered=True)

        # Disconnect
        self.camera.disconnect()
        print("Disconnected from the camera successfully.")


if __name__ == "__main__":
    app = CaptureTexturedPointCloud()
    app.main()