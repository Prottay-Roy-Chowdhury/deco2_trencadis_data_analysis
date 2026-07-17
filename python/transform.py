import time
import open3d as o3d
import numpy as np
import os, re, copy, json
import cv2
from glob import glob

# import session helper and resolve folders from the active session
from helpers.session_manager import load_session

# ─────────────── CONFIGURATION ───────────────

VOXEL_SIZE_MM = 0.5
NORMAL_RADIUS_MM = 1.0
ICP_THRESHOLDS = [10.0, 5.0, 1.0]  # mm

SHOW_PREVIEW = True
PREVIEW_TIME_SEC = 2.0

MERGED_OUTPUT_INDEX = 1

MERGED_PCD_TEMPLATE = "merged{idx:02d}.ply"
STITCHED_RGB_TEMPLATE = "stitched_rgb_{idx:02d}.png"
STITCHED_HEIGHT_TEMPLATE = "stitched_height_{idx:02d}.png"
EYE_TO_BASE_RGB_TEMPLATE = "eye_to_base_rgb_{idx:02d}.png"
EYE_TO_BASE_HEIGHT_TEMPLATE = "eye_to_base_height_{idx:02d}.png"
EYE_TO_BASE_PCD_TEMPLATE = "eye_to_base_point_cloud_{idx:02d}.ply"

# ─────────────────────────────────────────────


class PointCloudMerger:
    def __init__(
            self,
            pcd_dir,
            tf_dir,
            reference_index,

            eye_to_base_tf_dir=None,
            eye_to_base_index=None,

            image_dir=None,
            depth_dir=None,
            intrinsics_dir=None,

            merged_pcd_dir=None,
            merged_image_dir=None,
            merged_depth_dir=None,

            voxel_size_mm=0.5,
            normal_radius_mm=1.0,
            icp_thresholds_mm=(10.0, 5.0, 1.0),
            frame_mode="ref_cam",
            output_index=MERGED_OUTPUT_INDEX):

        self.pcd_dir = pcd_dir
        self.tf_dir = tf_dir
        self.reference_index = reference_index

        self.eye_to_base_tf_dir = eye_to_base_tf_dir or tf_dir
        self.eye_to_base_index = eye_to_base_index

        self.voxel = voxel_size_mm
        self.normal_radius = normal_radius_mm
        self.icp_thresholds = icp_thresholds_mm

        assert frame_mode in ("ref_cam", "base", "eye_to_base")
        self.frame_mode = frame_mode

        self.pcd_dict = {}
        self.tf_dict = {}
        self.T_ref = np.eye(4)

        # >>> Added (images): paths + final per-scan effective transforms (camera -> merged BASE)
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.intrinsics_dir = intrinsics_dir

        self.merged_pcd_dir = merged_pcd_dir
        self.merged_image_dir = merged_image_dir
        self.merged_depth_dir = merged_depth_dir

        self.T_eff = {}  # index -> 4x4 camera->final-base transform (includes ICP refinement)

        # >>> Added (images): stitching settings
        self.STITCH_RES_MM = 1.0  # 1 mm/px
        self.WINNER_RULE = "max_z"  # max-Z in BASE wins per XY cell (tabletop top-surface)

        self.output_index = output_index

    # ---------- utils ----------
    @staticmethod
    def load_transform_matrix(txt_path: str) -> np.ndarray:
        with open(txt_path, "r") as f:
            rows = [line.strip("[]\n ").split(",") for line in f.readlines()]
        return np.array([[float(x) for x in row] for row in rows])

    @staticmethod
    def get_index_from_pcd(file):
        match = re.search(r"point_cloud_(\d+)", os.path.basename(file))
        return int(match.group(1)) if match else None

    @staticmethod
    def get_index_from_tf(file):
        match = re.search(r"T_base_cam_pose(\d+)", os.path.basename(file))
        return int(match.group(1)) if match else None

    @staticmethod
    def show_pair(src, tgt, T=np.eye(4), title="", seconds=None):
        if not SHOW_PREVIEW:
            return

        if seconds is None:
            seconds = PREVIEW_TIME_SEC

        s, t = copy.deepcopy(src), copy.deepcopy(tgt)
        s.paint_uniform_color([1.0, 0.6, 0.0]); t.paint_uniform_color([0.0, 0.65, 0.93])
        s.transform(T)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(s); vis.add_geometry(t)
        vis.get_view_control().set_zoom(0.8)

        end_time = time.time() + seconds
        while time.time() < end_time:
            vis.poll_events()
            vis.update_renderer()

        vis.destroy_window()
    
    @staticmethod
    def show_timed_pointcloud(geometries, window_name="Preview", seconds=None):
        if not SHOW_PREVIEW:
            return

        if seconds is None:
            seconds = PREVIEW_TIME_SEC

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1280, height=720)

        for g in geometries:
            vis.add_geometry(g)

        end_time = time.time() + seconds
        while time.time() < end_time:
            vis.poll_events()
            vis.update_renderer()

        vis.destroy_window()

    @staticmethod
    def run_icp(src, tgt, threshold_mm, init_T, estimation):
        result = o3d.pipelines.registration.registration_icp(
            src, tgt, threshold_mm, init_T, estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        return result.transformation

    def process_point_cloud(self, pcd_path, tf_path, pretransform_to_base=False):
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = pcd.voxel_down_sample(self.voxel)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        T_base_cam = self.load_transform_matrix(tf_path)
        if pretransform_to_base:
            # T_base_cam maps points from camera to base: p_base = T_base_cam * p_cam
            pcd.transform(T_base_cam)
        return pcd, T_base_cam

    # >>> Added (images): intrinsics + IO + stitching helpers ----------
    def load_camera_intrinsics(self):
        """
        Load intrinsics once per session.
        For your camera, distortion is all zeros and depth_to_texture is identity (registered output).
        We only need fx, fy, cx, cy.
        """
        intr_file = os.path.join(self.intrinsics_dir, "camera_intrinsics.json")
        if not os.path.exists(intr_file):
            raise FileNotFoundError(f"camera intrinsics not found: {intr_file}")

        data = json.loads(open(intr_file, "r", encoding="utf-8").read())
        fx = float(data["texture"]["camera_matrix"]["fx"])
        fy = float(data["texture"]["camera_matrix"]["fy"])
        cx = float(data["texture"]["camera_matrix"]["cx"])
        cy = float(data["texture"]["camera_matrix"]["cy"])
        return fx, fy, cx, cy

    def load_rgb_depth(self, idx: int):
        """
        Load RGB and raw depth for a given capture index.
        Expects:
          - image_XX.png in Initial images
          - depth_XX.png in Initial depth images (uint16 mm)
        """
        rgb_path = os.path.join(self.image_dir, f"image_{idx:02d}.png")
        depth_path = os.path.join(self.depth_dir, f"depth_{idx:02d}.png")

        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth image not found: {depth_path}")

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # BGR
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16 (mm)

        if rgb is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
        if depth is None:
            raise RuntimeError(f"Failed to read depth image: {depth_path}")
        if depth.dtype != np.uint16:
            raise ValueError(f"Depth must be uint16 (mm). Got {depth.dtype} from {depth_path}")

        # Sanity check: in your setup these should match (H,W) == (1200,1920)
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            raise ValueError(f"RGB/depth resolution mismatch for idx {idx}: rgb={rgb.shape}, depth={depth.shape}")

        return rgb, depth

    def stitch_rgbd_after_registration(self, merged_base: o3d.geometry.PointCloud, out_rgb: str, out_height: str):
        """
        Stitch RGB + HEIGHT (Z in BASE) into a top-down ortho mosaic in BASE XY.
        - 1 mm/px
        - winner rule: max-Z in BASE wins per XY cell
        Uses the same per-scan transforms you already computed during ICP registration:
          T_eff[i] = T_icp[i] @ T_base_cam[i]
        """
        fx, fy, cx, cy = self.load_camera_intrinsics()

        # Use merged cloud bounds to define canvas (ensures match with merged_base frame)
        pts = np.asarray(merged_base.points)
        if pts.size == 0:
            print("[warn] merged cloud is empty; skipping image stitching.")
            return False

        # --- Height storage note ---
        # BASE Z can be negative. Since PNG uint16 can't store negative values,
        # we store "height above min Z" so the saved height map is non-negative.
        z_shift = float(np.min(pts[:, 2]))  # mm
        print(f"[debug] height map shift: z_shift(min Z) = {z_shift:.3f} mm")

        min_x, min_y = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
        max_x, max_y = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))

        res = float(self.STITCH_RES_MM)

        width = int(np.ceil((max_x - min_x) / res)) + 1
        height = int(np.ceil((max_y - min_y) / res)) + 1

        # Canvas outputs
        mosaic_rgb = np.zeros((height, width, 3), dtype=np.uint8)     # BGR
        mosaic_h   = np.zeros((height, width), dtype=np.uint16)       # Z_base in mm
        zbuf       = np.full((height, width), -np.inf, dtype=np.float32)

        # Flatten views for faster indexed updates
        zbuf_flat = zbuf.reshape(-1)
        h_flat = mosaic_h.reshape(-1)
        rgb_flat = mosaic_rgb.reshape(-1, 3)

        # Process scans in the same order as point clouds
        for i in sorted(self.pcd_dict.keys()):
            if i not in self.T_eff:
                print(f"[warn] Missing effective transform for scan {i}, skipping RGB-D.")
                continue

            try:
                rgb, depth_u16 = self.load_rgb_depth(i)
            except Exception as e:
                print(f"[warn] Failed to load RGB-D for scan {i}: {e}")
                continue

            T = self.T_eff[i]
            R = T[:3, :3].astype(np.float32)
            t = T[:3, 3].astype(np.float32)

            # Valid depth pixels
            valid = depth_u16 > 0
            if not np.any(valid):
                print(f"[warn] No valid depth pixels for scan {i}.")
                continue

            ys, xs = np.nonzero(valid)
            Z = depth_u16[ys, xs].astype(np.float32)  # mm

            # Backproject to camera 3D (mm) using intrinsics
            X = (xs.astype(np.float32) - cx) * (Z / fx)
            Y = (ys.astype(np.float32) - cy) * (Z / fy)

            P_cam = np.stack([X, Y, Z], axis=0)  # 3xN (mm)

            # Transform to BASE (mm)
            P_base = (R @ P_cam) + t.reshape(3, 1)  # 3xN
            Xb = P_base[0, :]
            Yb = P_base[1, :]
            Zb = P_base[2, :]  # height in BASE (mm)

            # Rasterize to base XY grid
            ix = np.floor((Xb - min_x) / res).astype(np.int32)
            iy = np.floor((max_y - Yb) / res).astype(np.int32)  # flip Y so +Y goes up in image

            inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            if not np.any(inside):
                continue

            ix = ix[inside]
            iy = iy[inside]
            zb = Zb[inside]
            xs_in = xs[inside]
            ys_in = ys[inside]

            # Winner rule: max Z in BASE per cell (top surface)
            lin = (iy.astype(np.int64) * width + ix.astype(np.int64))

            # To avoid huge Python loops: for this scan, pick the max-Z sample per cell
            order = np.argsort(zb)  # ascending
            lin_s = lin[order]
            zb_s = zb[order]
            xs_s = xs_in[order]
            ys_s = ys_in[order]

            # Keep last occurrence (highest Z) per lin_s by reversing and taking first unique
            lin_rev = lin_s[::-1]
            zb_rev = zb_s[::-1]
            xs_rev = xs_s[::-1]
            ys_rev = ys_s[::-1]

            uniq_lin, uniq_pos = np.unique(lin_rev, return_index=True)
            win_lin = uniq_lin
            win_zb = zb_rev[uniq_pos]
            win_xs = xs_rev[uniq_pos]
            win_ys = ys_rev[uniq_pos]

            # Compare to global z-buffer: only update where this scan is higher
            cur = zbuf_flat[win_lin]
            upd = win_zb > cur
            if not np.any(upd):
                continue

            win_lin = win_lin[upd]
            win_zb = win_zb[upd]
            win_xs = win_xs[upd]
            win_ys = win_ys[upd]

            # Update z-buffer, height map and RGB mosaic
            zbuf_flat[win_lin] = win_zb.astype(np.float32)
            # Store height as (Z_base - z_shift) in mm so it is non-negative in uint16 PNG
            win_h = win_zb - z_shift
            h_flat[win_lin] = np.clip(win_h, 0, 65535).astype(np.uint16)
            rgb_flat[win_lin, :] = rgb[win_ys, win_xs, :]

            print(f"[stitch] scan {i:02d}: updated {win_lin.size} px")

        # Ensure output dirs exist
        os.makedirs(os.path.dirname(out_rgb), exist_ok=True)
        os.makedirs(os.path.dirname(out_height), exist_ok=True)

        # --- Debug: report BASE height ranges (point cloud vs stitched height map) ---
        z_pc = pts[:, 2]
        if z_pc.size:
            print(f"[debug] merged cloud Z_base: min={z_pc.min():.3f}  max={z_pc.max():.3f}  (mm)")
        else:
            print("[debug] merged cloud Z_base: empty")

        finite_z = zbuf[np.isfinite(zbuf)]
        if finite_z.size:
            print(f"[debug] stitched z-buffer Z_base: min={finite_z.min():.3f}  max={finite_z.max():.3f}  (mm)")
        else:
            print("[debug] stitched z-buffer Z_base: empty")

        nonzero_h = mosaic_h[mosaic_h > 0]
        if nonzero_h.size:
            print(f"[debug] stitched height uint16 (Z_base - z_shift): min={int(nonzero_h.min())}  max={int(nonzero_h.max())}  (mm)")
        else:
            print("[debug] stitched height uint16: all zeros")

        cv2.imwrite(out_rgb, mosaic_rgb)
        cv2.imwrite(out_height, mosaic_h)

        return True
    
    # ---------- output paths & index numbers ----------

    def merged_pcd_path(self, idx=None):
        if idx is None:
            idx = self.output_index
        return os.path.join(
            self.merged_pcd_dir,
            MERGED_PCD_TEMPLATE.format(idx=idx)
        )

    def stitched_rgb_path(self, idx=None):
        if idx is None:
            idx = self.output_index
        return os.path.join(
            self.merged_image_dir,
            STITCHED_RGB_TEMPLATE.format(idx=idx)
        )

    def stitched_height_path(self, idx=None):
        if idx is None:
            idx = self.output_index
        return os.path.join(
            self.merged_depth_dir,
            STITCHED_HEIGHT_TEMPLATE.format(idx=idx)
        )

    def eye_to_base_pcd_path(self, idx=None):
        if idx is None:
            idx = self.output_index
        return os.path.join(
            self.merged_pcd_dir,
            EYE_TO_BASE_PCD_TEMPLATE.format(idx=idx)
        )
    
    def eye_to_base_rgb_path(self, idx=None):
        if idx is None:
            idx = self.output_index
        return os.path.join(
            self.merged_image_dir,
            EYE_TO_BASE_RGB_TEMPLATE.format(idx=idx)
        )
    
    def eye_to_base_height_path(self, idx=None):
        if idx is None:
            idx = self.output_index
        return os.path.join(
            self.merged_depth_dir,
            EYE_TO_BASE_HEIGHT_TEMPLATE.format(idx=idx)
        )

    # ---------- pipelines ----------
    def merge_ref_cam_pipeline(self):
        """Original behavior: align in reference CAMERA frame; return merged (camera frame) + T_ref."""
        all_pcd_files = glob(os.path.join(self.pcd_dir, "point_cloud_*.ply"))
        all_tf_files  = glob(os.path.join(self.tf_dir, "T_base_cam_pose*.txt"))
        self.pcd_dict = {self.get_index_from_pcd(f): f for f in all_pcd_files if self.get_index_from_pcd(f) is not None}
        self.tf_dict  = {self.get_index_from_tf(f): f for f in all_tf_files  if self.get_index_from_tf(f)  is not None}

        print("Found point clouds:", sorted(self.pcd_dict.keys()))
        print("Found transforms:  ", sorted(self.tf_dict.keys()))
        if self.reference_index not in self.pcd_dict or self.reference_index not in self.tf_dict:
            raise FileNotFoundError(f"Reference index {self.reference_index} not found in files.")

        ref_pcd, T_ref = self.process_point_cloud(self.pcd_dict[self.reference_index],
                                                  self.tf_dict[self.reference_index],
                                                  pretransform_to_base=False)
        self.T_ref = T_ref
        merged = copy.deepcopy(ref_pcd)

        for i in sorted(self.pcd_dict.keys()):
            if i == self.reference_index:
                continue
            if i not in self.tf_dict:
                print(f"[!] Missing transform for cloud {i}, skipping."); continue

            print(f"\n[•] Aligning point_cloud_{i:02d} to reference point_cloud_{self.reference_index:02d}...")
            src_pcd, T_i = self.process_point_cloud(self.pcd_dict[i], self.tf_dict[i], pretransform_to_base=False)

            # Same math as before: bring src into reference CAMERA frame
            T_init = np.linalg.inv(self.T_ref) @ T_i
            T_A = self.run_icp(src_pcd, merged, self.icp_thresholds[0], T_init,
                               o3d.pipelines.registration.TransformationEstimationPointToPoint())
            T_B = self.run_icp(src_pcd, merged, self.icp_thresholds[1], T_A,
                               o3d.pipelines.registration.TransformationEstimationPointToPlane())
            T_C = self.run_icp(src_pcd, merged, self.icp_thresholds[2], T_B,
                               o3d.pipelines.registration.TransformationEstimationPointToPlane())

            src_pcd.transform(T_C)
            self.show_pair(src_pcd, merged, np.eye(4), f"Merged with point_cloud_{i:02d}")
            merged += src_pcd
            merged = merged.voxel_down_sample(self.voxel)

        return merged, self.T_ref  # merged in reference CAMERA frame

    def merge_base_pipeline(self):
        """Proposed behavior: pre-transform each cloud into BASE using T_base_cam, then ICP in BASE."""
        all_pcd_files = glob(os.path.join(self.pcd_dir, "point_cloud_*.ply"))
        all_tf_files  = glob(os.path.join(self.tf_dir, "T_base_cam_pose*.txt"))
        self.pcd_dict = {self.get_index_from_pcd(f): f for f in all_pcd_files if self.get_index_from_pcd(f) is not None}
        self.tf_dict  = {self.get_index_from_tf(f): f for f in all_tf_files  if self.get_index_from_tf(f)  is not None}

        print("Found point clouds:", sorted(self.pcd_dict.keys()))
        print("Found transforms:  ", sorted(self.tf_dict.keys()))
        if self.reference_index not in self.pcd_dict or self.reference_index not in self.tf_dict:
            raise FileNotFoundError(f"Reference index {self.reference_index} not found in files.")

        # Reference cloud, already in BASE
        ref_pcd_base, T_ref = self.process_point_cloud(self.pcd_dict[self.reference_index],
                                                       self.tf_dict[self.reference_index],
                                                       pretransform_to_base=True)
        self.T_ref = T_ref  # not used for inverses in this pipeline, retained for logging if needed
        merged = copy.deepcopy(ref_pcd_base)

        # >>> Added (images): for reference scan, ICP correction is identity => T_eff = T_base_cam
        self.T_eff[self.reference_index] = T_ref

        for i in sorted(self.pcd_dict.keys()):
            if i == self.reference_index:
                continue
            if i not in self.tf_dict:
                print(f"[!] Missing transform for cloud {i}, skipping."); continue

            print(f"\n[•] Aligning (BASE) point_cloud_{i:02d} to merged...")
            # Source cloud, already in BASE
            src_pcd_base, T_base_cam = self.process_point_cloud(self.pcd_dict[i], self.tf_dict[i],
                                                                pretransform_to_base=True)

            # Since both are in BASE, a good init is the identity
            T_A = self.run_icp(src_pcd_base, merged, self.icp_thresholds[0], np.eye(4),
                               o3d.pipelines.registration.TransformationEstimationPointToPoint())
            T_B = self.run_icp(src_pcd_base, merged, self.icp_thresholds[1], T_A,
                               o3d.pipelines.registration.TransformationEstimationPointToPlane())
            T_C = self.run_icp(src_pcd_base, merged, self.icp_thresholds[2], T_B,
                               o3d.pipelines.registration.TransformationEstimationPointToPlane())

            # >>> Added (images): store effective camera->final-base transform for this scan
            self.T_eff[i] = T_C @ T_base_cam

            src_pcd_base.transform(T_C)
            self.show_pair(src_pcd_base, merged, np.eye(4), f"Merged (BASE) with point_cloud_{i:02d}")
            merged += src_pcd_base
            merged = merged.voxel_down_sample(self.voxel)

        # Result is already in BASE; no final transform needed
        return merged

    def eye_to_base_pipeline(self):
        """
        Eye-on-base / eye-to-base pipeline:
        - one capture only
        - separate eye_to_base_index
        - separate transform folder
        - no ICP
        - no merging
        """

        idx = self.eye_to_base_index

        all_pcd_files = glob(os.path.join(self.pcd_dir, "point_cloud_*.ply"))
        all_tf_files = glob(os.path.join(self.eye_to_base_tf_dir, "T_base_cam_pose*.txt"))

        self.pcd_dict = {
            self.get_index_from_pcd(f): f
            for f in all_pcd_files
            if self.get_index_from_pcd(f) is not None
        }

        self.tf_dict = {
            self.get_index_from_tf(f): f
            for f in all_tf_files
            if self.get_index_from_tf(f) is not None
        }

        print("Found point clouds:", sorted(self.pcd_dict.keys()))
        print("Found eye-to-base transforms:", sorted(self.tf_dict.keys()))
        print(f"Using eye-to-base index: {idx}")

        if idx not in self.pcd_dict:
            raise FileNotFoundError(f"Point cloud index {idx} not found.")

        if idx not in self.tf_dict:
            raise FileNotFoundError(
                f"Eye-to-base transform index {idx} not found in {self.eye_to_base_tf_dir}"
            )

        pcd_base, T_base_cam = self.process_point_cloud(
            self.pcd_dict[idx],
            self.tf_dict[idx],
            pretransform_to_base=True
        )

        # For eye-to-base, no ICP correction, so effective camera->base is just T_base_cam
        self.T_eff[idx] = T_base_cam

        return pcd_base, T_base_cam

    def run(self, output_path):
        if self.frame_mode == "ref_cam":
            merged_cam, T_ref = self.merge_ref_cam_pipeline()

            # If you still want the final in BASE, uncomment the line below:
            # merged_cam.transform(np.linalg.inv(T_ref))

            o3d.io.write_point_cloud(output_path, merged_cam)
            print(f"\nSaved merged cloud (reference CAMERA frame) → {output_path}")

            self.show_timed_pointcloud(
                [merged_cam],
                window_name="Merged (Ref Camera)"
            )

        elif self.frame_mode == "base":
            merged_base = self.merge_base_pipeline()

            o3d.io.write_point_cloud(output_path, merged_base)
            print(f"\nSaved merged cloud (ROBOT BASE frame) → {output_path}")

            self.show_timed_pointcloud(
                [merged_base],
                window_name="Merged (Robot Base)"
            )

            # >>> Added (images): stitch RGB + HEIGHT map after ICP registration in BASE
            stitched_rgb_path = self.stitched_rgb_path()

            stitched_height_path = self.stitched_height_path()

            ok = self.stitch_rgbd_after_registration(
                merged_base,
                stitched_rgb_path,
                stitched_height_path
            )

            if ok:
                print(f"\nSaved stitched RGB (BASE, 1mm/px) → {stitched_rgb_path}")
                print(f"Saved stitched height Z_base (mm, uint16) → {stitched_height_path}")
            else:
                print("[warn] stitching failed or skipped.")

        elif self.frame_mode == "eye_to_base":
            idx = self.eye_to_base_index

            eye_to_base_output_path = self.eye_to_base_pcd_path()

            pcd_base, T_base_cam = self.eye_to_base_pipeline()

            o3d.io.write_point_cloud(
                eye_to_base_output_path,
                pcd_base
            )

            print(
                f"\nSaved eye-to-base transformed cloud "
                f"(ROBOT BASE frame) → {eye_to_base_output_path}"
            )
            

            self.show_timed_pointcloud(
                [pcd_base],
                window_name="Eye-to-Base Point Cloud"
            )

            # Generate transformed RGB + height/depth map in ROBOT BASE frame
            eye_to_base_rgb_path = self.eye_to_base_rgb_path()
            eye_to_base_height_path = self.eye_to_base_height_path()

            ok = self.stitch_rgbd_after_registration(
                pcd_base,
                eye_to_base_rgb_path,
                eye_to_base_height_path
            )

            if ok:
                print(f"\nSaved eye-to-base RGB/height map (BASE, 1mm/px) → {eye_to_base_rgb_path}")
                print(f"Saved eye-to-base height Z_base (mm, uint16) → {eye_to_base_height_path}")
            else:
                print("[warn] eye-to-base RGB-D transform failed or skipped.")

        else:
            raise ValueError(f"Unknown frame_mode: {self.frame_mode}")


def run_transform_from_config(config: dict):
    """
    Non-interactive transform entry point for Python Agent / Grasshopper.
    Keeps main() available for manual interactive workflow.
    """

    session_name = config.get("session") or None
    pipeline = config.get("pipeline", "eye_to_base")

    ref_index = int(config.get("reference_index", 2))
    eye_base_index = int(config.get("eye_to_base_index", 1))

    paths = load_session(".", session_name)

    pcd_dir = str(paths.initial_point_clouds)
    tf_dir = str(paths.global_transforms_root)
    eye_tf_dir = str(paths.eye_to_base_transforms_root)

    output_index = int(config.get("output_index", MERGED_OUTPUT_INDEX))

    output_path_local = os.path.join(
        str(paths.merged_point_clouds),
        MERGED_PCD_TEMPLATE.format(idx=output_index)
    )

    merger = PointCloudMerger(
        pcd_dir=pcd_dir,
        tf_dir=tf_dir,
        reference_index=ref_index,

        eye_to_base_tf_dir=eye_tf_dir,
        eye_to_base_index=eye_base_index,

        image_dir=str(paths.initial_images),
        depth_dir=str(paths.initial_depth_images),
        intrinsics_dir=str(paths.camera_intrinsics),

        merged_pcd_dir=str(paths.merged_point_clouds),
        merged_image_dir=str(paths.merged_images),
        merged_depth_dir=str(paths.merged_depth_images),

        voxel_size_mm=VOXEL_SIZE_MM,
        normal_radius_mm=NORMAL_RADIUS_MM,
        icp_thresholds_mm=ICP_THRESHOLDS,
        frame_mode=pipeline,
        output_index=output_index
    )

    merger.run(output_path_local)

    return {
        "status": "ok",
        "session": paths.session_name,
        "pipeline": pipeline,
        "reference_index": ref_index,
        "eye_to_base_index": eye_base_index,
        "output_path": output_path_local,
    }


def main():
    # Resolve directories from the chosen (or last) session
    _session_name = input("Session name (blank = reuse last): ").strip() or None
    _paths = load_session(".", _session_name)

    pointcloud_dir = str(_paths.initial_point_clouds)           # sessions/<Session>/Initial point clouds
    transform_dir  = str(_paths.global_transforms_root)         # helpers/Transformation matrix_3 capture (shared)
    eye_to_base_transform_dir = str(_paths.eye_to_base_transforms_root) # Separate transform folder for eye-on-base / eye-to-base setup
    
    output_index=MERGED_OUTPUT_INDEX
    output_path = os.path.join(
    str(_paths.merged_point_clouds),
    MERGED_PCD_TEMPLATE.format(idx=output_index)
    )  # sessions/<Session>/Merged point clouds

    # >>> Added (images): source folders + outputs
    image_dir      = str(_paths.initial_images)                # sessions/<Session>/Initial images
    depth_dir      = str(_paths.initial_depth_images)          # sessions/<Session>/Initial depth images
    intrinsics_dir = str(_paths.camera_intrinsics)             # sessions/<Session>/Camera intrinsics

    reference_index = 2  # <-- anchor scan index

    eye_to_base_index = 1 # Separate index only for eye-on-base / eye-to-base pipeline

    # Choose pipeline: "ref_cam" (original) or "base" (pre-transform to robot base then ICP) or "eye_to_base"
    _pipeline_choice = input(
        "Choose pipeline [1=ref_cam, 2=base, 3=eye_to_base] (blank = base): "
    ).strip().lower()

    PIPELINE_FRAME = {
        "": "base",
        "1": "ref_cam",
        "ref_cam": "ref_cam",
        "2": "base",
        "base": "base",
        "3": "eye_to_base",
        "eye_to_base": "eye_to_base",
    }.get(_pipeline_choice)

    if PIPELINE_FRAME is None:
        raise ValueError("Invalid pipeline choice.")

    merger = PointCloudMerger(
        pcd_dir=pointcloud_dir,
        tf_dir=transform_dir,
        reference_index=reference_index,

        eye_to_base_tf_dir=eye_to_base_transform_dir,
        eye_to_base_index=eye_to_base_index,

        image_dir=image_dir,
        depth_dir=depth_dir,
        intrinsics_dir=intrinsics_dir,

        merged_pcd_dir=str(_paths.merged_point_clouds),
        merged_image_dir=str(_paths.merged_images),
        merged_depth_dir=str(_paths.merged_depth_images),

        voxel_size_mm=VOXEL_SIZE_MM,
        normal_radius_mm=NORMAL_RADIUS_MM,
        icp_thresholds_mm=ICP_THRESHOLDS,
        frame_mode=PIPELINE_FRAME,
        output_index=output_index
    )

    merger.run(output_path)


if __name__ == "__main__":
    main()