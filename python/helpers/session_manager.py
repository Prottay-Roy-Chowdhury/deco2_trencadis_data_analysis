# helpers/session_manager.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import platform

# Per-session subfolder names (exactly as specified by the user)
DIR_EXPORTED = "Exported data"
DIR_INITIAL = "Initial point clouds"
DIR_INITIAL_IMAGES = "Initial images"
DIR_INITIAL_DEPTH_IMAGES = "Initial depth images"
DIR_MERGED = "Merged point clouds"
DIR_MERGED_IMAGES = "Merged images"
DIR_MERGED_DEPTH_IMAGES = "Merged depth images"
DIR_INTRINSICS = "Camera intrinsics"

# Manually-created, shared transforms folder (do NOT create it in code)
GLOBAL_TRANSFORMS_DIR = Path(__file__).resolve().parent / "Transformation matrix_3 capture"
EYE_TO_BASE_TRANSFORMS_DIR = Path(__file__).resolve().parent / "Transformation matrix_eye_to_base"

# Small convenience file at project root to remember the last session user used
RECENT_FILE = ".last_session"


@dataclass
class SessionPaths:
    # Per-session paths
    root: Path
    exported_data: Path
    initial_point_clouds: Path
    initial_images: Path
    initial_depth_images: Path
    merged_point_clouds: Path
    merged_images: Path
    merged_depth_images: Path
    camera_intrinsics: Path
    manifest: Path
    session_name: str
    # Global transforms (common to all sessions)
    global_transforms_root: Path
    eye_to_base_transforms_root: Path


def _sessions_dir(project_root: Path) -> Path:
    return (project_root / "sessions").resolve()


def _safe_latest_pointer(parent: Path, session_dir: Path):
    """Create a 'latest' pointer (symlink if possible, else a tiny text file)."""
    latest = parent / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(session_dir.name, target_is_directory=True)  # relative symlink
    except Exception:
        latest.write_text(str(session_dir.resolve()), encoding="utf-8")  # text fallback


def init_session(project_root: str | Path = ".", session_name: str = "Test 01") -> SessionPaths:
    """
    Create the mother folder once per session with:
      - Exported data
      - Initial point clouds
      - Initial images
      - Initial depth images
      - Merged point clouds
      - Merged images
      - Merged depth images
      - Camera intrinsics
    Note: the global transforms folder under helpers/ must be created by you manually.
    """
    project_root = Path(project_root).resolve()
    sessions_dir = _sessions_dir(project_root)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    session_dir = sessions_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    exported = session_dir / DIR_EXPORTED
    initial = session_dir / DIR_INITIAL
    initial_images = session_dir / DIR_INITIAL_IMAGES
    initial_depth_images = session_dir / DIR_INITIAL_DEPTH_IMAGES
    merged = session_dir / DIR_MERGED
    merged_images = session_dir / DIR_MERGED_IMAGES
    merged_depth_images = session_dir / DIR_MERGED_DEPTH_IMAGES
    camera_intrinsics = session_dir / DIR_INTRINSICS

    for d in (
        exported,
        initial,
        initial_images,
        initial_depth_images,
        merged,
        merged_images,
        merged_depth_images,
        camera_intrinsics,
    ):
        d.mkdir(exist_ok=True)

    if not GLOBAL_TRANSFORMS_DIR.exists():
        print(f"[warn] Global transforms folder not found at: {GLOBAL_TRANSFORMS_DIR}")
        print("       Create it manually once as 'helpers/Transformation matrix_3 capture'.")
    
    if not EYE_TO_BASE_TRANSFORMS_DIR.exists():
        print(f"[warn] Eye-to-base transforms folder not found at: {EYE_TO_BASE_TRANSFORMS_DIR}")
        print("       Create it manually once as 'helpers/Transformation matrix_eye_to_base'.")

    manifest = session_dir / "session.json"
    manifest.write_text(
        json.dumps(
            {
                "session_name": session_name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "root": str(session_dir.resolve()),
                "exported_data": str(exported.resolve()),
                "initial_point_clouds": str(initial.resolve()),
                "initial_images": str(initial_images.resolve()),
                "initial_depth_images": str(initial_depth_images.resolve()),
                "merged_point_clouds": str(merged.resolve()),
                "merged_images": str(merged_images.resolve()),
                "merged_depth_images": str(merged_depth_images.resolve()),
                "camera_intrinsics": str(camera_intrinsics.resolve()),
                "platform": platform.platform(),
                "global_transforms_root": str(GLOBAL_TRANSFORMS_DIR.resolve()),
                "eye_to_base_transforms_root": str(EYE_TO_BASE_TRANSFORMS_DIR.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _safe_latest_pointer(sessions_dir, session_dir)
    (project_root / RECENT_FILE).write_text(session_name, encoding="utf-8")

    return SessionPaths(
        root=session_dir,
        exported_data=exported,
        initial_point_clouds=initial,
        initial_images=initial_images,
        initial_depth_images=initial_depth_images,
        merged_point_clouds=merged,
        merged_images=merged_images,
        merged_depth_images=merged_depth_images,
        camera_intrinsics=camera_intrinsics,
        manifest=manifest,
        session_name=session_name,
        global_transforms_root=GLOBAL_TRANSFORMS_DIR,
        eye_to_base_transforms_root=EYE_TO_BASE_TRANSFORMS_DIR,
    )


def load_session(project_root: str | Path = ".", session_name: str | None = None) -> SessionPaths:
    """
    Load an existing session by name or fall back to the most recent.
    Checks .last_session, then sessions/latest (symlink or text file).
    """
    project_root = Path(project_root).resolve()
    sessions_dir = _sessions_dir(project_root)

    if session_name:
        session_dir = sessions_dir / session_name
    else:
        # try recent, then latest pointer
        recent = project_root / RECENT_FILE
        if recent.exists():
            session_dir = sessions_dir / recent.read_text(encoding="utf-8").strip()
        else:
            latest = sessions_dir / "latest"
            if latest.is_symlink():
                session_dir = (sessions_dir / latest.readlink()).resolve()
            elif latest.exists() and latest.is_file():
                session_dir = Path(latest.read_text(encoding="utf-8").strip())
            else:
                raise FileNotFoundError("No session provided and no recent/latest session found.")

    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    manifest = session_dir / "session.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        global_tf = Path(data.get("global_transforms_root", GLOBAL_TRANSFORMS_DIR))
        if not global_tf.exists():
            print(f"[warn] Global transforms folder missing at: {global_tf}")
            print("       Create it manually: helpers/Transformation matrix_3 capture")

        eye_to_base_tf = Path(data.get("eye_to_base_transforms_root", EYE_TO_BASE_TRANSFORMS_DIR))
        if not eye_to_base_tf.exists():
            print(f"[warn] Eye-to-base transforms folder missing at: {eye_to_base_tf}")
            print("       Create it manually: helpers/Transformation matrix_eye_to_base")

        return SessionPaths(
            root=session_dir,
            exported_data=Path(data["exported_data"]),
            initial_point_clouds=Path(data["initial_point_clouds"]),
            initial_images=Path(data["initial_images"]),
            initial_depth_images=Path(data["initial_depth_images"]),
            merged_point_clouds=Path(data["merged_point_clouds"]),
            merged_images=Path(data["merged_images"]),
            merged_depth_images=Path(data["merged_depth_images"]),
            camera_intrinsics=Path(data["camera_intrinsics"]),
            manifest=manifest,
            session_name=data["session_name"],
            global_transforms_root=global_tf,
            eye_to_base_transforms_root=eye_to_base_tf,
        )

    # reconstruct if manifest missing
    if not GLOBAL_TRANSFORMS_DIR.exists():
        print(f"[warn] Global transforms folder not found at: {GLOBAL_TRANSFORMS_DIR}")

    if not EYE_TO_BASE_TRANSFORMS_DIR.exists():
        print(f"[warn] Eye-to-base transforms folder not found at: {EYE_TO_BASE_TRANSFORMS_DIR}")
        
    return SessionPaths(
        root=session_dir,
        exported_data=session_dir / DIR_EXPORTED,
        initial_point_clouds=session_dir / DIR_INITIAL,
        initial_images=session_dir / DIR_INITIAL_IMAGES,
        initial_depth_images=session_dir / DIR_INITIAL_DEPTH_IMAGES,
        merged_point_clouds=session_dir / DIR_MERGED,
        merged_images=session_dir / DIR_MERGED_IMAGES,
        merged_depth_images=session_dir / DIR_MERGED_DEPTH_IMAGES,
        camera_intrinsics=session_dir / DIR_INTRINSICS,
        manifest=manifest,
        session_name=session_dir.name,
        global_transforms_root=GLOBAL_TRANSFORMS_DIR,
        eye_to_base_transforms_root=EYE_TO_BASE_TRANSFORMS_DIR,
    )


def transforms_dir_for_session(session: SessionPaths) -> Path:
    """
    Compute the per-session subfolder under the fixed global transforms root.
    We do NOT create it here (since we preferred manual management).
    """
    return session.global_transforms_root / session.session_name

def eye_to_base_transforms_dir_for_session(session: SessionPaths) -> Path:
    """
    Compute the per-session subfolder under the fixed eye-to-base transforms root.
    We do NOT create it here (since we preferred manual management).
    """
    return session.eye_to_base_transforms_root / session.session_name          