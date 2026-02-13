import os
from pathlib import Path


def _detect_git_root() -> Path:
    """Locate repository root; fallback to current file ancestor."""
    try:
        from git import Repo

        repo = Repo(os.getcwd(), search_parent_directories=True)
        return Path(repo.git.rev_parse("--show-toplevel"))
    except Exception:
        return Path(__file__).resolve().parents[4]


def _resolve_assets_root() -> str:
    """Return env override if provided, otherwise default assets directory."""
    env_root = os.environ.get("LEISAAC_ASSETS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve().as_posix()

    return (_detect_git_root() / "assets").resolve().as_posix()


ASSETS_ROOT = _resolve_assets_root()

SINGLE_ARM_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
BI_ARM_JOINT_NAMES = ['left_shoulder_pan', 'left_shoulder_lift', 'left_elbow_flex', 'left_wrist_flex', 'left_wrist_roll', 'left_gripper',
                      'right_shoulder_pan', 'right_shoulder_lift', 'right_elbow_flex', 'right_wrist_flex', 'right_wrist_roll', 'right_gripper']
XTRAINER_DUAL_JOINT_NAMES = [
    "J1_1", "J1_2", "J1_3", "J1_4", "J1_5", "J1_6", "J1_7", "J1_8",
    "J2_1", "J2_2", "J2_3", "J2_4", "J2_5", "J2_6", "J2_7", "J2_8"
]