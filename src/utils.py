"""
utils.py - Shared configuration, security helpers, camera and drawing utilities.

Everything that more than one module needs lives here so the other modules stay
focused on a single pipeline stage.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"          # dataset/<voter_id>/*.jpg
ENCODINGS_DIR = BASE_DIR / "encodings"      # encodings/encodings.npz
MODELS_DIR = BASE_DIR / "models"            # secret key (+ optional custom models)
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "voting.db"
ENCODINGS_FILE = ENCODINGS_DIR / "encodings.npz"
SECRET_KEY_FILE = MODELS_DIR / ".secret_key"


# --------------------------------------------------------------------------- #
# Tunable parameters (kept in one place on purpose)
# --------------------------------------------------------------------------- #
class Config:
    # Recognition
    DETECT_SCALE = 0.5          # downscale factor for real-time detection (speed)
    MATCH_TOLERANCE = 0.45      # max face distance accepted (lower = stricter)
    MIN_CONFIDENCE = 0.55       # min confidence to accept a match
    AMBIGUITY_MARGIN = 0.04     # reject if 2 voters are this close to each other

    # Authentication session
    REQUIRED_CONSECUTIVE = 6    # consecutive frames that must agree on identity
    DENY_AFTER_FRAMES = 20      # consecutive unknown frames before "Access Denied"
    SCAN_TIMEOUT_S = 45         # give up scanning after this many seconds
    BALLOT_TIMEOUT_S = 60       # authenticated voter must vote within this time

    # Liveness (blink detection)
    EAR_THRESHOLD = 0.21        # eye-aspect-ratio below this == eyes closed

    # Enrollment quality gates
    MIN_REG_IMAGES = 5
    TARGET_REG_IMAGES = 8
    MIN_FACE_PX = 90            # smallest acceptable face (pixels)
    MIN_SHARPNESS = 25.0        # Laplacian variance - rejects blurry captures
    MAX_IMAGE_SIDE = 1280       # large uploads are downscaled before analysis


VOTER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")   # also blocks path traversal

# BGR colours
GREEN = (60, 180, 60)
RED = (50, 50, 220)
ORANGE = (0, 150, 255)
BLUE = (200, 120, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# Filesystem / logging
# --------------------------------------------------------------------------- #
def ensure_directories() -> None:
    for d in (DATASET_DIR, ENCODINGS_DIR, MODELS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the app-wide logger once (safe to call repeatedly)."""
    logger = logging.getLogger("smart_voting")
    if logger.handlers:
        return logger
    ensure_directories()
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    fh = RotatingFileHandler(LOGS_DIR / "voting.log", maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"smart_voting.{name}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_voter_id(voter_id: str) -> bool:
    return bool(voter_id and VOTER_ID_RE.match(voter_id))


# --------------------------------------------------------------------------- #
# Cryptographic helpers
# --------------------------------------------------------------------------- #
def get_secret_key() -> bytes:
    """Load (or create on first run) the server-side secret used for HMACs.

    The key never leaves this machine. Without it, the voter-hash column in
    the vote log cannot be linked back to a voter ID by brute force.
    """
    ensure_directories()
    if SECRET_KEY_FILE.exists():
        return bytes.fromhex(SECRET_KEY_FILE.read_text().strip())
    key = secrets.token_bytes(32)
    SECRET_KEY_FILE.write_text(key.hex())
    try:
        os.chmod(SECRET_KEY_FILE, 0o600)     # owner read/write only (no-op on Windows)
    except OSError:
        pass
    return key


def hash_voter_id(voter_id: str, key: bytes) -> str:
    """Keyed hash (HMAC-SHA256) of a voter ID - used in the vote log."""
    return hmac.new(key, voter_id.encode("utf-8"), hashlib.sha256).hexdigest()


def hash_password(password: str, iterations: int = 200_000) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"{iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        iterations, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


def open_camera(index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        raise CameraError(
            f"Cannot open camera #{index}. Check that it is connected, not used by "
            "another application, and that camera permission is granted."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(3):          # let auto-exposure settle
        cap.read()
    return cap


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def sharpness(frame: np.ndarray, box: Optional[Tuple[int, int, int, int]] = None) -> float:
    """Variance of the Laplacian - low values mean a blurry image."""
    if box is not None:
        top, right, bottom, left = box
        frame = frame[max(top, 0):bottom, max(left, 0):right]
    if frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def draw_face_box(frame: np.ndarray, box: Tuple[int, int, int, int], label: str,
                  color: Tuple[int, int, int]) -> None:
    """Draw a coloured rectangle with a label above it (in-place)."""
    top, right, bottom, left = box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    if label:
        (tw, th), base = cv2.getTextSize(label, FONT, 0.55, 1)
        y0 = max(0, top - th - base - 8)
        cv2.rectangle(frame, (left, y0), (left + tw + 10, y0 + th + base + 8), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 5, y0 + th + 3), FONT, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)


def put_banner(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    """Draw a status bar along the bottom of the frame (in-place)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 34), (w, h), color, cv2.FILLED)
    cv2.putText(frame, text, (10, h - 11), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
