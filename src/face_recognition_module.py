"""
face_recognition_module.py - Detection, 128-d encoding, matching and liveness.

Pipeline stage: frame -> detect faces -> encode (dlib ResNet, 128-d) -> compare
against registered voters -> MatchResult(identity, distance, confidence).

Encodings are stored in a plain NumPy .npz archive (no pickle => no code
execution risk when loading) and written atomically.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import face_recognition
except ImportError as exc:     # pragma: no cover
    raise ImportError(
        "The 'face_recognition' package is required. See README.md for installation "
        "(it needs dlib, which needs CMake and a C++ compiler)."
    ) from exc

from .utils import (Config, ENCODINGS_FILE, get_logger, resize_max_side, sharpness)

log = get_logger("face_engine")
Box = Tuple[int, int, int, int]      # (top, right, bottom, left)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class MatchResult:
    voter_id: Optional[str]          # None => not accepted
    distance: float                  # distance to the closest voter (lower = closer)
    confidence: float                # 0..1
    closest_id: Optional[str] = None
    ambiguous: bool = False


@dataclass
class FaceObservation:
    box: Box
    encoding: np.ndarray
    match: MatchResult
    ear: Optional[float] = None      # eye aspect ratio (liveness), if requested


@dataclass
class EnrollmentCheck:
    ok: bool
    message: str
    box: Optional[Box] = None
    encoding: Optional[np.ndarray] = None


def distance_to_confidence(distance: float, tolerance: float) -> float:
    """Map a dlib face distance to a human-friendly 0..1 confidence.

    distance == tolerance -> 50 %,   distance -> 0 -> ~100 %.
    """
    span = 1.0 - tolerance
    linear = max(0.0, (1.0 - distance) / (span * 2.0))
    if distance > tolerance:
        return float(min(1.0, linear))
    value = linear + (1.0 - linear) * math.pow(max(0.0, (linear - 0.5) * 2.0), 0.2)
    return float(min(1.0, value))


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class FaceEngine:
    def __init__(self, encodings_file: Path | str = ENCODINGS_FILE,
                 tolerance: float = Config.MATCH_TOLERANCE,
                 min_confidence: float = Config.MIN_CONFIDENCE):
        self.encodings_file = Path(encodings_file)
        self.tolerance = tolerance
        self.min_confidence = min_confidence
        self._lock = threading.RLock()
        self._ids: np.ndarray = np.empty((0,), dtype=str)
        self._encs: np.ndarray = np.empty((0, 128), dtype=np.float64)
        self.load()

    # ---------------------------------------------------------- persistence
    def load(self) -> None:
        with self._lock:
            if not self.encodings_file.exists():
                self._ids, self._encs = np.empty((0,), dtype=str), np.empty((0, 128))
                return
            try:
                data = np.load(self.encodings_file, allow_pickle=False)
                self._ids = data["voter_ids"].astype(str)
                self._encs = data["encodings"].astype(np.float64)
                log.info("Loaded %d encodings for %d voters", len(self._ids), self.voter_count)
            except Exception:
                log.exception("Encoding file unreadable - starting empty. "
                              "Use 'Rebuild encodings' in the admin panel.")
                self._ids, self._encs = np.empty((0,), dtype=str), np.empty((0, 128))

    def save(self) -> None:
        """Atomic write: temp file + rename, so a crash never corrupts the DB."""
        with self._lock:
            self.encodings_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.encodings_file.with_name("encodings.tmp.npz")
            np.savez_compressed(tmp, voter_ids=self._ids, encodings=self._encs)
            os.replace(tmp, self.encodings_file)

    # ------------------------------------------------------------ voter data
    @property
    def voter_count(self) -> int:
        return len(set(self._ids.tolist()))

    def has_voter(self, voter_id: str) -> bool:
        return bool((self._ids == voter_id).any())

    def set_voter_encodings(self, voter_id: str, encodings: Sequence[np.ndarray],
                            replace: bool = True) -> None:
        with self._lock:
            ids, encs = self._ids, self._encs
            if replace:
                keep = ids != voter_id
                ids, encs = ids[keep], encs[keep]
            new = np.asarray(encodings, dtype=np.float64).reshape(-1, 128)
            self._ids = np.concatenate([ids, np.full(len(new), voter_id)])
            self._encs = np.vstack([encs, new])
            self.save()

    def remove_voter(self, voter_id: str) -> None:
        with self._lock:
            keep = self._ids != voter_id
            self._ids, self._encs = self._ids[keep], self._encs[keep]
            self.save()

    def clear(self) -> None:
        with self._lock:
            self._ids, self._encs = np.empty((0,), dtype=str), np.empty((0, 128))
            self.save()

    # ------------------------------------------------------------- matching
    def match(self, encoding: np.ndarray, exclude_id: Optional[str] = None) -> MatchResult:
        """Compare one encoding against all registered voters."""
        with self._lock:
            ids, encs = self._ids, self._encs
        if exclude_id is not None:
            mask = ids != exclude_id
            ids, encs = ids[mask], encs[mask]
        if len(ids) == 0:
            return MatchResult(None, 1.0, 0.0)

        dists = face_recognition.face_distance(encs, encoding)
        best: Dict[str, float] = {}
        for vid, d in zip(ids, dists):               # best distance per voter
            if vid not in best or d < best[vid]:
                best[str(vid)] = float(d)
        ranked = sorted(best.items(), key=lambda kv: kv[1])
        closest_id, d = ranked[0]
        conf = distance_to_confidence(d, self.tolerance)

        # Two different voters almost equally close => refuse (safer than guessing)
        ambiguous = (len(ranked) > 1 and ranked[1][1] <= self.tolerance
                     and (ranked[1][1] - d) < Config.AMBIGUITY_MARGIN)
        accepted = d <= self.tolerance and conf >= self.min_confidence and not ambiguous
        return MatchResult(closest_id if accepted else None, d, conf, closest_id, ambiguous)

    # ------------------------------------------------------------ real-time
    def observe(self, frame_bgr: np.ndarray, compute_ear: bool = False) -> List[FaceObservation]:
        """Detect + encode + match every face in a live frame (fast path)."""
        scale = Config.DETECT_SCALE
        small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        boxes_small = face_recognition.face_locations(
            cv2.cvtColor(small, cv2.COLOR_BGR2RGB), model="hog")
        if not boxes_small:
            return []

        h, w = frame_bgr.shape[:2]
        inv = 1.0 / scale
        boxes = [(max(0, int(t * inv)), min(w, int(r * inv)),
                  min(h, int(b * inv)), max(0, int(l * inv))) for t, r, b, l in boxes_small]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=1)

        out = []
        for box, enc in zip(boxes, encodings):
            ear = self._eye_aspect_ratio(rgb, box) if compute_ear else None
            out.append(FaceObservation(box, enc, self.match(enc), ear))
        return out

    @staticmethod
    def _eye_aspect_ratio(rgb: np.ndarray, box: Box) -> Optional[float]:
        """Mean EAR of both eyes (Soukupova & Cech, 2016). Low => eyes closed."""
        lm = face_recognition.face_landmarks(rgb, [box])
        if not lm:
            return None

        def ear(pts) -> float:
            p = np.asarray(pts, dtype=float)
            return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (
                2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6)

        return (ear(lm[0]["left_eye"]) + ear(lm[0]["right_eye"])) / 2.0

    # ----------------------------------------------------------- enrollment
    def analyze_enrollment_frame(self, frame_bgr: np.ndarray, strict: bool = True) -> EnrollmentCheck:
        """Quality-gate one image: exactly one, large, sharp face -> encoding."""
        frame = resize_max_side(frame_bgr, Config.MAX_IMAGE_SIDE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="hog")
        if not boxes:
            return EnrollmentCheck(False, "No face detected")
        if len(boxes) > 1:
            return EnrollmentCheck(False, "Multiple faces - only one person at a time", boxes[0])
        box = boxes[0]
        top, right, bottom, left = box
        if strict:
            if min(bottom - top, right - left) < Config.MIN_FACE_PX:
                return EnrollmentCheck(False, "Move closer to the camera", box)
            if sharpness(frame, box) < Config.MIN_SHARPNESS:
                return EnrollmentCheck(False, "Image too blurry - hold still", box)
        enc = face_recognition.face_encodings(rgb, [box], num_jitters=2)[0]
        return EnrollmentCheck(True, "Good", box, enc)
