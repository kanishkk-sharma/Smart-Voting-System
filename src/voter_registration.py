"""
voter_registration.py - Enrollment pipeline: images -> quality gate -> duplicate
check -> dataset folder -> encodings -> database.

Folder layout produced:
    dataset/<voter_id>/<timestamp>_<n>.jpg     raw face images (for rebuilds)
    encodings/encodings.npz                    128-d encodings of every voter
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .database import Database
from .face_recognition_module import EnrollmentCheck, FaceEngine
from .utils import Config, DATASET_DIR, get_logger, validate_voter_id

log = get_logger("registration")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class RegistrationReport:
    success: bool
    message: str
    accepted: int = 0
    rejected: int = 0
    voter_id: Optional[str] = None


class VoterRegistrar:
    def __init__(self, db: Database, engine: FaceEngine, dataset_dir: Path | str = DATASET_DIR):
        self.db = db
        self.engine = engine
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- helpers
    def assess_frame(self, frame: np.ndarray) -> EnrollmentCheck:
        """Used by the UI for live feedback while capturing."""
        return self.engine.analyze_enrollment_frame(frame, strict=True)

    def _voter_dir(self, voter_id: str) -> Path:
        return self.dataset_dir / voter_id

    def _count_images(self, voter_id: str) -> int:
        d = self._voter_dir(voter_id)
        return sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.exists() else 0

    def _encode_frames(self, frames: Sequence[np.ndarray], exclude_id: Optional[str]):
        """Return (good_frames, encodings, rejection_reasons, duplicate_of)."""
        good, encs, reasons = [], [], []
        duplicate_of: Optional[str] = None
        for i, frame in enumerate(frames, 1):
            check = self.engine.analyze_enrollment_frame(frame, strict=True)
            if not check.ok:
                reasons.append(f"image {i}: {check.message}")
                continue
            m = self.engine.match(check.encoding, exclude_id=exclude_id)
            if m.closest_id is not None and m.distance <= self.engine.tolerance:
                duplicate_of = m.closest_id          # same face already enrolled
                break
            good.append(frame)
            encs.append(check.encoding)
        return good, encs, reasons, duplicate_of

    def _save_images(self, voter_id: str, frames: Sequence[np.ndarray]) -> List[Path]:
        folder = self._voter_dir(voter_id)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        paths = []
        for i, f in enumerate(frames):
            p = folder / f"{stamp}_{i:02d}.jpg"
            if not cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError(f"Could not write {p}")
            paths.append(p)
        return paths

    # --------------------------------------------------------------- register
    def register(self, voter_id: str, name: str, frames: Sequence[np.ndarray]) -> RegistrationReport:
        voter_id, name = voter_id.strip(), name.strip()
        if not validate_voter_id(voter_id):
            return RegistrationReport(False, "Voter ID must be 3-32 characters: letters, digits, '_' or '-'.")
        if not name:
            return RegistrationReport(False, "Name is required.")
        if self.db.get_voter(voter_id):
            return RegistrationReport(False, f"Voter ID '{voter_id}' already exists.")

        good, encs, reasons, dup = self._encode_frames(frames, exclude_id=None)
        if dup:
            self.db.log_event("REGISTER_REJECTED_DUPLICATE", f"id={voter_id} matches={dup}")
            return RegistrationReport(False, f"This face is already registered as voter '{dup}'.",
                                      rejected=len(frames))
        if len(good) < Config.MIN_REG_IMAGES:
            detail = "; ".join(reasons[:3])
            return RegistrationReport(
                False, f"Only {len(good)} usable image(s); need at least {Config.MIN_REG_IMAGES}. {detail}",
                len(good), len(frames) - len(good))

        created_dir = not self._voter_dir(voter_id).exists()
        try:
            self._save_images(voter_id, good)
            self.db.add_voter(voter_id, name, len(good))
            self.engine.set_voter_encodings(voter_id, encs, replace=True)
        except Exception as exc:                      # roll back everything on any failure
            log.exception("Registration failed for %s", voter_id)
            if created_dir:
                shutil.rmtree(self._voter_dir(voter_id), ignore_errors=True)
            self.db.delete_voter(voter_id)
            self.engine.remove_voter(voter_id)
            return RegistrationReport(False, f"Registration failed: {exc}")

        self.db.log_event("VOTER_REGISTERED", f"id={voter_id} images={len(good)}")
        return RegistrationReport(True, f"Registered {name} ({voter_id}) with {len(good)} images.",
                                  len(good), len(frames) - len(good), voter_id)

    # ----------------------------------------------------------------- update
    def update(self, voter_id: str, name: Optional[str] = None,
               frames: Optional[Sequence[np.ndarray]] = None,
               replace_images: bool = False) -> RegistrationReport:
        if not self.db.get_voter(voter_id):
            return RegistrationReport(False, f"Voter '{voter_id}' not found.")
        if name is not None and not name.strip():
            return RegistrationReport(False, "Name cannot be empty.")

        if frames:
            good, encs, reasons, dup = self._encode_frames(frames, exclude_id=voter_id)
            if dup:
                return RegistrationReport(False, f"These images match another voter ('{dup}').")
            need = Config.MIN_REG_IMAGES if replace_images else 1
            if len(good) < need:
                return RegistrationReport(False, f"Only {len(good)} usable image(s); need {need}. "
                                                 + "; ".join(reasons[:3]))
            if replace_images:
                shutil.rmtree(self._voter_dir(voter_id), ignore_errors=True)
            self._save_images(voter_id, good)
            self.engine.set_voter_encodings(voter_id, encs, replace=replace_images)
            self.db.update_voter(voter_id, num_images=self._count_images(voter_id))

        if name is not None:
            self.db.update_voter(voter_id, name=name)
        self.db.log_event("VOTER_UPDATED", f"id={voter_id}")
        return RegistrationReport(True, f"Updated voter '{voter_id}'.", voter_id=voter_id)

    # ----------------------------------------------------------------- delete
    def delete(self, voter_id: str) -> RegistrationReport:
        if not self.db.get_voter(voter_id):
            return RegistrationReport(False, f"Voter '{voter_id}' not found.")
        if self.db.has_voted(voter_id):
            # Deleting would free the face, letting the same person re-enroll and vote twice.
            return RegistrationReport(False, "This voter has already voted and cannot be deleted.")
        shutil.rmtree(self._voter_dir(voter_id), ignore_errors=True)
        self.engine.remove_voter(voter_id)
        self.db.delete_voter(voter_id)
        self.db.log_event("VOTER_DELETED", f"id={voter_id}")
        return RegistrationReport(True, f"Deleted voter '{voter_id}'.", voter_id=voter_id)

    # ---------------------------------------------------------------- rebuild
    def rebuild_encodings(self) -> RegistrationReport:
        """Re-encode every image in dataset/ (recovery after a lost/corrupt encodings file)."""
        self.engine.clear()
        total, voters = 0, 0
        for voter in self.db.list_voters():
            encs = []
            folder = self._voter_dir(voter["voter_id"])
            for p in sorted(folder.glob("*")) if folder.exists() else []:
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                check = self.engine.analyze_enrollment_frame(img, strict=False)
                if check.ok:
                    encs.append(check.encoding)
            if encs:
                self.engine.set_voter_encodings(voter["voter_id"], encs)
                voters += 1
                total += len(encs)
        self.db.log_event("ENCODINGS_REBUILT", f"voters={voters} encodings={total}")
        return RegistrationReport(True, f"Rebuilt {total} encodings for {voters} voter(s).")
