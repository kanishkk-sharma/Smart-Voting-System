"""
voting_system.py - Authentication state machine + secure vote casting.

Stages handled here:
    live frame -> recognition -> multi-frame confirmation -> liveness (blink)
    -> AUTHENTICATED -> eligibility (not already voted) -> cast vote.

AuthSession is a small per-attempt state machine. It is deliberately UI-agnostic:
`process(frame)` returns an AuthResult with an annotated frame and a message the
UI can display as-is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .database import AlreadyVotedError, Database
from .face_recognition_module import FaceEngine
from .utils import (BLUE, Config, GREEN, ORANGE, RED, draw_face_box, get_logger, put_banner)

log = get_logger("voting")


class AuthState(str, Enum):
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    UNRECOGNIZED = "unrecognized"        # transient
    VERIFYING = "verifying"              # transient: recognized, confirming
    BLINK_REQUIRED = "blink_required"    # transient: liveness
    AUTHENTICATED = "authenticated"      # terminal
    ALREADY_VOTED = "already_voted"      # terminal
    DENIED = "denied"                    # terminal

    @property
    def terminal(self) -> bool:
        return self in (AuthState.AUTHENTICATED, AuthState.ALREADY_VOTED, AuthState.DENIED)


@dataclass
class AuthResult:
    state: AuthState
    message: str
    level: str = "info"                  # info | success | warning | error
    voter_id: Optional[str] = None
    name: Optional[str] = None
    confidence: float = 0.0
    progress: float = 0.0                # 0..1 verification progress
    annotated: Optional[np.ndarray] = None


@dataclass
class VoteReceipt:
    vote_id: int
    timestamp: str
    entry_hash: str
    candidate_name: str


class AuthSession:
    """One face-authentication attempt."""

    def __init__(self, engine: FaceEngine, db: Database, require_liveness: bool = True,
                 required_consecutive: int = Config.REQUIRED_CONSECUTIVE,
                 deny_after: int = Config.DENY_AFTER_FRAMES):
        self.engine, self.db = engine, db
        self.require_liveness = require_liveness
        self.required = required_consecutive
        self.deny_after = deny_after
        self._unknown = 0
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        self._candidate: Optional[str] = None
        self._name: Optional[str] = None
        self._streak = 0
        self._eyes_closed = False
        self._blinked = False

    def _update_blink(self, ear: Optional[float]) -> None:
        if ear is None:
            return
        if ear < Config.EAR_THRESHOLD:
            self._eyes_closed = True
        elif self._eyes_closed:              # eyes re-opened after being closed => blink
            self._blinked = True

    # --------------------------------------------------------------- main step
    def process(self, frame: np.ndarray) -> AuthResult:
        annotated = frame.copy()
        observations = self.engine.observe(frame, compute_ear=self.require_liveness)

        if not observations:
            self._reset_tracking()
            self._unknown = 0
            put_banner(annotated, "No face detected", ORANGE)
            return AuthResult(AuthState.NO_FACE, "No face detected - please look at the camera.",
                              "info", annotated=annotated)

        if len(observations) > 1:
            self._reset_tracking()
            for o in observations:
                draw_face_box(annotated, o.box, "Multiple faces", ORANGE)
            put_banner(annotated, "Only one person may stand in front of the camera", ORANGE)
            return AuthResult(AuthState.MULTIPLE_FACES, "Multiple faces detected - one person at a time.",
                              "warning", annotated=annotated)

        obs = observations[0]
        m = obs.match

        # ------------------------------------------------ face not recognized
        if m.voter_id is None:
            self._reset_tracking()
            self._unknown += 1
            draw_face_box(annotated, obs.box, f"Unknown {m.confidence:.0%}", RED)
            if self._unknown >= self.deny_after:
                self.db.log_event("AUTH_DENIED",
                                  f"closest={m.closest_id} dist={m.distance:.3f} ambiguous={m.ambiguous}")
                put_banner(annotated, "ACCESS DENIED", RED)
                return AuthResult(AuthState.DENIED, "Access Denied - face not recognized as a registered voter.",
                                  "error", confidence=m.confidence, annotated=annotated)
            put_banner(annotated, f"Face detected - not recognized ({self._unknown}/{self.deny_after})", ORANGE)
            return AuthResult(AuthState.UNRECOGNIZED,
                              f"Face detected - not recognized yet ({self._unknown}/{self.deny_after}).",
                              "warning", confidence=m.confidence, annotated=annotated)

        # --------------------------------------------------- face recognized
        self._unknown = 0
        if m.voter_id != self._candidate:            # identity changed mid-scan -> restart
            self._reset_tracking()
            self._candidate = m.voter_id
            voter = self.db.get_voter(m.voter_id)
            self._name = voter["name"] if voter else m.voter_id
        self._streak += 1
        self._update_blink(obs.ear)

        label = f"{self._name} {m.confidence:.0%}"
        progress = min(1.0, self._streak / self.required)
        base = dict(voter_id=m.voter_id, name=self._name, confidence=m.confidence,
                    progress=progress, annotated=annotated)
        recog_msg = f"Recognized as {self._name} - Confidence: {m.confidence:.0%}"

        if self._streak < self.required:
            draw_face_box(annotated, obs.box, label, BLUE)
            put_banner(annotated, f"Verifying... {self._streak}/{self.required}", BLUE)
            return AuthResult(AuthState.VERIFYING, recog_msg + " - verifying...", "info", **base)

        # Identity confirmed over several consecutive frames -> eligibility
        if self.db.has_voted(m.voter_id):
            draw_face_box(annotated, obs.box, label, RED)
            put_banner(annotated, "ALREADY VOTED", RED)
            self.db.log_event("AUTH_ALREADY_VOTED", f"id={m.voter_id}")
            return AuthResult(AuthState.ALREADY_VOTED, f"Already voted - {self._name} has used their vote.",
                              "error", **base)

        # Liveness (anti-photo): require one blink
        if self.require_liveness and not self._blinked:
            draw_face_box(annotated, obs.box, label, ORANGE)
            put_banner(annotated, "Please BLINK once to prove you are live", ORANGE)
            return AuthResult(AuthState.BLINK_REQUIRED, recog_msg + " - please blink once.",
                              "warning", **base)

        draw_face_box(annotated, obs.box, label, GREEN)
        put_banner(annotated, "AUTHENTICATED", GREEN)
        self.db.log_event("AUTH_SUCCESS", f"id={m.voter_id} conf={m.confidence:.3f}")
        return AuthResult(AuthState.AUTHENTICATED, recog_msg + " - authenticated.", "success", **base)


class VotingSystem:
    """Facade used by the UI: auth sessions, ballot, vote casting, results."""

    def __init__(self, db: Database, engine: FaceEngine):
        self.db, self.engine = db, engine

    def new_auth_session(self, require_liveness: bool = True) -> AuthSession:
        return AuthSession(self.engine, self.db, require_liveness)

    def candidates(self) -> List[Dict[str, Any]]:
        return self.db.list_candidates(active_only=True)

    def cast_vote(self, voter_id: str, candidate_id: int) -> VoteReceipt:
        """Record a vote for an already-authenticated voter.

        Raises AlreadyVotedError (double vote) or ValueError (bad voter/candidate).
        The database enforces one-vote-per-voter atomically.
        """
        try:
            r = self.db.cast_vote(voter_id, candidate_id)
        except AlreadyVotedError:
            self.db.log_event("VOTE_BLOCKED_DOUBLE", f"id={voter_id}")
            raise
        self.db.log_event("VOTE_CAST", f"vote_id={r['vote_id']}")   # no voter or candidate info
        return VoteReceipt(r["vote_id"], r["timestamp"], r["entry_hash"], r["candidate_name"])

    def results(self) -> List[Dict[str, Any]]:
        return self.db.get_results()

    def turnout(self) -> Tuple[int, int]:
        """(votes cast, registered voters)"""
        return self.db.total_votes(), len(self.db.list_voters())

    def verify_integrity(self) -> Tuple[bool, str]:
        return self.db.verify_chain()
