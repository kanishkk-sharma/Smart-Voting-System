"""
database.py - SQLite persistence for voters, candidates, votes, audit log, settings.

Security design
---------------
* One person = one vote is enforced by the DATABASE (UNIQUE constraint on the
  keyed voter hash), not just by application logic.
* The vote log stores only HMAC(voter_id) - never the plain voter ID.
* Votes form a hash chain: each row's `entry_hash` is an HMAC over the previous
  row's hash + its own content. Editing any historical row breaks the chain,
  which `verify_chain()` detects.
* SQLite triggers make the `votes` table append-only (UPDATE/DELETE aborted).
* Cast-vote runs in a BEGIN IMMEDIATE transaction => safe under concurrency.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .utils import (DB_PATH, get_logger, get_secret_key, hash_password,
                    hash_voter_id, utc_now_iso, verify_password)

log = get_logger("database")
GENESIS_HASH = "0" * 64
ADMIN_USER_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


class AlreadyVotedError(Exception):
    """Raised when a voter tries to vote a second time."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS voters (
    voter_id      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    num_images    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL UNIQUE,
    party         TEXT NOT NULL DEFAULT '',
    active        INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS votes (
    vote_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    voter_hash    TEXT NOT NULL UNIQUE,            -- one vote per voter, enforced here
    candidate_id  INTEGER NOT NULL REFERENCES candidates(candidate_id),
    timestamp     TEXT NOT NULL,
    prev_hash     TEXT NOT NULL,
    entry_hash    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event     TEXT NOT NULL,
    details   TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS admins (
    admin_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS votes_no_update BEFORE UPDATE ON votes
BEGIN SELECT RAISE(ABORT, 'votes are immutable'); END;
CREATE TRIGGER IF NOT EXISTS votes_no_delete BEFORE DELETE ON votes
BEGIN SELECT RAISE(ABORT, 'votes are immutable'); END;
"""


class Database:
    def __init__(self, path: Path | str = DB_PATH, secret_key: Optional[bytes] = None):
        self.path = Path(path)
        self._key = secret_key or get_secret_key()
        self._init_schema()

    # ------------------------------------------------------------------ infra
    @contextmanager
    def _conn(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Short-lived connection with explicit transaction (thread-safe)."""
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self.path, isolation_level=None)
        try:
            conn.executescript(SCHEMA)
            # Migrate the old single shared admin password (v1) to an 'admin' account
            legacy = conn.execute("SELECT value FROM settings WHERE key='admin_password'").fetchone()
            if legacy and conn.execute("SELECT COUNT(*) FROM admins").fetchone()[0] == 0:
                conn.execute("INSERT INTO admins (username, password_hash, created_at) VALUES (?,?,?)",
                             ("admin", legacy[0], utc_now_iso()))
                conn.execute("DELETE FROM settings WHERE key='admin_password'")
        finally:
            conn.close()

    def _entry_hash(self, prev: str, voter_hash: str, candidate_id: int, ts: str) -> str:
        msg = f"{prev}|{voter_hash}|{candidate_id}|{ts}".encode()
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    # ----------------------------------------------------------------- voters
    def add_voter(self, voter_id: str, name: str, num_images: int) -> None:
        try:
            with self._conn() as c:
                c.execute("INSERT INTO voters VALUES (?,?,?,?)",
                          (voter_id, name.strip(), utc_now_iso(), num_images))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Voter ID '{voter_id}' already exists.") from exc

    def update_voter(self, voter_id: str, name: Optional[str] = None,
                     num_images: Optional[int] = None) -> None:
        with self._conn() as c:
            if name is not None:
                c.execute("UPDATE voters SET name=? WHERE voter_id=?", (name.strip(), voter_id))
            if num_images is not None:
                c.execute("UPDATE voters SET num_images=? WHERE voter_id=?", (num_images, voter_id))

    def delete_voter(self, voter_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM voters WHERE voter_id=?", (voter_id,))

    def get_voter(self, voter_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM voters WHERE voter_id=?", (voter_id,)).fetchone()
        return dict(row) if row else None

    def list_voters(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM voters ORDER BY registered_at").fetchall()
            voted = {r["voter_hash"] for r in c.execute("SELECT voter_hash FROM votes")}
        return [{**dict(r), "has_voted": hash_voter_id(r["voter_id"], self._key) in voted}
                for r in rows]

    def has_voted(self, voter_id: str) -> bool:
        vh = hash_voter_id(voter_id, self._key)
        with self._conn() as c:
            return c.execute("SELECT 1 FROM votes WHERE voter_hash=?", (vh,)).fetchone() is not None

    # ------------------------------------------------------------- candidates
    def add_candidate(self, name: str, party: str = "") -> int:
        name = name.strip()
        if not name:
            raise ValueError("Candidate name cannot be empty.")
        try:
            with self._conn() as c:
                cur = c.execute("INSERT INTO candidates (name, party) VALUES (?,?)",
                                (name, party.strip()))
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Candidate '{name}' already exists.") from exc

    def list_candidates(self, active_only: bool = True) -> List[Dict[str, Any]]:
        q = "SELECT * FROM candidates" + (" WHERE active=1" if active_only else "") + " ORDER BY candidate_id"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q).fetchall()]

    def update_candidate(self, candidate_id: int, name: Optional[str] = None,
                         party: Optional[str] = None, active: Optional[bool] = None) -> None:
        """Edit a candidate's name / party / active flag (votes reference the ID, so history stays valid)."""
        if name is not None and not name.strip():
            raise ValueError("Candidate name cannot be empty.")
        try:
            with self._conn() as c:
                if not c.execute("SELECT 1 FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone():
                    raise ValueError("Candidate not found.")
                if name is not None:
                    c.execute("UPDATE candidates SET name=? WHERE candidate_id=?", (name.strip(), candidate_id))
                if party is not None:
                    c.execute("UPDATE candidates SET party=? WHERE candidate_id=?", (party.strip(), candidate_id))
                if active is not None:
                    c.execute("UPDATE candidates SET active=? WHERE candidate_id=?", (int(active), candidate_id))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Candidate '{name}' already exists.") from exc

    def remove_candidate(self, candidate_id: int) -> str:
        """Delete a candidate with no votes; otherwise deactivate (keeps history valid)."""
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM votes WHERE candidate_id=?", (candidate_id,)).fetchone()[0]
            if n:
                c.execute("UPDATE candidates SET active=0 WHERE candidate_id=?", (candidate_id,))
                return "deactivated"
            c.execute("DELETE FROM candidates WHERE candidate_id=?", (candidate_id,))
            return "deleted"

    # ------------------------------------------------------------------ votes
    def cast_vote(self, voter_id: str, candidate_id: int) -> Dict[str, Any]:
        """Atomically record a vote. Raises AlreadyVotedError on a second attempt."""
        vh = hash_voter_id(voter_id, self._key)
        ts = utc_now_iso()
        try:
            with self._conn(immediate=True) as c:
                if not c.execute("SELECT 1 FROM voters WHERE voter_id=?", (voter_id,)).fetchone():
                    raise ValueError("Voter is not registered.")
                cand = c.execute("SELECT name, active FROM candidates WHERE candidate_id=?",
                                 (candidate_id,)).fetchone()
                if not cand or not cand["active"]:
                    raise ValueError("Invalid candidate.")
                if c.execute("SELECT 1 FROM votes WHERE voter_hash=?", (vh,)).fetchone():
                    raise AlreadyVotedError(voter_id)
                last = c.execute("SELECT entry_hash FROM votes ORDER BY vote_id DESC LIMIT 1").fetchone()
                prev = last["entry_hash"] if last else GENESIS_HASH
                entry = self._entry_hash(prev, vh, candidate_id, ts)
                cur = c.execute(
                    "INSERT INTO votes (voter_hash, candidate_id, timestamp, prev_hash, entry_hash) "
                    "VALUES (?,?,?,?,?)", (vh, candidate_id, ts, prev, entry))
                return {"vote_id": cur.lastrowid, "timestamp": ts, "entry_hash": entry,
                        "candidate_name": cand["name"]}
        except sqlite3.IntegrityError as exc:       # UNIQUE(voter_hash) race-condition backstop
            raise AlreadyVotedError(voter_id) from exc

    def get_results(self) -> List[Dict[str, Any]]:
        q = """SELECT c.candidate_id, c.name, c.party, c.active, COUNT(v.vote_id) AS votes
               FROM candidates c LEFT JOIN votes v ON v.candidate_id = c.candidate_id
               GROUP BY c.candidate_id
               HAVING c.active = 1 OR COUNT(v.vote_id) > 0
               ORDER BY votes DESC, c.name"""
        with self._conn() as c:
            return [dict(r) for r in c.execute(q).fetchall()]

    def total_votes(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM votes").fetchone()[0]

    def export_votes(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT vote_id, voter_hash, candidate_id, timestamp, entry_hash "
                "FROM votes ORDER BY vote_id")]

    def verify_chain(self) -> Tuple[bool, str]:
        """Recompute the whole hash chain; returns (ok, message)."""
        prev = GENESIS_HASH
        with self._conn() as c:
            rows = c.execute("SELECT * FROM votes ORDER BY vote_id").fetchall()
        for r in rows:
            expected = self._entry_hash(prev, r["voter_hash"], r["candidate_id"], r["timestamp"])
            if r["prev_hash"] != prev or r["entry_hash"] != expected:
                return False, f"Tampering detected at vote #{r['vote_id']}."
            prev = r["entry_hash"]
        return True, f"Integrity verified - {len(rows)} vote(s), chain intact."

    # ------------------------------------------------------------------ audit
    def log_event(self, event: str, details: str = "") -> None:
        try:
            with self._conn() as c:
                c.execute("INSERT INTO audit_log (timestamp, event, details) VALUES (?,?,?)",
                          (utc_now_iso(), event, details))
        except sqlite3.Error:
            log.exception("Failed to write audit log")
        log.info("AUDIT %s %s", event, details)

    def recent_audit(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT timestamp, event, details FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]

    # --------------------------------------------------------------- settings
    def get_setting(self, key: str, default: str = "") -> str:
        with self._conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO settings (key, value) VALUES (?,?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    # ------------------------------------------------------------ admin accounts
    @staticmethod
    def _validate_admin_credentials(username: Optional[str], password: Optional[str]) -> None:
        if username is not None and not ADMIN_USER_RE.match(username):
            raise ValueError("Username must be 3-32 characters: letters, digits, '.', '_' or '-'.")
        if password is not None and len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")

    def has_admins(self) -> bool:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM admins").fetchone()[0] > 0

    def create_admin(self, username: str, password: str) -> int:
        username = username.strip()
        self._validate_admin_credentials(username, password)
        try:
            with self._conn() as c:
                cur = c.execute("INSERT INTO admins (username, password_hash, created_at) VALUES (?,?,?)",
                                (username, hash_password(password), utc_now_iso()))
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Administrator '{username}' already exists.") from exc

    def authenticate_admin(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Return {'admin_id', 'username'} on success, else None."""
        with self._conn() as c:
            row = c.execute("SELECT * FROM admins WHERE username=?", (username.strip(),)).fetchone()
        if row is None:
            verify_password(password, hash_password("dummy-password"))   # equalise timing
            return None
        if verify_password(password, row["password_hash"]):
            return {"admin_id": row["admin_id"], "username": row["username"]}
        return None

    def list_admins(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT admin_id, username, created_at FROM admins ORDER BY admin_id")]

    def update_admin(self, admin_id: int, username: Optional[str] = None,
                     password: Optional[str] = None) -> None:
        if username is not None:
            username = username.strip()
        self._validate_admin_credentials(username, password)
        try:
            with self._conn() as c:
                if not c.execute("SELECT 1 FROM admins WHERE admin_id=?", (admin_id,)).fetchone():
                    raise ValueError("Administrator not found.")
                if username is not None:
                    c.execute("UPDATE admins SET username=? WHERE admin_id=?", (username, admin_id))
                if password is not None:
                    c.execute("UPDATE admins SET password_hash=? WHERE admin_id=?",
                              (hash_password(password), admin_id))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Administrator '{username}' already exists.") from exc

    def delete_admin(self, admin_id: int) -> None:
        """Delete an admin; the last remaining admin can never be removed (no lock-out)."""
        with self._conn(immediate=True) as c:
            if c.execute("SELECT COUNT(*) FROM admins").fetchone()[0] <= 1:
                raise ValueError("Cannot delete the last administrator.")
            if not c.execute("DELETE FROM admins WHERE admin_id=?", (admin_id,)).rowcount:
                raise ValueError("Administrator not found.")
