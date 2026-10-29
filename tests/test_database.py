"""Run with:  python -m unittest discover -s tests -v   (no webcam / dlib required)"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.database import AlreadyVotedError, Database
from src.utils import hash_password, validate_voter_id, verify_password


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.db", secret_key=b"k" * 32)
        self.db.add_voter("alice", "Alice", 8)
        self.db.add_voter("bob", "Bob", 8)
        self.c1 = self.db.add_candidate("Cand A")
        self.c2 = self.db.add_candidate("Cand B")

    def tearDown(self):
        self.tmp.cleanup()

    def test_vote_and_results(self):
        self.db.cast_vote("alice", self.c1)
        self.db.cast_vote("bob", self.c1)
        self.assertEqual(self.db.get_results()[0]["votes"], 2)

    def test_double_vote_blocked(self):
        self.db.cast_vote("alice", self.c1)
        with self.assertRaises(AlreadyVotedError):
            self.db.cast_vote("alice", self.c2)
        self.assertTrue(self.db.has_voted("alice"))
        self.assertEqual(self.db.total_votes(), 1)

    def test_unregistered_voter_and_bad_candidate(self):
        with self.assertRaises(ValueError):
            self.db.cast_vote("mallory", self.c1)
        with self.assertRaises(ValueError):
            self.db.cast_vote("alice", 999)

    def test_plain_voter_id_not_stored(self):
        self.db.cast_vote("alice", self.c1)
        row = self.db.export_votes()[0]
        self.assertNotIn("alice", str(row))

    def test_chain_detects_tampering_and_triggers_block_edits(self):
        self.db.cast_vote("alice", self.c1)
        self.db.cast_vote("bob", self.c2)
        self.assertTrue(self.db.verify_chain()[0])
        conn = sqlite3.connect(self.db.path)
        with self.assertRaises(sqlite3.DatabaseError):     # append-only trigger
            conn.execute("UPDATE votes SET candidate_id=? WHERE vote_id=1", (self.c2,))
        conn.execute("DROP TRIGGER votes_no_update")        # attacker removes the trigger...
        conn.execute("UPDATE votes SET candidate_id=? WHERE vote_id=1", (self.c2,))
        conn.commit()
        conn.close()
        self.assertFalse(self.db.verify_chain()[0])          # ...but the chain still catches it

    def test_candidate_with_votes_is_deactivated_not_deleted(self):
        self.db.cast_vote("alice", self.c1)
        self.assertEqual(self.db.remove_candidate(self.c1), "deactivated")
        self.assertEqual(self.db.remove_candidate(self.c2), "deleted")

    # ---- administrator CRUD -------------------------------------------------
    def test_admin_crud(self):
        self.assertFalse(self.db.has_admins())
        a1 = self.db.create_admin("root_user", "correct horse")
        a2 = self.db.create_admin("second.admin", "another pass1")
        self.assertTrue(self.db.has_admins())
        self.assertEqual([a["username"] for a in self.db.list_admins()], ["root_user", "second.admin"])
        self.assertNotIn("password_hash", self.db.list_admins()[0])        # never exposed to the UI

        # Read / authenticate
        self.assertEqual(self.db.authenticate_admin("root_user", "correct horse")["admin_id"], a1)
        self.assertEqual(self.db.authenticate_admin("ROOT_USER", "correct horse")["admin_id"], a1)  # case-insens.
        self.assertIsNone(self.db.authenticate_admin("root_user", "wrong"))
        self.assertIsNone(self.db.authenticate_admin("nobody", "correct horse"))

        # Update: password + rename
        self.db.update_admin(a2, password="brand new pass")
        self.assertIsNone(self.db.authenticate_admin("second.admin", "another pass1"))
        self.assertIsNotNone(self.db.authenticate_admin("second.admin", "brand new pass"))
        self.db.update_admin(a2, username="renamed_admin")
        self.assertIsNotNone(self.db.authenticate_admin("renamed_admin", "brand new pass"))

        # Delete; last admin is protected
        self.db.delete_admin(a2)
        self.assertEqual(len(self.db.list_admins()), 1)
        with self.assertRaises(ValueError):
            self.db.delete_admin(a1)

    def test_admin_validation(self):
        with self.assertRaises(ValueError):
            self.db.create_admin("ab", "long enough pw")          # username too short
        with self.assertRaises(ValueError):
            self.db.create_admin("good_name", "short")            # password too short
        self.db.create_admin("good_name", "long enough pw")
        with self.assertRaises(ValueError):
            self.db.create_admin("GOOD_NAME", "long enough pw")   # duplicate (case-insensitive)
        other = self.db.create_admin("other_name", "long enough pw")
        with self.assertRaises(ValueError):
            self.db.update_admin(other, username="good_name")     # rename collision
        with self.assertRaises(ValueError):
            self.db.update_admin(999, password="long enough pw")  # unknown admin

    def test_legacy_single_password_is_migrated(self):
        path = Path(self.tmp.name) / "legacy.db"
        db = Database(path, secret_key=b"k" * 32)
        db.set_setting("admin_password", hash_password("legacy password"))
        db2 = Database(path, secret_key=b"k" * 32)                # re-open => migration runs
        self.assertIsNotNone(db2.authenticate_admin("admin", "legacy password"))
        self.assertEqual(db2.get_setting("admin_password"), "")

    # ---- candidate update ----------------------------------------------------
    def test_candidate_update(self):
        self.db.update_candidate(self.c1, name="Cand Alpha", party="Blue")
        c = next(x for x in self.db.list_candidates() if x["candidate_id"] == self.c1)
        self.assertEqual((c["name"], c["party"]), ("Cand Alpha", "Blue"))
        with self.assertRaises(ValueError):
            self.db.update_candidate(self.c1, name="Cand B")      # duplicate name
        with self.assertRaises(ValueError):
            self.db.update_candidate(self.c1, name="  ")
        with self.assertRaises(ValueError):
            self.db.update_candidate(999, name="X")

    def test_candidate_deactivate_and_reactivate_keeps_chain_valid(self):
        self.db.cast_vote("alice", self.c1)
        self.db.update_candidate(self.c1, name="Renamed")          # rename after votes
        self.db.update_candidate(self.c1, active=False)
        self.assertNotIn(self.c1, [c["candidate_id"] for c in self.db.list_candidates()])
        with self.assertRaises(ValueError):                         # inactive => cannot be voted for
            self.db.cast_vote("bob", self.c1)
        self.db.update_candidate(self.c1, active=True)
        self.assertTrue(self.db.verify_chain()[0])
        self.assertEqual(self.db.get_results()[0]["votes"], 1)

    def test_helpers(self):
        self.assertTrue(validate_voter_id("user_01"))
        self.assertFalse(validate_voter_id("../etc"))
        self.assertTrue(verify_password("x", hash_password("x")))


if __name__ == "__main__":
    unittest.main()
