"""
app.py - Streamlit front-end for the AI Smart Voting System.

Run with:   streamlit run app.py

Pages
-----
* Voting Booth : live webcam -> face authentication -> ballot -> receipt
* Admin Panel  : results, voter management, candidates, audit log, settings
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Voting System", page_icon="🗳️", layout="wide")

try:
    from src.database import AlreadyVotedError, Database
    from src.face_recognition_module import FaceEngine
    from src.utils import (CameraError, Config, GREEN, RED, bgr_to_rgb, draw_face_box,
                           ensure_directories, open_camera, setup_logging)
    from src.voter_registration import VoterRegistrar
    from src.voting_system import AuthState, VotingSystem
except ImportError as exc:
    st.error(f"Missing dependency: {exc}\n\nInstall with `pip install -r requirements.txt` "
             "(see README for dlib/face_recognition notes).")
    st.stop()

GUIDE = ["Look straight at the camera", "Turn your head slightly LEFT",
         "Turn your head slightly RIGHT", "Tilt your chin UP a little",
         "Tilt your chin DOWN a little", "Smile", "Neutral expression", "Look straight again"]


# --------------------------------------------------------------------------- #
# Shared services (created once per server process)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading face-recognition engine...")
def get_services():
    ensure_directories()
    setup_logging()
    db = Database()
    engine = FaceEngine(tolerance=float(db.get_setting("match_tolerance", str(Config.MATCH_TOLERANCE))))
    return db, engine, VoterRegistrar(db, engine), VotingSystem(db, engine)


db, engine, registrar, voting = get_services()


def init_state() -> None:
    defaults = dict(scanning=False, stage="idle", voter_id=None, voter_name=None, confidence=0.0,
                    ballot_started=0.0, last_result=None, receipt=None, admin_ok=False,
                    flash=None, reg_nonce=0, admin_id=None, admin_user=None)
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()


def camera_index() -> int:
    return int(db.get_setting("camera_index", "0"))


def liveness_on() -> bool:
    return db.get_setting("require_liveness", "1") == "1"


def show_msg(placeholder, level: str, text: str) -> None:
    {"success": placeholder.success, "error": placeholder.error,
     "warning": placeholder.warning}.get(level, placeholder.info)(text)


def flash(level: str, text: str) -> None:
    st.session_state.flash = (level, text)


def show_flash() -> None:
    if st.session_state.flash:
        level, text = st.session_state.flash
        show_msg(st, level, text)
        st.session_state.flash = None


def reset_voting() -> None:
    st.session_state.update(scanning=False, stage="idle", voter_id=None, voter_name=None,
                            confidence=0.0, receipt=None, last_result=None)


# =========================================================================== #
# PAGE 1 - VOTING BOOTH
# =========================================================================== #
def start_scan() -> None:
    st.session_state.update(scanning=True, last_result=None)


def stop_scan() -> None:
    st.session_state.scanning = False


def run_scan(video_ph, status_ph, progress_ph) -> None:
    """Live loop: webcam -> AuthSession -> annotated video + status messages."""
    session = voting.new_auth_session(require_liveness=liveness_on())
    finished = False
    cap = None
    try:
        cap = open_camera(camera_index())
        t0 = time.time()
        while st.session_state.scanning:
            ok, frame = cap.read()
            if not ok:
                raise CameraError("Camera stopped returning frames.")
            frame = cv2.flip(frame, 1)                      # mirror view feels natural
            res = session.process(frame)
            video_ph.image(bgr_to_rgb(res.annotated), channels="RGB", use_container_width=True)
            show_msg(status_ph, res.level, res.message)
            progress_ph.progress(res.progress)

            if res.state == AuthState.AUTHENTICATED:
                st.session_state.update(stage="ballot", voter_id=res.voter_id, voter_name=res.name,
                                        confidence=res.confidence, ballot_started=time.time(),
                                        scanning=False)
                finished = True
            elif res.state in (AuthState.ALREADY_VOTED, AuthState.DENIED):
                st.session_state.update(last_result=(res.level, res.message), scanning=False)
                finished = True
            elif time.time() - t0 > Config.SCAN_TIMEOUT_S:
                st.session_state.update(
                    last_result=("warning", "Verification timed out. Please try again."), scanning=False)
                finished = True
            if finished:
                time.sleep(1.2)                             # let the voter see the final frame
                break
            time.sleep(0.01)
    except CameraError as exc:
        st.session_state.update(scanning=False, last_result=("error", str(exc)))
        finished = True
    finally:
        if cap is not None:
            cap.release()
    if finished:
        st.rerun()


def page_vote() -> None:
    st.title("🗳️ Smart Voting Booth")
    stage = st.session_state.stage

    if stage == "ballot":
        return page_ballot()
    if stage == "done":
        return page_done()

    if engine.voter_count == 0:
        st.warning("No voters are registered yet. An administrator must register voters first.")
    if not voting.candidates():
        st.warning("No candidates have been added yet (Admin Panel → Candidates).")

    col_video, col_status = st.columns([3, 2])
    with col_status:
        st.subheader("Verification")
        st.markdown("1. Stand in front of the camera\n2. Hold still while your face is verified\n"
                    + ("3. Blink once when asked\n" if liveness_on() else "")
                    + "4. Choose your candidate and confirm")
        status_ph, progress_ph = st.empty(), st.empty()
        if st.session_state.scanning:
            st.button("⏹ Cancel", on_click=stop_scan)
        else:
            st.button("▶ Start face verification", on_click=start_scan, type="primary",
                      disabled=engine.voter_count == 0 or not voting.candidates())
            if st.session_state.last_result:
                show_msg(status_ph, *st.session_state.last_result)
    video_ph = col_video.empty()
    if not st.session_state.scanning:
        video_ph.info("Camera is off. Press **Start face verification**.")
    else:
        run_scan(video_ph, status_ph, progress_ph)


def page_ballot() -> None:
    elapsed = time.time() - st.session_state.ballot_started
    if elapsed > Config.BALLOT_TIMEOUT_S:
        reset_voting()
        st.session_state.last_result = ("warning", "Ballot session expired. Please verify again.")
        st.rerun()

    st.success(f"Recognized as {st.session_state.voter_name} - Confidence: "
               f"{st.session_state.confidence:.0%}. Identity verified.")
    st.caption(f"Please vote within {int(Config.BALLOT_TIMEOUT_S - elapsed)} seconds.")

    cands = voting.candidates()
    labels = {c["candidate_id"]: c["name"] + (f"  ({c['party']})" if c["party"] else "") for c in cands}
    choice = st.radio("Select your candidate", list(labels), format_func=labels.get, index=None)
    confirm = st.checkbox("I confirm my choice. This cannot be changed after submission.")

    c1, c2 = st.columns(2)
    if c1.button("✅ Cast vote", type="primary", disabled=not (choice and confirm)):
        if time.time() - st.session_state.ballot_started > Config.BALLOT_TIMEOUT_S:
            reset_voting()
            st.rerun()
        try:
            st.session_state.receipt = voting.cast_vote(st.session_state.voter_id, choice)
            st.session_state.stage = "done"
        except AlreadyVotedError:
            reset_voting()
            st.session_state.last_result = ("error", "Already voted - Access Denied.")
        except ValueError as exc:
            st.error(str(exc))
            return
        st.rerun()
    if c2.button("Cancel"):
        reset_voting()
        st.rerun()


def page_done() -> None:
    r = st.session_state.receipt
    st.success("✅ Vote recorded successfully. Thank you for voting!")
    st.balloons()
    st.markdown(f"**Receipt** - Vote #{r.vote_id} · {r.timestamp}")
    st.code(r.entry_hash, language=None)
    st.caption("Keep this receipt code if you want to verify later that your vote is in the log. "
               "It does not reveal who you voted for.")
    if st.button("Next voter", type="primary"):
        reset_voting()
        st.rerun()


# =========================================================================== #
# PAGE 2 - ADMIN PANEL
# =========================================================================== #
def _start_admin_session(admin: dict) -> None:
    st.session_state.update(admin_ok=True, admin_id=admin["admin_id"], admin_user=admin["username"])


def admin_login() -> bool:
    if st.session_state.admin_ok:
        return True
    st.title("🔐 Admin Panel")
    if not db.has_admins():
        st.info("First run: create the first administrator account.")
        user = st.text_input("Username")
        p1 = st.text_input("Password (min. 8 characters)", type="password")
        p2 = st.text_input("Repeat password", type="password")
        if st.button("Create administrator", type="primary"):
            if p1 != p2:
                st.error("Passwords do not match.")
            else:
                try:
                    aid = db.create_admin(user, p1)
                    db.log_event("ADMIN_CREATED", f"user={user.strip()} (first-run setup)")
                    _start_admin_session({"admin_id": aid, "username": user.strip()})
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        return False
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Log in", type="primary"):
        admin = db.authenticate_admin(user, pw)
        if admin:
            db.log_event("ADMIN_LOGIN", f"user={admin['username']}")
            _start_admin_session(admin)
            st.rerun()
        else:
            db.log_event("ADMIN_LOGIN_FAILED", f"user={user.strip()}")
            time.sleep(1.0)                                  # slow down brute force
            st.error("Wrong username or password.")
    return False


def decode_uploads(files) -> list:
    frames = []
    for f in files or []:
        img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def capture_enrollment_frames(target: int, video_ph, status_ph, progress_ph, timeout: float = 90.0) -> list:
    """Capture `target` quality-checked frames from the webcam with live guidance."""
    frames, last_t, start = [], 0.0, time.time()
    cap = open_camera(camera_index())
    try:
        while len(frames) < target and time.time() - start < timeout:
            ok, frame = cap.read()
            if not ok:
                raise CameraError("Camera stopped returning frames.")
            frame = cv2.flip(frame, 1)
            check = registrar.assess_frame(frame)
            vis = frame.copy()
            if check.box:
                draw_face_box(vis, check.box, "OK" if check.ok else check.message,
                              GREEN if check.ok else RED)
            video_ph.image(bgr_to_rgb(vis), channels="RGB", use_container_width=True)
            now = time.time()
            if check.ok and now - last_t >= 0.9:             # spaced captures = pose variety
                frames.append(frame.copy())
                last_t = now
            if check.ok:
                status_ph.info(f"📸 {len(frames)}/{target} - {GUIDE[len(frames) % len(GUIDE)]}")
            else:
                status_ph.warning(check.message)
            progress_ph.progress(min(1.0, len(frames) / target))
    finally:
        cap.release()
    return frames


def get_frames(source: str, files, target: int, ph) -> list:
    if source == "Webcam":
        return capture_enrollment_frames(target, *ph)
    return decode_uploads(files)


def tab_results() -> None:
    results = voting.results()
    cast, registered = voting.turnout()
    m1, m2, m3 = st.columns(3)
    m1.metric("Votes cast", cast)
    m2.metric("Registered voters", registered)
    m3.metric("Turnout", f"{(cast / registered * 100) if registered else 0:.0f}%")
    if results:
        df = pd.DataFrame(results)[["name", "party", "votes"]]
        st.bar_chart(df.set_index("name")["votes"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No candidates yet.")
    if st.button("🔎 Verify vote-log integrity"):
        ok, msg = voting.verify_integrity()
        (st.success if ok else st.error)(msg)


def tab_voters() -> None:
    show_flash()
    voters = db.list_voters()
    if voters:
        df = pd.DataFrame(voters).rename(columns={"has_voted": "voted"})
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No voters registered.")

    n = st.session_state.reg_nonce
    with st.expander("➕ Register new voter", expanded=not voters):
        vid = st.text_input("Voter ID (letters/digits/_/-)", key=f"rid{n}")
        name = st.text_input("Full name", key=f"rname{n}")
        src = st.radio("Face images from", ["Webcam", "Upload"], horizontal=True, key=f"rsrc{n}")
        files = st.file_uploader("Upload 5+ clear photos", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key=f"rfiles{n}") if src == "Upload" else None
        ph = (st.empty(), st.empty(), st.empty())
        if st.button("Register voter", type="primary"):
            try:
                frames = get_frames(src, files, Config.TARGET_REG_IMAGES, ph)
                rep = registrar.register(vid, name, frames)
                flash("success" if rep.success else "error", rep.message)
                if rep.success:
                    st.session_state.reg_nonce += 1
            except CameraError as exc:
                flash("error", str(exc))
            st.rerun()

    if voters:
        ids = [v["voter_id"] for v in voters]
        with st.expander("✏️ Update voter"):
            uid = st.selectbox("Voter", ids, key="upd_id")
            cur = next(v for v in voters if v["voter_id"] == uid)
            new_name = st.text_input("Name", value=cur["name"], key=f"uname_{uid}")
            add_imgs = st.checkbox("Capture/upload new face images")
            replace = st.checkbox("Replace existing images (otherwise add)", disabled=not add_imgs)
            usrc = st.radio("Images from", ["Webcam", "Upload"], horizontal=True, key="usrc",
                            disabled=not add_imgs)
            ufiles = st.file_uploader("Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                      key="ufiles") if add_imgs and usrc == "Upload" else None
            uph = (st.empty(), st.empty(), st.empty())
            if st.button("Save changes"):
                try:
                    frames = get_frames(usrc, ufiles, Config.TARGET_REG_IMAGES, uph) if add_imgs else None
                    rep = registrar.update(uid, name=new_name, frames=frames, replace_images=replace)
                    flash("success" if rep.success else "error", rep.message)
                except CameraError as exc:
                    flash("error", str(exc))
                st.rerun()

        with st.expander("🗑️ Delete voter"):
            did = st.selectbox("Voter", ids, key="del_id")
            typed = st.text_input(f"Type the voter ID ({did}) to confirm")
            if st.button("Delete voter", disabled=typed != did):
                rep = registrar.delete(did)
                flash("success" if rep.success else "error", rep.message)
                st.rerun()


def tab_candidates() -> None:
    show_flash()
    cands = db.list_candidates(active_only=False)
    votes = {r["candidate_id"]: r["votes"] for r in voting.results()}
    if cands:
        df = pd.DataFrame(cands)
        df["votes"] = df["candidate_id"].map(lambda i: votes.get(i, 0))
        df["active"] = df["active"].astype(bool)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No candidates yet.")

    with st.expander("➕ Add candidate", expanded=not cands):
        with st.form("add_cand", clear_on_submit=True):
            c1, c2 = st.columns(2)
            cname, cparty = c1.text_input("Candidate name"), c2.text_input("Party (optional)")
            if st.form_submit_button("Add candidate"):
                try:
                    db.add_candidate(cname, cparty)
                    db.log_event("CANDIDATE_ADDED", f"{st.session_state.admin_user}: {cname.strip()}")
                    flash("success", f"Added candidate '{cname.strip()}'.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    if not cands:
        return

    by_id = {c["candidate_id"]: c for c in cands}
    with st.expander("✏️ Edit candidate"):
        cid = st.selectbox("Candidate", list(by_id), format_func=lambda i: by_id[i]["name"], key="cand_edit")
        cur = by_id[cid]
        if votes.get(cid, 0):
            st.warning("Votes are already recorded for this candidate. Renaming changes how past votes are displayed.")
        n_name = st.text_input("Name", value=cur["name"], key=f"cn_{cid}")
        n_party = st.text_input("Party", value=cur["party"], key=f"cp_{cid}")
        n_active = st.checkbox("Active (shown on the ballot)", value=bool(cur["active"]), key=f"ca_{cid}")
        if st.button("Save candidate"):
            try:
                db.update_candidate(cid, name=n_name, party=n_party, active=n_active)
                db.log_event("CANDIDATE_UPDATED", f"{st.session_state.admin_user}: id={cid}")
                flash("success", "Candidate updated.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    active = [c for c in cands if c["active"]]
    if active:
        with st.expander("🗑️ Remove candidate"):
            rid = st.selectbox("Candidate", [c["candidate_id"] for c in active],
                               format_func=lambda i: by_id[i]["name"], key="cand_del")
            st.caption("Candidates who already have votes are deactivated instead of deleted, "
                       "so the vote history stays valid.")
            sure = st.checkbox("I understand", key="cand_del_ok")
            if st.button("Remove candidate", disabled=not sure):
                outcome = db.remove_candidate(rid)
                db.log_event("CANDIDATE_REMOVED",
                             f"{st.session_state.admin_user}: {by_id[rid]['name']} ({outcome})")
                flash("success", f"Candidate {outcome}.")
                st.rerun()


def tab_audit() -> None:
    votes = db.export_votes()
    if votes:
        st.download_button("⬇ Download vote log (CSV)", pd.DataFrame(votes).to_csv(index=False),
                           "vote_log.csv", "text/csv")
    ev = db.recent_audit()
    st.dataframe(pd.DataFrame(ev) if ev else pd.DataFrame(columns=["timestamp", "event", "details"]),
                 use_container_width=True, hide_index=True)


def tab_settings() -> None:
    liveness = st.checkbox("Require blink liveness check", value=liveness_on())
    tol = st.slider("Match tolerance (lower = stricter)", 0.35, 0.55,
                    float(db.get_setting("match_tolerance", str(Config.MATCH_TOLERANCE))), 0.01)
    cam = st.number_input("Camera index", 0, 10, camera_index())
    if st.button("Save settings", type="primary"):
        db.set_setting("require_liveness", "1" if liveness else "0")
        db.set_setting("match_tolerance", str(tol))
        db.set_setting("camera_index", str(int(cam)))
        engine.tolerance = tol
        db.log_event("SETTINGS_CHANGED", f"liveness={liveness} tol={tol} cam={cam}")
        st.success("Saved.")
    st.divider()
    if st.button("♻ Rebuild encodings from dataset images"):
        with st.spinner("Re-encoding images..."):
            st.success(registrar.rebuild_encodings().message)


def tab_admins() -> None:
    show_flash()
    me, me_name = st.session_state.admin_id, st.session_state.admin_user
    admins = db.list_admins()
    st.dataframe(pd.DataFrame(admins), use_container_width=True, hide_index=True)
    st.caption(f"Logged in as **{me_name}**.")
    by_id = {a["admin_id"]: a for a in admins}

    with st.expander("➕ Create administrator"):
        with st.form("create_admin", clear_on_submit=True):
            u = st.text_input("Username")
            p1 = st.text_input("Password (min. 8 characters)", type="password")
            p2 = st.text_input("Repeat password", type="password")
            if st.form_submit_button("Create"):
                if p1 != p2:
                    st.error("Passwords do not match.")
                else:
                    try:
                        db.create_admin(u, p1)
                        db.log_event("ADMIN_CREATED", f"{me_name} created {u.strip()}")
                        flash("success", f"Administrator '{u.strip()}' created.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

    with st.expander("✏️ Edit administrator (rename / change password)"):
        aid = st.selectbox("Administrator", list(by_id), format_func=lambda i: by_id[i]["username"],
                           key="adm_edit")
        cur = by_id[aid]
        new_user = st.text_input("Username", value=cur["username"], key=f"adm_u_{aid}")
        new_pw = st.text_input("New password (leave blank to keep current)", type="password",
                               key=f"adm_p_{aid}")
        if st.button("Save administrator"):
            try:
                renamed = new_user.strip() != cur["username"]
                db.update_admin(aid, username=new_user if renamed else None, password=new_pw or None)
                if aid == me and renamed:
                    st.session_state.admin_user = new_user.strip()
                db.log_event("ADMIN_UPDATED", f"{me_name} updated {cur['username']}"
                             f"{' (password changed)' if new_pw else ''}{' (renamed)' if renamed else ''}")
                flash("success", "Administrator updated.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    with st.expander("🗑️ Delete administrator"):
        others = [i for i in by_id if i != me]
        if not others:
            st.info("There is no other administrator to delete. You cannot delete your own account "
                    "while logged in, and the last administrator can never be removed.")
        else:
            did = st.selectbox("Administrator", others, format_func=lambda i: by_id[i]["username"],
                               key="adm_del")
            typed = st.text_input(f"Type the username ({by_id[did]['username']}) to confirm", key="adm_del_t")
            if st.button("Delete administrator", disabled=typed != by_id[did]["username"]):
                try:
                    db.delete_admin(did)
                    db.log_event("ADMIN_DELETED", f"{me_name} deleted {by_id[did]['username']}")
                    flash("success", "Administrator deleted.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


def page_admin() -> None:
    st.session_state.scanning = False
    if not admin_login():
        return
    st.title("🛠️ Admin Panel")
    tabs = st.tabs(["📊 Results", "👥 Voters", "🧑‍💼 Candidates", "🔑 Admins", "📜 Audit", "⚙️ Settings"])
    for tab, fn in zip(tabs, (tab_results, tab_voters, tab_candidates, tab_admins, tab_audit, tab_settings)):
        with tab:
            fn()
    st.sidebar.caption(f"Admin: {st.session_state.admin_user}")
    if st.sidebar.button("Log out"):
        db.log_event("ADMIN_LOGOUT", f"user={st.session_state.admin_user}")
        st.session_state.update(admin_ok=False, admin_id=None, admin_user=None)
        st.rerun()


# =========================================================================== #
# Navigation
# =========================================================================== #
page = st.sidebar.radio("Navigation", ["Voting Booth", "Admin Panel"])
voted, registered = voting.turnout()
st.sidebar.caption(f"Registered voters: {registered} · Votes cast: {voted}")
if page == "Voting Booth":
    page_vote()
else:
    page_admin()
