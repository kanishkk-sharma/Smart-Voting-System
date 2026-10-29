# 🗳️ AI-Based Smart Voting System

Real-time face-recognition authentication + secure one-person-one-vote workflow.
Python · OpenCV · `face_recognition` (dlib ResNet, 128-d embeddings) · SQLite · Streamlit.

## Pipeline

```
Registration                         Voting
────────────                         ──────
webcam / photos                      live webcam frame
   │ quality gate (1 face, size,        │ detect (HOG, downscaled)
   │ sharpness)                         │ encode (128-d)
   │ duplicate-face check               │ match vs all voters (per-voter best distance,
   ▼                                    │ ambiguity rejection) → identity + confidence
dataset/<id>/*.jpg                      │ N consecutive frames must agree
encodings/encodings.npz                 │ blink liveness check (anti-photo)
voters table (SQLite)                   │ eligibility: not already voted
                                        ▼
                                     ballot → atomic vote insert (UNIQUE voter hash)
                                        → hash-chained log → receipt + UI feedback
```

## Project structure

```
smart_voting_system/
├── dataset/                 per-voter face images (dataset/<voter_id>/…)
├── encodings/               encodings.npz (128-d vectors, no pickle)
├── models/                  .secret_key (auto-generated HMAC key)
├── logs/                    rotating app log
├── src/
│   ├── face_recognition_module.py   detect / encode / match / liveness (EAR)
│   ├── voter_registration.py        add / update / delete / rebuild
│   ├── voting_system.py             AuthSession state machine + vote casting
│   ├── database.py                  SQLite + hash-chained vote log
│   └── utils.py                     config, hashing, camera, drawing
├── tests/test_database.py
├── app.py                   Streamlit UI (voting booth + admin panel)
├── requirements.txt
└── README.md
```

## Installation

Python **3.9+** and a webcam are required.

```powershell
# Windows (PowerShell) - run line by line
cd smart_voting_system
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install face_recognition --no-deps
python -c "import face_recognition, cv2; print('OK')"
```

```bash
# Linux / macOS
cd smart_voting_system
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install face_recognition --no-deps
```

`requirements.txt` uses **`dlib-bin`** (a prebuilt wheel), so no C++ compiler or CMake is needed. `face_recognition` is installed
separately with `--no-deps` because its metadata asks for plain `dlib`, which would trigger a source build that fails on
Windows without Visual Studio.

**Python version:** use **3.11 or 3.12** if you can. If `dlib-bin` reports "No matching distribution found" (common on
the newest Python, e.g. 3.13), install Python 3.12, delete `.venv`, and repeat with `py -3.12 -m venv .venv`.

**Fallback (compile dlib yourself):** install Visual Studio Build Tools ("Desktop development with C++"), then
`pip install cmake dlib` followed by `pip install face_recognition --no-deps`. Linux: `sudo apt install build-essential cmake`; macOS: `brew install cmake`.

If you see `ModuleNotFoundError: pkg_resources`, run `pip install "setuptools<81"` (already pinned in `requirements.txt`).

## Run

```bash
streamlit run app.py
```

Open the printed URL (usually http://localhost:8501) and allow camera access.

### First-time setup (≈2 minutes)
1. Sidebar → **Admin Panel** → create the first administrator (username + password, min. 8 characters).
2. **Candidates** tab → add candidates (you can also edit or remove them later).
3. **Voters** tab → *Register new voter*: enter ID + name, choose **Webcam** (guided capture of 8 images with
   pose prompts) or **Upload** (≥5 clear photos).
4. Sidebar → **Voting Booth** → *Start face verification* → blink when asked → pick a candidate → confirm.

### Live feedback messages
`Face detected` · `Recognized as Alice - Confidence: 93%` · `Vote recorded successfully` ·
`Already voted` · `Access Denied`, with green/blue/orange/red face boxes and a status banner on the video.

## Admin panel – what you can manage (full CRUD)

| Area | Create | Read | Update | Delete |
|------|--------|------|--------|--------|
| **Administrators** (🔑 Admins tab) | add admin (username + password) | list admins | rename, change password | delete (never yourself, never the last admin) |
| **Candidates** (🧑‍💼 tab) | add | table with vote counts | edit name / party / active flag | remove (deactivated instead if it already has votes) |
| **Voters** (👥 tab) | register via webcam/upload | table incl. voted status | rename, add/replace face images | delete (blocked once the voter has voted) |

Every admin action is recorded in the audit log together with the acting administrator's username.
Databases created by an earlier version with a single shared password are migrated automatically to an `admin` account
with the same password.

## Security design

| Threat | Mitigation |
|--------|-----------|
| Double voting | `UNIQUE(voter_hash)` in SQLite + atomic `BEGIN IMMEDIATE` transaction (race-safe) |
| Same person enrolling twice | Duplicate-face check at registration; voters who already voted can't be deleted/re-enrolled |
| Voter ID exposure in log | Log holds only `HMAC-SHA256(secret_key, voter_id)` – key in `models/.secret_key` |
| Vote tampering | Append-only triggers + HMAC hash chain (`Verify vote-log integrity` in admin) |
| Photo/screen spoofing | Blink liveness check (eye-aspect-ratio); multiple faces rejected |
| Misidentification | Distance + confidence thresholds, multi-frame agreement, ambiguity rejection between look-alike voters |
| Admin access | Per-admin accounts, PBKDF2-SHA256 (200k iterations) hashes, failed-login delay, audit log of who did what |
| Path traversal | Strict voter-ID regex (`[A-Za-z0-9_-]{3,32}`) |
| Encoding file attacks | Stored as `.npz` loaded with `allow_pickle=False` |

**Ballot secrecy note:** the vote row links `HMAC(voter_id)` → candidate (as the spec requires), so anyone holding
both the database **and** `models/.secret_key` could re-link votes to voters. Protect/back up the key separately and
restrict file access on the host.

## Tuning (`src/utils.py → Config`, or Admin → Settings)

* `MATCH_TOLERANCE` (default 0.45): lower is stricter (fewer false accepts, more false rejects).
* `REQUIRED_CONSECUTIVE`, `DENY_AFTER_FRAMES`, `SCAN_TIMEOUT_S`, `BALLOT_TIMEOUT_S`.
* Disable liveness in Settings for testing, or if the camera frame rate is very low.

## Tests

```bash
python -m unittest discover -s tests -v     # DB, double-vote, tamper-detection, hashing (no camera needed)
```

## Troubleshooting

* **Cannot open camera** – close other apps using it; change *Camera index* in Admin → Settings.
* **Slow video** – lower `DETECT_SCALE` (e.g. 0.4) or close other CPU-heavy programs.
* **Face never recognized** – re-register in good, even lighting with varied poses; loosen tolerance to ≤0.50.
* **Encodings lost/corrupt** – Admin → Settings → *Rebuild encodings from dataset images*.
* **Reset everything** – stop the app and delete `voting.db*`, `dataset/*/`, `encodings/*.npz`, `models/.secret_key`.

## Limitations (please read)

This is a strong demonstration/prototype and suitable for classrooms, clubs and internal polls. **It is not a
certified system for legally binding public elections.** Blink detection stops printed photos and most static
screens but not sophisticated attacks (video replay, 3-D masks, deepfakes); a hardened deployment would add
depth-/IR-based anti-spoofing, a trusted kiosk environment, HTTPS, encrypted-at-rest storage, an independent
audit process and legal/privacy review for biometric data (e.g. GDPR/BIPA consent).
#   S m a r t - V o t i n g - S y s t e m  
 