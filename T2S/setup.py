# setup.py  — create venv with Python 3.10 and install requirements
import sys
import os
import subprocess
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"

TARGET_MAJOR = 3
TARGET_MINOR = 10  # ensure venv uses Python 3.10 (e.g., 3.10.9)

def run_ok(cmd):
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def find_target_python_cmd():
    """
    Return a list representing the command to invoke Python 3.10.
    Prefer the Windows py launcher (py -3.10). Fallback to common paths.
    """
    # Prefer Windows launcher
    if os.name == "nt" and which("py"):
        if run_ok(["py", f"-{TARGET_MAJOR}.{TARGET_MINOR}", "--version"]):
            return ["py", f"-{TARGET_MAJOR}.{TARGET_MINOR}"]

    # Common Windows install paths
    candidates = []
    if os.name == "nt":
        candidates += [
            r"C:\Users\{}\AppData\Local\Programs\Python\Python310\python.exe".format(os.getenv("USERNAME", "")),
            r"C:\Program Files\Python310\python.exe",
            r"C:\Program Files (x86)\Python310\python.exe",
        ]
    # MSYS/Unix names
    candidates += ["python3.10", "python3.10.exe"]

    for c in candidates:
        exe = which(c) if isinstance(c, str) else None
        if exe and run_ok([exe, "--version"]):
            return [exe]
        # If absolute path in candidates
        if isinstance(c, str) and Path(c).exists() and run_ok([c, "--version"]):
            return [c]

    return None

def ensure_correct_version(py_cmd):
    """Verify py_cmd is actually Python 3.10.x."""
    out = subprocess.check_output(py_cmd + ["-c", "import sys; print(sys.version)"], text=True).strip()
    if not out.startswith(f"{TARGET_MAJOR}.{TARGET_MINOR}."):
        raise RuntimeError(f"Found Python but not {TARGET_MAJOR}.{TARGET_MINOR}.x:\n{out}")

def find_venv_python():
    """Return venv python path handling Windows/MSYS/Unix layouts."""
    candidates = [
        VENV_DIR / "Scripts" / "python.exe",  # Windows
        VENV_DIR / "Scripts" / "python",
        VENV_DIR / "bin" / "python.exe",      # MSYS/Unix
        VENV_DIR / "bin" / "python",
        VENV_DIR / "bin" / "python3",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def install_requirements(venv_python: Path):
    if not REQ_FILE.exists():
        print("ℹ️ No requirements.txt found — skipping.")
        return

    # Read requirements and treat torch specially (optional but helpful on Windows)
    with REQ_FILE.open("r", encoding="utf-8") as f:
        pkgs = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    # split torch from others
    torch_spec = None
    rest = []
    for p in pkgs:
        name = p.split("==")[0].split(">=")[0].split("<=")[0].strip().lower()
        if name == "torch":
            torch_spec = p
        else:
            rest.append(p)

    # install non-torch packages first
    if rest:
        subprocess.run([str(venv_python), "-m", "pip", "install", *rest], check=True)

    # install torch from the official index (CPU by default)
    if torch_spec:
        cuda = os.getenv("TORCH_CUDA", "cpu").lower()  # set TORCH_CUDA=cu124 for CUDA 12.4
        if cuda not in {"cpu", "cu124"}:
            cuda = "cpu"
        index = "https://download.pytorch.org/whl/cpu" if cuda == "cpu" else "https://download.pytorch.org/whl/cu124"
        print(f"📦 Installing {torch_spec} from {index}")
        subprocess.run([str(venv_python), "-m", "pip", "install", torch_spec, "--index-url", index], check=True)

def create_and_install():
    # 0) Locate a Python 3.10 interpreter
    py_cmd = find_target_python_cmd()
    if not py_cmd:
        raise FileNotFoundError(
            "Python 3.10 not found.\n"
            "- Install Python 3.10.9 (Windows x64) from the official site.\n"
            "- Then re-run this script.\n"
        )
    ensure_correct_version(py_cmd)
    print(f"✅ Using base interpreter for venv: {' '.join(py_cmd)}")

    # 1) Create venv with that interpreter if missing
    if not VENV_DIR.exists():
        print(f"⚙️ Creating virtual environment at {VENV_DIR} with Python 3.10 ...")
        subprocess.run(py_cmd + ["-m", "venv", str(VENV_DIR)], check=True)

    # 2) Find venv python
    venv_python = find_venv_python()
    if not venv_python:
        raise FileNotFoundError(
            "Could not find the venv interpreter after creation.\n"
            f"Checked: {VENV_DIR/'Scripts'/'python.exe'}, {VENV_DIR/'bin'/'python'} ..."
        )
    print(f"🐍 Venv interpreter: {venv_python}")

    # 3) Upgrade pip
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # 4) Install requirements (with torch handling)
    install_requirements(venv_python)

    return str(venv_python)

if __name__ == "__main__":
    create_and_install()
