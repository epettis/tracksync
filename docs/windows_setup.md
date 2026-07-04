# Windows setup & usage

Installation and execution instructions for tracksync on Windows 10/11. The
main [`README.md`](../README.md) covers Linux/macOS; this document covers the
Windows-specific steps (PowerShell activation, path separators, and — most
importantly — how to install a **CUDA/GPU build of PyTorch** for scene mode).

For what the tool does and the full command reference, see the
[Usage](../README.md#usage) section of the README. The subcommands and options
are identical across platforms.

---

## 1. Prerequisites

- **Python 3.8+** (3.11 recommended; the project is developed and tested on
  3.11). Install from [python.org](https://www.python.org/downloads/windows/)
  and tick **"Add python.exe to PATH"** in the installer. Verify:

  ```powershell
  python --version
  ```

  The `py` launcher (`py -3.11 --version`) also works if you installed Python
  via the official installer.

- **Git** — required to build **LightGlue**, which the `scene` extra installs
  directly from GitHub. Install [Git for Windows](https://git-scm.com/download/win)
  and verify with `git --version`.

- **(GPU only) NVIDIA GPU + driver.** For GPU acceleration you need an NVIDIA
  card and a recent driver. No separate CUDA Toolkit install is required — the
  PyTorch CUDA wheels bundle their own runtime. Check your driver and the
  maximum CUDA version it supports:

  ```powershell
  nvidia-smi
  ```

  The **"CUDA Version"** shown in the top-right is the highest your driver
  supports; drivers are backward-compatible, so a newer driver runs older CUDA
  wheels fine.

---

## 2. Create and activate a virtual environment

From the repository root (`c:\Users\<you>\src\tracksync`):

```powershell
# Create the virtual environment (.venv is already in .gitignore)
python -m venv .venv

# Activate it (PowerShell)
.\.venv\Scripts\Activate.ps1
```

Your prompt should now be prefixed with `(.venv)`.

> **PowerShell execution policy.** If activation fails with
> *"running scripts is disabled on this system"*, allow local scripts for your
> user (one time), then activate again:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

**Other shells:**

| Shell             | Activate command                    |
| ----------------- | ----------------------------------- |
| PowerShell        | `.\.venv\Scripts\Activate.ps1`      |
| Command Prompt    | `.\.venv\Scripts\activate.bat`      |
| Git Bash          | `source .venv/Scripts/activate`     |

Deactivate any time with `deactivate`.

The rest of this guide assumes the venv is **active**. If you prefer not to
activate, prefix commands with the interpreter directly, e.g.
`.\.venv\Scripts\python.exe -m pytest`.

---

## 3. Install tracksync

Pick **one** of the two paths below.

### Option A — CPU only (core + scene on CPU)

Simplest, no GPU. Scene mode still runs, just slower on the CPU.

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[scene,dev]"
```

This installs the package in **editable** mode (source edits take effect with
no reinstall) with the scene extra (torch, timm, LightGlue, scipy) and the dev
extra (pytest). On Windows/PyPI this pulls the **CPU-only** torch wheel.

### Option B — GPU (CUDA) acceleration ✅ recommended for NVIDIA cards

The catch: `pip install -e ".[scene]"` pulls torch from **PyPI**, which on
Windows is the **CPU-only** build. To get GPU acceleration you must install the
**CUDA build of PyTorch first**, from PyTorch's own package index, and only
then install the scene extra.

```powershell
# 1. Upgrade the build tooling
python -m pip install --upgrade pip setuptools wheel

# 2. Install the CUDA build of torch FIRST (cu124 = CUDA 12.4 runtime).
#    This is a large (~2.5 GB) download.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Now install the package + scene/dev extras. Because a compatible torch is
#    already present, pip will NOT replace it with the CPU wheel.
python -m pip install -e ".[scene,dev]"
```

> **Ordering matters.** If you run step 3 before step 2, you get the CPU wheel
> and `torch.cuda.is_available()` returns `False`. Fix it by reinstalling torch
> from the CUDA index:
> `python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124`

**Which CUDA version?** `cu124` (CUDA 12.4) works on all modern NVIDIA cards and
any driver reporting CUDA 12.4 or newer in `nvidia-smi`. Newer runtimes
(`cu126`, `cu128`) exist on the same [PyTorch index](https://pytorch.org/get-started/locally/)
if you want them; just swap the `--index-url` suffix.

### Optional extras

```powershell
# OCR-based features (Catalyst mode helpers)
python -m pip install pytesseract
```

---

## 4. Verify the install

```powershell
# Core CLI works
python tracksync.py --help
#   ...or, since the console script is installed:
tracksync --help

# GPU check (only meaningful after Option B)
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Scene dependencies import cleanly
python -c "from tracksync.scene_deps import require_scene_deps; require_scene_deps(); import timm, lightglue, scipy; print('scene deps OK')"
```

A successful GPU check prints something like
`cuda: True | NVIDIA GeForce RTX 3090`.

---

## 5. Running tracksync

Once installed, use the `tracksync` command (or `python tracksync.py`). The
commands are the same as on other platforms — see the
[README Usage section](../README.md#usage) for the full reference. A few
Windows notes:

- **Paths** use backslashes and should be quoted if they contain spaces:

  ```powershell
  tracksync sync "C:\videos\lap_a.mp4" "C:\videos\lap_b.mp4" -o sync.csv
  ```

- **Scene mode is the default** and runs on the GPU automatically when a CUDA
  build of torch is installed (Option B). No flag is needed to enable the GPU.

- **Generate a comparison video:**

  ```powershell
  tracksync sync "C:\videos\lap_a.mp4" "C:\videos\lap_b.mp4" --generate-video --output-dir .\out
  ```

- **Batch-sync several videos:**

  ```powershell
  tracksync sync --all "C:\videos\lap1.mp4" "C:\videos\lap2.mp4" "C:\videos\lap3.mp4" --output-dir .\sync_output
  ```

---

## 6. Running the tests

```powershell
python -m pytest tests\ -q
```

The suite deselects `slow`-marked tests by default (see
[`pyproject.toml`](../pyproject.toml)). Scene-related tests require the `scene`
extra to be installed.

---

## 7. VSCode integration

The repo ships a [`.vscode/settings.json`](../.vscode/settings.json) that points
VSCode at `.venv` and enables the pytest test explorer. After creating the venv:

1. Install the **Python** extension (Microsoft) if you haven't.
2. Reload the window, or run **Ctrl+Shift+P → "Python: Select Interpreter"** and
   choose the interpreter at `.\.venv\Scripts\python.exe`.
3. New integrated terminals auto-activate the venv, and the **Testing** panel
   discovers the pytest suite.

Because the package is installed editable (`pip install -e`), edits to files
under `tracksync\` take effect immediately — no reinstall needed.

---

## 8. Troubleshooting (Windows-specific)

- **`torch.cuda.is_available()` is `False` after Option B.** You most likely
  installed the CPU wheel (ran the scene extra before the CUDA torch, or your
  environment already had a CPU torch). Force-reinstall from the CUDA index:

  ```powershell
  python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```

- **`Activate.ps1 cannot be loaded because running scripts is disabled`.** Set
  the execution policy for your user (see §2), or use `activate.bat` from
  Command Prompt instead.

- **LightGlue fails to install / `git` not found.** The `scene` extra builds
  LightGlue from GitHub. Install [Git for Windows](https://git-scm.com/download/win),
  reopen your terminal so `git` is on `PATH`, and retry.

- **`pip install -e '.[scene]'` message when running a command.** Scene mode
  needs the `scene` extra. Note that on Windows PowerShell you should quote the
  extra with **double** quotes: `pip install -e ".[scene]"` (single quotes are a
  bash-ism and are treated literally by PowerShell).

- **`py` / `python` not found.** Reinstall Python with **"Add python.exe to
  PATH"** checked, or launch a new terminal after installation so `PATH` is
  refreshed.

For non-Windows-specific issues (low-confidence spans, trim warnings, etc.) see
the [Troubleshooting](../README.md#troubleshooting) section of the README.
