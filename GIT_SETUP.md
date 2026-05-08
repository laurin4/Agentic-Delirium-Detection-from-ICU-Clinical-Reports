# Git setup — sync code between Mac and Ubuntu, never patient data

## 1. Purpose

- GitHub holds **code only** for syncing the workstation (Mac) and the deployment host (Ubuntu).
- Patient data, model outputs, environments, and model artifacts **stay local** on each machine.
- The `.gitignore` in this folder enforces this; the pre-commit script in `scripts/check_no_sensitive_files.py` is a second safety net.

Never commit any of these:

- `Berichte.csv`, `ICD.csv`, `ICDSC.csv`, anything else under `data/`
- `outputs/`, `reports/`, `plots/`, `*.png`, `*.pdf`, `*.parquet`, `*.jsonl`
- Environments: `Ba_venv/`, `Ba_venv_backup/`, `.venv/`, `venv/`, `env/`, `delirium_env/`
- Model files: `models/`, `models_Qwen/`, `models_Ollama/`, `wheelhouse_linux/`, `*.tar.gz`
- Secrets: `.env`, `*.env`, `*.token`, `*.key`, `*.pem`

---

## 2. First-time setup — Mac (origin of code)

From the project root (`delirium_project/`):

```bash
git init
git status                       # confirm sensitive files are ignored
git add .
git status                       # double-check what is staged
python scripts/check_no_sensitive_files.py  # safety check
git commit -m "Initial code import (no patient data)"
git branch -M main
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```

If `git status` shows files under `data/`, `outputs/`, etc., **do not commit**. Update `.gitignore` and untrack them first (see Section 7).

---

## 3. First-time setup — Ubuntu server

Treat the existing folder as **read-only** until the clone is verified.

```bash
# 1) Back up the existing project folder (data + outputs are NOT in the repo)
mv delirium_project delirium_project_backup_$(date +%Y%m%d_%H%M%S)

# 2) Clone fresh from GitHub
git clone git@github.com:<your-user>/<your-repo>.git delirium_project
cd delirium_project

# 3) Restore real data manually (from backup or secure transfer)
mkdir -p data/raw
cp /path/to/local/Berichte.csv data/raw/
cp /path/to/local/ICD.csv      data/raw/
cp /path/to/local/ICDSC.csv    data/raw/

# 4) Recreate the virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Confirm the data files are **ignored** (they should not show up in `git status`).

---

## 4. Normal workflow

**Mac (push code):**

```bash
git status
git add <changed files>
python scripts/check_no_sensitive_files.py
git commit -m "<message>"
git push
```

**Ubuntu (pull code):**

```bash
git pull
```

Re-run the pipeline after pulling if anything in `src/` changed.

---

## 5. Safety checks before every commit

```bash
git status
git diff --cached --name-only
python scripts/check_no_sensitive_files.py
```

The script exits with **code 1** and prints offending paths if staged files match risky patterns. Fix the staging set before committing.

---

## 6. If sensitive files were accidentally staged (not committed yet)

```bash
git restore --staged <file>
# or restore everything that's staged:
git reset
```

Then update `.gitignore` if the file should never be committed again.

---

## 7. If sensitive files were already tracked (committed previously)

Stop tracking them without deleting the local file:

```bash
git rm --cached <file>
git rm -r --cached data outputs           # for whole folders
git commit -m "Stop tracking sensitive paths"
git push
```

If sensitive content was already pushed, you must additionally:
1. **Rotate** any leaked secret (token / key).
2. Rewrite history with `git filter-repo` or BFG to scrub the file from prior commits.
3. Force-push (only with team coordination).

Prevention is much cheaper than rewriting history.

---

## 8. Folder placeholders

The repo keeps these empty markers so the layout is clear after a fresh clone:

```
data/.gitkeep
data/raw/.gitkeep
outputs/.gitkeep
```

`.gitignore` whitelists only the `.gitkeep` files inside otherwise-ignored folders. **Real data still cannot be committed.**

---

## 9. Quick reference

| What | Where |
|------|-------|
| Ignore rules | `.gitignore` |
| Pre-commit safety | `scripts/check_no_sensitive_files.py` |
| Setup notes | `GIT_SETUP.md` (this file) |
| Project usage | `README.md`, `RUNBOOK.md` |
