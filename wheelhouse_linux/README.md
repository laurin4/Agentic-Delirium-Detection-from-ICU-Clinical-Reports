# Linux wheelhouse (offline / air-gapped installs)

This directory contains **manylinux x86_64** wheels (and `py3-none-any` packages) suitable for typical **Ubuntu on amd64**. It includes the main `requirements.txt` stack plus **pytest** and its dependencies.

## Install pytest only (from repo root)

```bash
python3 -m pip install --no-index --find-links=wheelhouse_linux pytest
```

## Install project dependencies + pytest (offline)

```bash
python3 -m pip install --no-index --find-links=wheelhouse_linux -r requirements.txt pytest
```

## Notes

- Wheels tagged **`cp312`** require **Python 3.12** on Ubuntu. If the server uses **3.10** or **3.11**, regenerate the stack for that interpreter (see `RUNBOOK.md` or your internal docs) or add matching wheels with `pip download … --python-version 310`.
- **`py3-none-any`** wheels (e.g. `pytest`, `pluggy`, `packaging`) install on any supported Python version on Linux.
- Refresh pytest in this folder from a machine with network access, for example:

  ```bash
  pip download pytest -d wheelhouse_linux \
    --platform manylinux_2_17_x86_64 \
    --python-version 311 \
    --implementation cp \
    --abi cp311 \
    --only-binary=:all:
  ```
