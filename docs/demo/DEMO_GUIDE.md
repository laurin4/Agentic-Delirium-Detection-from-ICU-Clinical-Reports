# Delirium Pipeline Demo — Guide

Hemorrhage-style step-by-step walkthrough for your thesis slides.

All exports are **anonymized** (`Beispiel-Fall A` / `B` — no patient IDs).

**Default case pairing:** True Positive (TP) + False Negative (FN).

---

## Recommended: `.txt` for your own figures

```bash
cd delirium_project
source Ba_venv/bin/activate

# 1. Regenerate snapshots (TP + FN; FN prefers Patient_0057 / Patient_0075)
python -m src.analysis.demo_delirium_case --snapshot-positive --snapshot-false-negative

# 2. Export walkthrough text
python -m src.analysis.demo_delirium_case --txt
```

**Outputs:**

| File | Content |
|------|---------|
| `outputs/demo/delirium_demo_fall_a_walkthrough.txt` | True positive (TP) |
| `outputs/demo/delirium_demo_fall_b_walkthrough.txt` | False negative (FN) |
| `outputs/demo/delirium_pipeline_demo_walkthrough.txt` | Both cases combined |

Structure mirrors the hemorrhage demo (v2 trace with prompts + raw JSON):

1. Original clinical reports  
2. Rule-based evidence extraction  
3. Agent 1 prompt  
4. Agent 1 raw LLM response → parsed JSON  
5. Agent 2 prompt  
6. Agent 2 raw LLM response → parsed JSON  
7. Clinical guardrails → klasse  
8. Validation label + final classification box  

On the **server**, capture real LLM responses once:

```bash
python -m src.analysis.demo_delirium_case --snapshot-positive --live
python -m src.analysis.demo_delirium_case --snapshot-false-negative --live
```

Copy the resulting `data/demo/*.json` to your laptop for offline `--both` / `--txt` replay.

---

## Pick demo cases

List top candidates (on server with full validation data):

```bash
python -m src.analysis.demo_delirium_case --list-positive-candidates
python -m src.analysis.demo_delirium_case --list-false-negative-candidates
```

Force a specific FN report (e.g. Patient 0057 or 0075 from your error analysis):

```bash
python -m src.analysis.demo_delirium_case --snapshot-false-negative \
  --validation-report-id Patient_0057_Report_0001

# Or Patient 0075:
python -m src.analysis.demo_delirium_case --snapshot-false-negative \
  --validation-report-id Patient_0075_Report_0001

python -m src.analysis.demo_delirium_case --txt
```

Positive auto-pick prefers: **short reports**, **`Delir` in Diagnosen**, **`direct_delir_positive`**, few snippets.

FN auto-pick prefers: **Patient_0057** and **Patient_0075**, then other verified FN reports (`klasse=0`, manual GT=1).

---

## Other commands

```bash
python -m src.analysis.demo_delirium_case --both          # live terminal walkthrough (TP + FN)
python -m src.analysis.demo_delirium_case --html          # browser preview only
```

Legacy flag names still work: `--snapshot-negative`, `--negative` (both map to FN).

PNG export (`--png`) remains optional; `.txt` is the intended path for custom slides.
