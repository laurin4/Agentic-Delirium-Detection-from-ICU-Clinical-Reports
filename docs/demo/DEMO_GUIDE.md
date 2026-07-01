# Delirium Pipeline Demo — Guide

Hemorrhage-style step-by-step walkthrough for your thesis slides.

All exports are **anonymized** (`Beispiel-Fall A` / `B` — no patient IDs).

---

## Recommended: `.txt` for your own figures

```bash
cd delirium_project
source Ba_venv/bin/activate

# 1. Regenerate snapshots (picks clearer positive case automatically)
python -m src.analysis.demo_delirium_case --snapshot-positive --snapshot-negative

# 2. Export walkthrough text
python -m src.analysis.demo_delirium_case --txt
```

**Outputs:**

| File | Content |
|------|---------|
| `outputs/demo/delirium_demo_fall_a_walkthrough.txt` | True positive — STEP 1…7 |
| `outputs/demo/delirium_demo_fall_b_walkthrough.txt` | True negative — STEP 1…7 |
| `outputs/demo/delirium_pipeline_demo_walkthrough.txt` | Both cases combined |

Structure mirrors the hemorrhage demo:

1. Original clinical reports  
2. Rule-based evidence extraction  
3. Evidence bundle → LLM  
4. Agent 1 signals *(if LLM ran)*  
5. Agent 2 interpretation *(or LLM SKIPPED branch)*  
6. Clinical guardrails → klasse  
7. Validation label + final classification box  

Copy sections into PowerPoint / Figma and style as you like.

---

## Pick a different positive case

List top candidates (on server with full validation data):

```bash
python -m src.analysis.demo_delirium_case --list-positive-candidates
```

Skip the case you did not like and pick the next one:

```bash
python -m src.analysis.demo_delirium_case --snapshot-positive \
  --exclude-validation-report-id Patient_XXXX_Report_YYYY

# Or force a specific report:
python -m src.analysis.demo_delirium_case --snapshot-positive \
  --validation-report-id Patient_AAAA_Report_BBBB

python -m src.analysis.demo_delirium_case --txt
```

Positive auto-pick now prefers: **short reports**, **`Delir` in Diagnosen**, **`direct_delir_positive`**, few snippets — not long hypoaktiv-heavy epikrisen.

---

## Other commands

```bash
python -m src.analysis.demo_delirium_case --both          # live terminal walkthrough
python -m src.analysis.demo_delirium_case --html          # browser preview only
```

PNG export (`--png`) remains optional; `.txt` is the intended path for custom slides.
