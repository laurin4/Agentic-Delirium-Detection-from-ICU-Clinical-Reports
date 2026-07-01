# Delirium Pipeline Demo — Thesis Case Summaries

**Primary output:** publication-quality case summaries for the Results chapter and presentation.

All exports are **anonymized** (`Beispiel-Fall A` / `B` — no patient IDs).

**Case pairing:** True Positive (TP) + False Negative (FN).

---

## Recommended workflow

```bash
cd delirium_project
source Ba_venv/bin/activate

# 1. Build snapshots from validation cohort (server; FN prefers Patient_0057 / Patient_0075)
python -m src.analysis.demo_delirium_case --snapshot-positive --snapshot-false-negative

# 2. Export thesis summaries
python -m src.analysis.demo_delirium_case --thesis
```

**Outputs** (`outputs/demo/`):

| File | Content |
|------|---------|
| `thesis_case_a_true_positive.md` | Case A only (Markdown) |
| `thesis_case_b_false_negative.md` | Case B only (Markdown) |
| `thesis_pipeline_case_summaries.md` | Both cases — copy into thesis |
| `thesis_pipeline_case_summaries.txt` | Plain text variant |

Each case contains five sections (~half a page):

1. **Klinischer Berichtsauszug** — 2–4 relevant sentences only  
2. **Regelbasierte Evidenzextraktion** — bullet summary  
3. **Evidenz-Bündel ans LLM** — condensed input, no prompts  
4. **LLM-Interpretation** — max. 3–4 bullets (Begründung, Signalstärke)  
5. **Finale Entscheidung** — Guardrail, Modellvorhersage, manuelle Referenz, Korrekt/Inkorrekt  

---

## Pick cases on the server

```bash
python -m src.analysis.demo_delirium_case --list-positive-candidates
python -m src.analysis.demo_delirium_case --list-false-negative-candidates

# Why didn't Patient_0057 / 0075 get picked?
python -m src.analysis.demo_delirium_case --diagnose-fn-patients
```

**Important:** Case B needs a **report-level FN** (`klasse=0` and `manual_report_ground_truth=1` on the **same** report). Patient_0057 and Patient_0075 from your error analysis may be **FP** (`klasse=1`, `manual=0`) — then they cannot be Case B. The diagnose command shows each report’s confusion group.

If a real FN report exists for 0057/0075:

```bash
python -m src.analysis.demo_delirium_case --snapshot-false-negative --fn-patient 0057
# or force exact report:
python -m src.analysis.demo_delirium_case --snapshot-false-negative \
  --validation-report-id Patient_0057_Report_YYYY

python -m src.analysis.demo_delirium_case --thesis
```

If you see `curated fallback` after snapshot build, Case B is **not** from the validation cohort — run diagnose on the server.

---

## Legacy / optional

```bash
python -m src.analysis.demo_delirium_case --txt    # hemorrhage-style walkthrough logs
python -m src.analysis.demo_delirium_case --both   # interactive terminal (not primary)
python -m src.analysis.demo_delirium_case --html   # browser preview
```

Legacy aliases: `--snapshot-negative`, `--negative` → FN case.
