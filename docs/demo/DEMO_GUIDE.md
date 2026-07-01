# Delirium Pipeline Demo — Thesis Case Summaries

**Primary output:** publication-quality case summaries for the Results chapter and presentation.

All exports are **anonymized** (`Beispiel-Fall A` / `B` — no patient IDs).

**Case pairing:** True Positive (TP) + False Negative (FN).

---

## Recommended workflow

```bash
cd delirium_project
source Ba_venv/bin/activate

# 1. Build snapshots from validation cohort (server; FN prefers PatientenID 308617 / 308954)
python3 -m src.analysis.demo_delirium_case --snapshot-positive --snapshot-false-negative

# 2. Export thesis summaries
python3 -m src.analysis.demo_delirium_case --thesis
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
python3 -m src.analysis.demo_delirium_case --list-positive-candidates
python3 -m src.analysis.demo_delirium_case --list-false-negative-candidates
python3 -m src.analysis.demo_delirium_case --diagnose-fn-patients
```

### Case B (FN) — pinned from evaluation

Case B is built from `outputs/analysis/manual_validation/final_evaluation/model_FN.csv`, **not** by guessing from report-level labels.

1. Confirms PatientenID `308617` / `308954` is FN in your final evaluation  
2. Picks one **model=0** report for that patient (real pipeline replay)  
3. **Forces** patient-level manual reference = Delir in the snapshot  

The thesis table shows **Modellvorhersage (Bericht)** vs **Manuelle Referenz (Patient)** — not 0/0 TN.

```bash
python3 -m src.analysis.demo_delirium_case --snapshot-false-negative --patienten-id 308617
python3 -m src.analysis.demo_delirium_case --thesis
```

If build fails, the command errors with the list of FN PatientenIDs in `model_FN.csv` (no silent wrong patient).

Prerequisite on server:

```bash
python3 -m src.analysis.final_manual_validation_evaluation   # creates model_FN.csv
```

---

## Legacy / optional

```bash
python3 -m src.analysis.demo_delirium_case --txt    # hemorrhage-style walkthrough logs
python3 -m src.analysis.demo_delirium_case --both   # interactive terminal (not primary)
python3 -m src.analysis.demo_delirium_case --html   # browser preview
```

Legacy aliases: `--snapshot-negative`, `--negative` → FN case.
