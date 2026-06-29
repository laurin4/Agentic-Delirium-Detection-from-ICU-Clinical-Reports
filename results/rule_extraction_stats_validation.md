# Rule-based Evidence Extraction — Frozen Delirium Validation Cohort

_Generated: 2026-06-29 17:00_

- **Prediction-trace source:** `/Users/laurinseelig/Desktop/ZHAW Semester 6/ BA/Bachelor_Thesis/delirium_project/outputs/predictions/validation_cohort_predictions.csv` (outputs/predictions (legacy))
- **Patient count source:** prediction trace (outputs/predictions (legacy))
- **Report count source:** prediction trace (outputs/predictions (legacy))

> **Warnings**
>
> - Frozen report labels not found (manual_report_labels_frozen.csv).
>
> - Frozen patient cohort not found (patient_validation_cohort_frozen.csv).
>
> - Trace contains only 1 report(s) (< expected 616). This appears to be a partial or stub trace. Run on the analysis server for the full frozen cohort. All statistics below describe ONLY these 1 report(s).

## Summary

| Metric | Value | % of reports |
|---|---:|---:|
| Patients in validation cohort | 1 | – |
| Reports in validation cohort | 1 | – |
| Reports with ≥1 evidence snippet | 0 | 0.00% |
| Reports without evidence snippet | 1 | 100.00% |
| Reports skipped by prefilter (no LLM) | 1 | 100.00% |
| Reports eligible / sent to LLM | 0 | 0.00% |
| — of which short-report full-text fallback | 0 | 0.00% |
| Total evidence snippets extracted | 0 | – |
| Snippets per report — mean | 0 | – |
| Snippets per report — median | 0.0 | – |
| Snippets per report — min | 0 | – |
| Snippets per report — max | 0 | – |
| Snippets per report — mean (reports with ≥1) | not available | – |
| Mean original report length (chars) | 1.0 | – |
| Mean LLM evidence-bundle length (chars) | 0.0 | – |
| Text-length reduction before LLM | not available | – |

## Evidence-type breakdown

| Evidence type | Snippets | % of all snippets |
|---|---:|---:|
| `direct_delir` | 0 | not available |
| `indirect_symptom` | 0 | not available |
| `negation` | 0 | not available |
| `prophylaxis_or_risk` | 0 | not available |

## Interpretation (thesis-ready)

Of the 1 reports in the frozen validation cohort, 0 (0.00%) contained at least one rule-based delirium-related evidence snippet. After the negation/relevance prefilter, 0 reports (0.00%) were eligible for LLM-based interpretation, whereas 1 reports (100.00%) were skipped before any LLM call. In total, 0 evidence snippets were extracted.

_Definitions: a report is **eligible for LLM interpretation** when the rule layer produces at least one non-negation snippet (`direct_delir`, `indirect_symptom`, or `prophylaxis_or_risk`) or, where enabled, when a short report without snippets is forwarded as full text; reports with only negation evidence or no evidence are **skipped by the prefilter**. Counts are derived from the stored extraction metadata, not recomputed by re-running inference._
