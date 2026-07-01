# Delirium Pipeline Demo — Guide

Presentation walkthrough: **report → rule extraction → LLM → guardrails → klasse**.

All exported snapshots are **anonymized for public presentation**:
- No `PatientenID`, no `validation_report_id`, no report dates
- Display labels: **Beispiel-Fall A** (positiv) and **Beispiel-Fall B** (negativ)

---

## PowerPoint — what works best

| Method | PowerPoint fit | Recommendation |
|--------|----------------|----------------|
| **`--png` export** | Excellent — native images | **Use this** |
| HTML in browser | Poor — cannot embed live HTML | Preview only, then screenshot if needed |
| Terminal demo | N/A — live backup | Good for Q&A, not slides |

### Recommended workflow for your thesis talk

```bash
cd delirium_project
python -m src.analysis.demo_delirium_case --png
```

Inserts into PowerPoint:

```
outputs/demo/delirium_demo_fall_a.png   → slide “Beispiel-Fall A (Delir positiv)”
outputs/demo/delirium_demo_fall_b.png   → slide “Beispiel-Fall B (Delir negativ)”
```

In PowerPoint: **Insert → Pictures → select PNG**. One case per slide works well.

Optional: open `outputs/demo/delirium_pipeline_demo.html` in a browser to check layout, then screenshot — but `--png` is simpler.

---

## Commands

```bash
# PNG slides for PowerPoint (recommended)
python -m src.analysis.demo_delirium_case --png

# HTML preview in browser
python -m src.analysis.demo_delirium_case --html

# Live terminal walkthrough
python -m src.analysis.demo_delirium_case --both
```

---

## Regenerate from real validation data (server)

On the server (full frozen cohort). Identifiers are **stripped automatically** on export:

```bash
python -m src.analysis.demo_delirium_case --snapshot-positive
python -m src.analysis.demo_delirium_case --snapshot-negative
python -m src.analysis.demo_delirium_case --png
```

Copy to your laptop:

- `data/demo/positive_case.json`
- `data/demo/negative_case.json`
- `outputs/demo/delirium_demo_fall_*.png`

---

## Privacy note

Snapshots stored under `data/demo/` are safe to show in a public thesis defence:
clinical text is anonymized synthetic or scrubbed; all hospital/validation IDs are removed before save.
