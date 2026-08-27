# Predictive Maintenance of Aircraft Engines using Deep Autoencoders

Predictive maintenance on NASA C-MAPSS turbofan data (FD001–FD004): a deep autoencoder
learns healthy-cycle sensor behavior and flags abnormal degradation from rising
reconstruction error, ahead of failure.

**Team:** Aryaman Ghaisas (23BDS0071) · Akriti Agarwal (23BDS0038) · Nethra Krishnan (23BDS0093)

Full project context, research gaps, architecture rationale, and the current execution
plan live in [`AI_CONTEXT.md`](./AI_CONTEXT.md) — read that first. This README only covers
setup and day-to-day usage.

## Current direction

- **Baseline (Review-1):** autoencoder → reconstruction error → fixed threshold.
- **Proposed novelty:** regime-conditioned autoencoder + conformal-calibrated dynamic
  threshold + per-sensor attribution. See `AI_CONTEXT.md` §1.4 and §7 (gaps G1–G7).
- **Checkpoint (5 Sep 2026):** leakage-free preprocessing (FD001/FD002), trained baseline
  AE, fixed-threshold results, first dynamic-threshold comparison. Regime conditioning,
  conformal calibration, and attribution are prototypes only at this stage.

## Setup

```bash
git clone <repo-url>
cd predictive-maintenance-cmapss
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Place raw C-MAPSS files under `data/raw/CMAPSS/FD00{1,2,3,4}/` (already gitignored —
never commit raw data). Run tests with:

```bash
pytest
```

## Repository structure

```
configs/        Experiment configs (base + per-dataset + per-experiment overrides)
data/           raw/ (never edit) → interim/ → processed/  [all gitignored]
notebooks/      EDA and exploration only — no critical logic here
src/            production code: data/, models/, anomaly/, training/, evaluation/,
                experiments/, utils/
tests/          pytest suite, one file per src module family
experiments/    experiment run outputs + metadata (config, split, seed, metrics)
results/        tables/ figures/ checkpoints/ logs/
docs/           literature, novelty notes, architecture decisions, report drafts
dashboard/      health-monitoring dashboard app
```

See `AI_CONTEXT.md` §18–19 for full module contracts.

## Team ownership

| Area | Owner |
|---|---|
| Core ML pipeline, autoencoder, thresholds, integration | Aryaman |
| Data ingestion, EDA, normalization, regime representation | Akriti |
| Evaluation, literature/IP, reporting, dashboard | Nethra |

## Non-negotiable rules (see `AI_CONTEXT.md` §17, §28)

1. Split by **engine ID**, never by row — no engine crosses train/val/test.
2. Fit normalization and thresholds on **training/calibration data only**, never on test.
3. Keep the 3 operating settings explicitly represented (don't fold them into sensors).
4. Every experiment run must log its config, split, seed, and metrics.
5. Don't claim novelty/patentability or statistical guarantees as established fact —
   see `AI_CONTEXT.md` §26, §32 for IP/disclosure handling.

## AI coding assistant

`AI_CONTEXT.md` is written to be loaded as context by an AI coding assistant. Team
members can say "I am Aryaman / Akriti / Nethra" to load their role, responsibilities,
and current action-plan tasks (see `AI_CONTEXT.md` §6, §34).
