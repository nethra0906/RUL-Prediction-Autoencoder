# IP Strategy & Novelty — 1-Page Summary

**Project:** Predictive Maintenance of Aircraft Engines using Deep Autoencoders for Early Failure Detection
**Owner (this document):** Nethra Krishnan (23BDS0093) — evaluation, literature/IP documentation, reporting
**Team:** Aryaman Ghaisas (23BDS0071) · Akriti Agarwal (23BDS0038) · Nethra Krishnan (23BDS0093)
**Source:** `Novelty_Gaps_LitReview_ActionPlan.pdf`, Section 3 · Prepared 27 Aug 2026 · Target milestone 5 Sep 2026

> **Status note:** Novelty is the team's working research hypothesis, not a verified or granted claim. Nothing below should be represented as "patented" or "definitively novel" until an institutional IP cell / patent agent has completed a formal prior-art and patentability review.

---

## 1. Differentiated Novelty Statement

> **"A Regime-Conditioned Autoencoder with Conformal-Calibrated Dynamic Thresholding and Per-Sensor Anomaly Attribution for Early Failure Detection in Turbofan Engines."**

The Review-1 baseline — a healthy-region autoencoder with fixed-threshold reconstruction-error alerting — is mainstream prior art (published repeatedly on C-MAPSS since 2018). The three components below are individually known in isolated subfields (industrial fault detection, conformal/uncertainty quantification, explainable AI) but have **not been found combined into one turbofan early-warning system**. The combination, not any single piece, is the claimed contribution:

1. **Operating-regime-conditioned encoder** (closes gap G2 — single-operating-condition training). The 3 operational settings feed a small regime-embedding sub-network that conditions the encoder/decoder (concatenation or FiLM-style modulation), so one shared model generalizes across FD001–FD004 instead of retraining per sub-dataset.
2. **Conformal-calibrated dynamic threshold** (closes G1 & G3 — fixed thresholds, no statistical guarantee). A held-out healthy-cycle calibration set yields a conformal prediction interval on reconstruction error, conditioned on the regime embedding — giving a distribution-free, provable bound on false-alarm rate (illustrative target: ≤5% false alarms at 95% confidence) rather than a manually tuned cutoff.
3. **Per-sensor reconstruction-error attribution** (closes G4 — no explainability). Reconstruction error is decomposed back into its 21 sensor + 3 setting channels and ranked, so an alert carries "sensors 4, 7, 11 are driving this anomaly" instead of one opaque scalar.

*Stretch goal (closes G5/G6, not a prerequisite):* a shared regime-conditioned latent space feeding both reconstruction and a lightweight RUL regression head, trained with an asymmetric (late-prediction-penalizing) loss.

---

## 2. Patent-Style Claim Skeleton

*(Drafting skeleton for the report / provisional specification only — an institution's IP cell or a patent agent must refine actual claim language before any filing.)*

A computer-implemented method for early failure detection in a turbofan engine, comprising:

1. **(a)** receiving multivariate sensor and operating-condition data;
2. **(b)** generating a regime-embedding vector from the operating-condition data;
3. **(c)** conditioning an autoencoder's encoding and decoding on the regime-embedding vector to produce a reconstruction of the sensor data;
4. **(d)** computing a reconstruction-error signal;
5. **(e)** computing a dynamic detection threshold for the reconstruction-error signal using conformal calibration conditioned on the regime-embedding vector, such that the threshold provides a bounded false-alarm rate;
6. **(f)** decomposing the reconstruction error into per-sensor-channel contributions and ranking them; and
7. **(g)** generating an early-failure alert with an attached sensor-attribution report when the reconstruction-error signal exceeds the dynamic threshold.

---

## 3. Prior-Art Differentiation — arXiv 2601.10269 (2026)

*"Early Fault Detection on CMAPSS with Unsupervised LSTM Autoencoders"* is the **closest prior art** and must be read and cited first. It removes operating-condition effects via **regression-based normalization** applied *before* autoencoding, then uses a plain LSTM autoencoder with an adaptive, data-driven threshold.

**Differentiation points to state explicitly in the specification:**

| Dimension | arXiv 2601.10269 | This project |
|---|---|---|
| Handling operating conditions | Regression-based *normalization* pre-processing step (removes regime effects statistically, upstream of the model) | *Learned regime-conditioning inside the model* — a regime embedding directly modulates the encoder/decoder (concatenation / FiLM), not a pre-processing correction |
| Threshold | Adaptive, data-driven threshold (heuristic) | **Conformal calibration** conditioned on the regime embedding, yielding a distribution-free, statistically bounded false-alarm guarantee |
| Explainability | Aggregate/scalar reconstruction error only | **Per-sensor attribution map** ranking the 21+3 channels driving each alert |
| Combination | Single mechanism (normalization + adaptive threshold) | Three mechanisms fused into one pipeline: regime-conditioning + conformal threshold + sensor attribution |

Two further adjacent patents to differentiate from during a formal search: *"System and method for monitoring a turbomachine with anomaly detection corrected by a wear factor"* (US11521430) and *"Methods and systems for operating an aircraft engine"* (US12140075, dynamic threshold based on engine health/age). Neither combines regime-conditioning with conformal calibration and sensor attribution.

---

## 4. Practical Steps for Patentability

- **Notebook / commit tracking:** Keep a dated lab notebook and rely on Git commit history (this repository) as invention-date evidence. Every experiment must log its configuration and timestamp per AI_CONTEXT.md Section 21.
- **No public disclosure before IP-cell filing:** Do not publish the specific novelty combination (Section 1 above) in a paper, public conference talk, or a **public** GitHub repository before a provisional filing is made. India allows only a 12-month grace period for the inventor's own prior disclosure — publication should follow, or be simultaneous with, the provisional filing, not precede it.
- **Contact the institutional IP/Innovation cell this week**, in parallel with development, to start a provisional patent application — provisional filings can be amended later, but an early filing date matters most.
- **Run a real patent search** (not just a literature search) on Google Patents / Espacenet for "turbofan" + "autoencoder" + "dynamic threshold" and "aircraft engine health monitoring" + "conformal" before filing.
- **Frame claims as a system**, not pure mathematics: sensor pipeline → embedded/edge inference → dashboard → maintenance-trigger output. This addresses Section 3(k) of the Indian Patents Act, which bars a "mathematical method... per se" but permits a claim to a technical system producing a measurable real-world effect (e.g., bounded false-alarm rate, detection lead time).

---

*See also AI_CONTEXT.md Sections 1.4, 7 (G1–G4), and the source action-plan PDF Section 3 for full detail.*
