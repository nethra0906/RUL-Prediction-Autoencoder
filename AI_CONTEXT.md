# AI Coding Context --- NASA C-MAPSS Predictive Maintenance Project

> **Purpose:** This document is the canonical context file for an AI
> coding assistant working on the team's predictive-maintenance
> project.\
> **Primary dataset:** NASA C-MAPSS Turbofan Engine Degradation
> Simulation Dataset (FD001--FD004).\
> **Team:** Aryaman Ghaisas, Akriti Agarwal, Nethra Krishnan.\
> **Target milestone from the current action plan:** 5 September 2026.\
> **Current project theme:** Predictive Maintenance of Aircraft Engines
> using Deep Autoencoders for Early Failure Detection.

------------------------------------------------------------------------

## 0. How an AI coding assistant should use this file

This file is intended to be loaded as project-level context by a coding
assistant.

### Identity shortcut

When the user says:

> **"I am Aryaman."**

the assistant should immediately load the **Aryaman role/profile** in
Section 6 and understand:

-   Aryaman is on the core ML pipeline and integration
    work.
-   His main technical responsibility is the end-to-end
    predictive-maintenance pipeline: preprocessing, sequence generation,
    model integration, anomaly scoring, evaluation, experimentation, and
    integration of the novelty components.
-   He is especially responsible for understanding and implementing the
    **regime-conditioned autoencoder**, **dynamic/conformal
    thresholding**, and their integration into the overall system.
-   Code suggestions should preserve the project's existing architecture
    and research direction rather than replacing it with an unrelated
    standard approach.
-   The assistant should prefer reproducible, modular Python code with
    clear experiment configuration and tests.
-   The assistant should distinguish the **baseline system** from the
    **novel proposed system**.

If the user says **"I am Akriti"** or **"I am Nethra"**, use their
respective profiles in Section 6.

### Important assistant behavior

When helping with code:

1.  **Do not silently change the research objective.**
2.  **Do not assume a feature is already implemented if it is only
    proposed.**
3.  **Preserve the distinction between:**
    -   baseline autoencoder,
    -   dynamic-threshold precursor,
    -   regime-conditioned model,
    -   conformal calibration,
    -   sensor-level attribution,
    -   optional joint RUL head.
4.  **Avoid data leakage.** In particular, do not normalize, fit
    thresholds, or tune hyperparameters using future/test information.
5.  **Keep FD001 and FD002 as mandatory early validation datasets.**
6.  Prefer reusable modules over notebook-only implementations.
7.  Every experiment should record configuration, dataset split, random
    seed, metrics, and output location.
8.  Do not claim that the proposed combination is definitively novel or
    patentable without a verified prior-art search. Treat novelty as the
    team's working research hypothesis.
9.  If a requested implementation conflicts with the source project
    plan, explain the conflict before changing the architecture.
10. When debugging, first identify whether the issue is:
    -   data ingestion,
    -   split/leakage,
    -   normalization,
    -   window construction,
    -   labels,
    -   model shape,
    -   training,
    -   threshold calibration,
    -   metric implementation,
    -   plotting,
    -   experiment configuration,
    -   or integration.

------------------------------------------------------------------------

# 1. Project identity

## 1.1 Working title

**Predictive Maintenance of Aircraft Engines using Deep Autoencoders for
Early Failure Detection**

## 1.2 Core problem

Aircraft engines degrade gradually under continuous thermal and
mechanical stress. Unplanned in-flight failures are costly, disruptive,
and safety-critical. Fixed-interval scheduled maintenance can also
discard useful remaining component life.

The project therefore investigates whether multivariate engine sensor
streams can be used to learn normal/healthy degradation behavior and
detect abnormal degradation early enough to support predictive
maintenance.

## 1.3 Original Review-1 objective

The Review-1 proposal is:

> Train a deep autoencoder on healthy-region cycles of NASA C-MAPSS
> multivariate sensor streams, use reconstruction error as an anomaly
> score, detect a rising degradation signal before failure, and
> separately estimate RUL.

The original one-line approach is:

> **Unsupervised anomaly scoring via reconstruction error →
> early-warning signal for Remaining Useful Life (RUL) estimation.**

This is the **baseline research direction**, not the final novelty
claim.

## 1.4 Current research direction after literature/gap analysis

The literature/action-plan document identifies that a simple
reconstruction-error autoencoder on C-MAPSS is already a
well-established approach.

The project should therefore evolve from:

**Healthy-data Autoencoder → Reconstruction Error → Fixed Threshold →
Alert**

toward:

**Multivariate Sensor + Operating Settings → Regime Representation →
Regime-Conditioned Autoencoder → Reconstruction Error →
Conformal/Dynamic Threshold → Sensor Attribution → Early Failure Alert**

with a possible stretch extension:

**Shared Latent Representation → RUL Regression Head → Asymmetric RUL
Loss**

The recommended primary novelty described in the action plan is:

> **"A Regime-Conditioned Autoencoder with Conformal-Calibrated Dynamic
> Thresholding and Per-Sensor Anomaly Attribution for Early Failure
> Detection in Turbofan Engines."**

The three main components are:

1.  Operating-regime-conditioned autoencoder.
2.  Conformal-calibrated dynamic reconstruction-error threshold.
3.  Per-sensor reconstruction-error attribution.

The joint RUL head is a stretch goal rather than a prerequisite for the
core contribution.

------------------------------------------------------------------------

# 2. Source-of-truth documents

This context is based primarily on the team's literature/action-plan PDF
and secondarily on the Review-1 PPT.

### Primary source

**Novelty_Gaps_LitReview_ActionPlan.pdf**

This document contains: - the current project position, - research gaps
G1--G7, - proposed novelty, - patent-style claim skeleton, - relevant
papers, - the 9-day execution plan from 27 August to 5 September 2026, -
concrete deliverables for the September checkpoint.

### Secondary source

**Predictive_Maintenance_Aircraft_Engines_Review1.pptx**

This document contains: - original problem statement, - original
workflow, - C-MAPSS dataset description, - evaluation metrics, -
tangible project outcomes, - original 12-week project plan, - initial
reference list.

### Source precedence

When the two documents overlap:

1.  Use the **PDF/action plan** for current novelty, research gaps,
    immediate execution priorities, and the 27 Aug--5 Sep checkpoint.
2.  Use the **PPT** for the original baseline framing, workflow,
    deliverables, and 12-week structure.
3.  If the documents differ, do not silently merge contradictory claims.
    State which project stage the information belongs to.

------------------------------------------------------------------------

# 3. Dataset context --- NASA C-MAPSS

## 3.1 Dataset

**NASA C-MAPSS --- Commercial Modular Aero-Propulsion System
Simulation**

The project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation
Dataset.

The Review-1 material describes four sub-datasets:

-   FD001
-   FD002
-   FD003
-   FD004

The datasets represent simulated run-to-failure trajectories for
turbofan engines.

## 3.2 Dataset characteristics

According to the project material:

-   Each engine begins with an unknown initial wear/degradation state.
-   Engines degrade over operating cycles.
-   Training trajectories run to failure.
-   Test trajectories are truncated before failure.
-   The sub-datasets differ in operating-condition complexity and
    fault-mode complexity.
-   FD001--FD004 therefore provide increasing modeling difficulty.

The project material describes:

-   **21 sensor measurements**
-   **3 operational settings**
-   **100+ engines per sub-dataset** in the project overview.

The actual raw C-MAPSS files should always be inspected rather than
assuming a particular row count or exact column behavior.

## 3.3 Input representation

For each engine cycle, the conceptual input is:

``` text
3 operational settings
+
21 sensor channels
=
24 raw variables
```

The model should not blindly treat all 24 variables identically.

The three operational settings are especially important because they
characterize the operating regime and are central to the proposed
regime-conditioning mechanism.

## 3.4 Engine trajectory structure

Conceptually:

``` text
Engine ID
    ↓
Cycle 1
Cycle 2
Cycle 3
...
Cycle T
    ↓
Degradation / failure
```

For training engines:

``` text
Cycle 1 -----------------------> Failure
       healthy → degradation → failure
```

For test engines:

``` text
Cycle 1 -----------------> Truncated observation
                                  ↑
                           actual failure is later
```

The test RUL ground truth is used for quantitative evaluation.

------------------------------------------------------------------------

# 4. C-MAPSS file conventions to preserve

A typical C-MAPSS text file has:

``` text
unit_id
cycle
setting_1
setting_2
setting_3
sensor_1
...
sensor_21
```

The implementation should explicitly assign column names instead of
relying on unnamed dataframe columns.

Recommended canonical names:

``` python
[
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    ...
    "sensor_21",
]
```

The assistant should inspect the actual files before coding assumptions
about missing values, constant sensors, whitespace, or formatting.

------------------------------------------------------------------------

# 5. RUL labeling

## 5.1 Training RUL

For a training engine with final cycle `T`, the standard conceptual RUL
at cycle `t` is:

``` text
RUL(t) = T - t
```

A capped/piecewise-linear RUL formulation is planned for the project.

If a cap `RUL_MAX` is used:

``` text
RUL_capped(t) = min(T - t, RUL_MAX)
```

The exact cap must be stored in configuration rather than hard-coded
throughout the project.

## 5.2 Test RUL

The test set ends before actual failure. The provided test RUL
information is needed to evaluate predictions.

The assistant must be extremely careful to distinguish:

-   the observed final cycle,
-   the provided test RUL,
-   inferred failure cycle,
-   predicted RUL.

Do not accidentally use future failure information as a model input.

## 5.3 Healthy-region labels for autoencoder training

The core anomaly-detection idea is to train the autoencoder using only
early/healthy-region cycles.

This is critical.

The intended logic is:

``` text
Training trajectory
      ↓
identify healthy region
      ↓
train autoencoder primarily/only on healthy cycles
      ↓
learn normal reconstruction behavior
      ↓
degradation causes reconstruction error to increase
```

The precise healthy-region definition should be configurable and
documented.

------------------------------------------------------------------------

# 6. Team ownership and coding-assistant profiles

## 6.1 Team

  -----------------------------------------------------------------------
  Member                  ID                      Primary responsibility
  ----------------------- ----------------------- -----------------------
  **Aryaman Ghaisas**     23BDS0071               Core ML pipeline,
                                                  autoencoder, anomaly
                                                  detection, integration

  **Akriti Agarwal**      23BDS0038               Data pipeline, EDA,
                                                  normalization, regime
                                                  representation,
                                                  experimental support

  **Nethra Krishnan**     23BDS0093               Evaluation,
                                                  literature/IP
                                                  documentation,
                                                  reporting,
                                                  dashboard/demo support
  -----------------------------------------------------------------------

The assignments below follow the action-plan PDF but are structured so a
coding assistant can infer ownership.

------------------------------------------------------------------------

## 6.2 Aryaman --- core ML and integration owner

### Identity

When the user says:

> **"I am Aryaman."**

treat them as the owner of the main implementation/integration path.

### Primary responsibilities

Aryaman owns or co-owns:

-   sliding-window sequence generation,
-   leakage-free train/validation/test logic,
-   baseline autoencoder implementation,
-   reconstruction-error computation,
-   fixed-threshold baseline,
-   dynamic-threshold integration,
-   regime-conditioning integration,
-   model training/tuning,
-   end-to-end experiment orchestration,
-   core anomaly-detection evaluation,
-   integration of the final pipeline.

### Immediate action-plan tasks

From the 27 Aug--5 Sep execution plan:

**Day 1** - Read arXiv 2601.10269 and dynamic-threshold papers. - Draft
novelty differentiation paragraph.

**Day 2** - Finish gap table with citations.

**Day 3** - Build sliding-window sequence generator. - Build
train/validation/test split logic.

**Day 4** - Unit-test preprocessing on FD001. - Verify tensor shapes and
leakage-free splits.

**Day 5** - Implement and train vanilla autoencoder on FD001 healthy
cycles. - Implement reconstruction-error calculation. - Implement basic
fixed-threshold detector as the "before" baseline.

**Day 6** - Tune baseline AE: latent dimension, epochs, etc. - Log
results.

**Day 7** - Integrate dynamic threshold. - Compare false-alarm rate
against fixed threshold.

**Day 8** - Bug-fix. - Re-run experiments. - Finalize plots/tables.

**Day 9** - Full-team rehearsal and reproducibility verification.

### Aryaman coding preferences for this project

When helping Aryaman code:

-   Keep core model code in reusable Python modules.
-   Do not bury important logic in notebooks.
-   Use configuration objects/dataclasses/YAML rather than scattered
    constants.
-   Keep model input dimensions explicit.
-   Make `window_size`, `stride`, `latent_dim`, `epochs`, `batch_size`,
    learning rate, healthy-region definition, threshold parameters, and
    random seed configurable.
-   Add assertions for shape and leakage checks.
-   Return structured outputs rather than loose tuples when a pipeline
    stage has many outputs.
-   Preserve engine IDs and cycle indices alongside tensors.
-   Never shuffle across engines before a trajectory-level split.
-   Ensure each window can be traced back to its engine and ending
    cycle.

### Aryaman's expected debugging workflow

When Aryaman reports a model problem:

1.  Print/inspect dataframe shape.
2.  Inspect engine IDs and cycle ranges.
3.  Validate feature columns.
4.  Check train/validation/test engine separation.
5.  Check normalization fitting scope.
6.  Check window shape.
7.  Check model input/output shape.
8.  Check reconstruction loss.
9.  Check reconstruction error distribution.
10. Check threshold calibration.
11. Check event-level metrics.
12. Only then tune the model.

------------------------------------------------------------------------

## 6.3 Akriti --- preprocessing, EDA, regime representation

### Identity

When the user says:

> **"I am Akriti."**

treat them as the primary owner of the data-preparation/EDA path.

### Primary responsibilities

Akriti owns or co-owns:

-   repository/environment setup,
-   C-MAPSS FD001--FD004 acquisition and organization,
-   exploratory data analysis,
-   sensor distributions,
-   healthy vs degraded cycle analysis,
-   operating-condition analysis,
-   normalization pipeline,
-   operating-regime representation,
-   cross-dataset preprocessing QA,
-   regime-embedding input preparation.

### Immediate action-plan tasks

**Day 1** - Set up shared repository. - Set up environment. -
Download/organize C-MAPSS FD001--FD004. - Establish folder structure.

**Day 2** - EDA: - sensor distributions, - healthy vs degraded cycles, -
operating-condition counts for each FD00x dataset.

**Day 3** - Build normalization pipeline. - Use per-condition z-score as
the planned initial normalization approach. - Provide a stub/interface
for future regime-embedding input.

**Day 4** - Cross-check preprocessing on FD002 with six operating
conditions. - Confirm pipeline generalization beyond FD001.

**Day 5** - Implement reconstruction-error computation and basic
fixed-threshold detector as shared work where needed.

**Day 6** - Implement an initial adaptive/dynamic threshold precursor: -
percentile-based, or - standard-deviation-based. - This is a precursor
to the full conformal threshold.

**Day 7** - Build regime-embedding input: - concatenation, or - small
MLP on the three operational settings. - Provide it as a stub layer
feeding the encoder.

**Day 8** - Cross-validate that regime embedding does not break FD001
results.

**Day 9** - Proofread report/deck and support demo preparation.

### Akriti coding rules

When helping Akriti:

-   Preserve raw data separately from processed data.
-   Make preprocessing deterministic.
-   Fit scalers only on allowed training/calibration data.
-   Do not calculate normalization statistics from the entire dataset
    unless explicitly justified.
-   Keep the three operational settings available even if they are
    excluded from the standard sensor feature tensor.
-   Document which sensors are constant or near-constant.
-   Keep preprocessing reusable across FD001--FD004.
-   Avoid writing FD001-specific logic that cannot generalize to
    FD002--FD004.

------------------------------------------------------------------------

## 6.4 Nethra --- evaluation, literature/IP, reporting and dashboard

### Identity

When the user says:

> **"I am Nethra."**

treat them as the owner of evaluation/documentation/demo work.

### Primary responsibilities

Nethra owns or co-owns:

-   evaluation harness,
-   Precision/Recall/F1/ROC-AUC,
-   RMSE/MAE/NASA scoring,
-   detection lead-time calculations,
-   experiment result tables,
-   reconstruction-error plots,
-   literature/gap documentation,
-   patent landscape research,
-   report/deck preparation,
-   dashboard/demo support.

### Immediate action-plan tasks

**Day 1** - Draft patent-style claim skeleton into the report. - Start
search for adjacent patents.

**Day 2** - Continue patent landscape search. - Write a one-page
novelty/IP strategy section.

**Day 3** - Build piecewise-linear/capped RUL label generator for later
regression use.

**Day 4** - Draft baseline autoencoder architecture/design document.

**Day 5** - Set up evaluation harness: - Precision, - Recall, - F1, -
ROC-AUC, - RMSE, - MAE, - NASA scoring function.

**Day 6** - Run baseline evaluation. - Tabulate metrics. - Plot
reconstruction-error trend against true failure cycle.

**Day 7** - Start compiling Review-2 report: - problem statement, - gap
table, - novelty, - dataset, - initial results.

**Day 8** - Finish report draft. - Build 6--8 slide Review-2 deck.

**Day 9** - Proofread report/deck. - Prepare speaking parts. -
Anticipate reviewer questions on novelty/patentability.

------------------------------------------------------------------------

# 7. Research gaps that drive the implementation

The action-plan PDF identifies seven gaps.

## G1 --- Fixed/global reconstruction-error thresholds

### Problem

Many autoencoder anomaly detectors use a single static or manually tuned
threshold.

### Project response

Use a dynamic threshold that adapts to operating conditions and
eventually uses conformal calibration.

### Implementation priority

**High.**

Baseline:

``` text
global fixed threshold
```

Improved precursor:

``` text
percentile/std-based dynamic threshold
```

Final target:

``` text
regime-conditioned conformal threshold
```

------------------------------------------------------------------------

## G2 --- Single-operating-condition training

### Problem

Many studies train/evaluate separately on FD001 and do not generalize
one shared model across FD002--FD004.

### Project response

Condition the autoencoder on operating regime.

Possible mechanisms:

-   concatenate learned regime vector with sensor representation,
-   use a small MLP to encode the three settings,
-   use FiLM-style conditioning,
-   otherwise use a simple conditioning architecture that is
    experimentally justified.

### Implementation priority

**High.**

The initial implementation can be a simple regime embedding before
attempting a more sophisticated conditioning mechanism.

------------------------------------------------------------------------

## G3 --- Lack of statistically calibrated confidence/false-alarm guarantees

### Problem

Point anomaly scores do not provide rigorous uncertainty/false-alarm
guarantees.

### Project response

Use conformal calibration on a held-out healthy calibration set.

Conceptually:

``` text
healthy calibration data
        ↓
reconstruction errors
        ↓
conformal calibration
        ↓
quantile / prediction threshold
        ↓
dynamic alert criterion
```

The action-plan document gives an illustrative target such as:

> ≤5% false alarms with 95% confidence

This should be treated as an illustrative statistical target, not a
guaranteed empirical outcome.

### Implementation priority

**Core novelty component after baseline/dynamic-threshold precursor.**

------------------------------------------------------------------------

## G4 --- No sensor-level attribution

### Problem

A scalar reconstruction error tells us that something is wrong but not
which channels drive the anomaly.

### Project response

Compute per-channel reconstruction error.

For feature/channel `j`:

``` text
error_j = aggregation over time of (x_j - x_hat_j)^2
```

Then rank channels.

For a window:

``` text
sensor_7      0.31
sensor_11     0.27
sensor_4      0.22
...
```

The exact aggregation should be documented and kept consistent across
experiments.

### Output

The alert should ideally contain:

-   total anomaly score,
-   threshold,
-   alert status,
-   top contributing sensors,
-   contribution values,
-   engine ID,
-   cycle/window endpoint.

------------------------------------------------------------------------

## G5 --- Detection decoupled from RUL regression

### Problem

The baseline project uses an autoencoder for detection and a separate
RUL model.

### Proposed response

Optional joint model:

``` text
                    ┌──> reconstruction
shared latent space ┤
                    └──> RUL regression
```

### Status

**Stretch goal.**

Do not block the core project on this.

------------------------------------------------------------------------

## G6 --- Symmetric RUL losses

### Problem

Standard MSE treats early and late RUL errors symmetrically, although
late predictions are operationally more dangerous.

### Proposed response

Use an asymmetric loss related to the NASA scoring objective.

Potential structure:

``` text
L_total =
    reconstruction_loss
    + λ * asymmetric_RUL_loss
```

The exact formula must be selected and documented from verified
literature before being presented as the final methodology.

### Status

**Stretch goal.**

------------------------------------------------------------------------

## G7 --- Deployment constraints

### Problem

Many academic models focus on predictive accuracy without considering
model size, latency, or edge/on-board computation.

### Proposed response

Eventually report deployment-oriented properties such as:

-   number of parameters,
-   model size,
-   inference latency,
-   approximate memory footprint,
-   batch vs single-window inference,
-   feasibility of edge deployment.

### Status

Secondary research/deployment objective.

------------------------------------------------------------------------

# 8. Baseline vs proposed system

This distinction is critical.

## 8.1 Baseline system

The baseline should be deliberately simple and reproducible:

``` text
C-MAPSS
  ↓
preprocessing
  ↓
healthy-cycle selection
  ↓
normalization
  ↓
sliding windows
  ↓
vanilla deep autoencoder
  ↓
reconstruction error
  ↓
fixed threshold
  ↓
anomaly/early-warning signal
```

This establishes the **before** condition.

## 8.2 Dynamic-threshold precursor

The first improvement should be:

``` text
reconstruction error
       ↓
adaptive threshold
       ↓
alert
```

Candidate simple approaches:

-   rolling percentile,
-   rolling mean + k·standard deviation,
-   condition-specific percentile.

This is an engineering precursor, not the final novelty.

## 8.3 Proposed primary system

``` text
Sensor streams + operational settings
                  ↓
        Regime embedding
                  ↓
       Regime-conditioned AE
                  ↓
       Reconstruction error
                  ↓
     Conformal calibration
                  ↓
    Dynamic regime-aware threshold
                  ↓
     ┌────────────┴─────────────┐
     ↓                          ↓
Early failure alert      Sensor attribution
```

The alert should contain both an anomaly decision and an explanation.

## 8.4 Stretch joint system

``` text
                ┌──────────────> Reconstruction
                │
Input → Encoder → Shared latent
                │
                └──────────────> RUL head
                                   ↓
                            Asymmetric loss
```

------------------------------------------------------------------------

# 9. Model architecture guidance

## 9.1 Baseline autoencoder

A generic architecture can be:

``` text
Input window
    ↓
Flatten / temporal representation
    ↓
Dense / temporal encoder
    ↓
Latent vector
    ↓
Decoder
    ↓
Reconstructed window
```

The exact layer sizes are an experiment variable.

Do not hard-code a supposedly optimal architecture without experimental
evidence.

Track:

-   input dimension,
-   window size,
-   latent dimension,
-   hidden dimensions,
-   activation,
-   dropout if used,
-   normalization layers if used,
-   optimizer,
-   learning rate,
-   batch size,
-   epochs,
-   early stopping,
-   random seed.

## 9.2 Regime-conditioned encoder

The three operational settings should feed a regime sub-network.

Conceptual version:

``` text
settings_1 ─┐
settings_2 ─┼─> Regime MLP ─> regime_embedding
settings_3 ─┘
                              ↓
sensor/window representation → conditioned encoder
```

Two initial implementations are acceptable:

### Option A --- concatenation

``` text
sensor_embedding || regime_embedding
```

then feed the combined vector into the encoder.

### Option B --- FiLM-style modulation

``` text
h' = γ(regime) ⊙ h + β(regime)
```

Use Option B only if implementation complexity is justified.

Start with the simpler mechanism so the novelty experiment is
attributable and debuggable.

------------------------------------------------------------------------

# 10. Reconstruction error

For input `x` and reconstruction `x_hat`, the base per-element squared
error is:

``` text
e = (x - x_hat)^2
```

A total reconstruction error can be:

``` text
MSE(x, x_hat)
```

For sensor attribution, preserve the channel-wise error before reducing
it to a scalar.

For a window of length `W` and channel `j`:

``` text
E_j = mean_t [(x_t,j - x_hat_t,j)^2]
```

Then:

``` text
E_total = mean_j E_j
```

The exact reduction should be defined once and reused consistently.

------------------------------------------------------------------------

# 11. Dynamic thresholding

## 11.1 Fixed baseline

A simple baseline:

``` text
alert = reconstruction_error > threshold
```

The threshold must be fitted/calibrated using appropriate healthy data.

Do not select a threshold based on the test set.

## 11.2 Dynamic threshold

The threshold can depend on context:

``` text
threshold_t = f(regime_t, calibration_history)
```

The purpose is to prevent normal changes in operating regime from being
incorrectly interpreted as degradation.

## 11.3 Conformal calibration target

The proposed final mechanism is a conformal-calibrated threshold based
on a held-out healthy calibration set.

High-level workflow:

``` text
TRAIN
healthy training cycles
       ↓
fit autoencoder

CALIBRATION
separate healthy calibration cycles
       ↓
compute nonconformity scores
       ↓
fit conformal quantile / threshold

TEST
new window
       ↓
regime representation
       ↓
reconstruction error
       ↓
dynamic conformal threshold
       ↓
alert if score exceeds threshold
```

The calibration set must be isolated from the training set.

The assistant should explicitly warn if code accidentally uses the same
observations to both train the model and establish the claimed
statistical calibration.

------------------------------------------------------------------------

# 12. Event-level anomaly detection

A common implementation mistake is to evaluate every window
independently and report a large number of point-level false alarms.

The project is interested in **early failure detection**, so event-level
logic matters.

Potential conceptual pipeline:

``` text
window scores
     ↓
threshold crossings
     ↓
persistence / consecutive-alert rule
     ↓
first valid alert
     ↓
failure cycle
     ↓
lead time
```

The exact persistence rule must be an explicit configuration.

For example, if a threshold is crossed once due to noise, it may not
constitute a valid early-warning event.

Do not invent a persistence value unless the experiment specifies one.

------------------------------------------------------------------------

# 13. Detection lead time

The Review-1 metric is:

``` text
Δt = actual failure cycle - alert cycle
```

A positive value means the alert occurred before failure.

The evaluation implementation should clearly define:

-   which cycle is considered the alert cycle,
-   whether the first threshold crossing is used,
-   whether persistence is required,
-   how multiple alerts are handled,
-   what happens if no alert occurs.

These choices must be consistent across models.

------------------------------------------------------------------------

# 14. RUL evaluation

The project plans to evaluate RUL using:

## MAE

``` text
MAE = mean(|y - y_hat|)
```

## RMSE

``` text
RMSE = sqrt(mean((y - y_hat)^2))
```

RMSE penalizes large deviations more heavily.

## NASA scoring function

The project material explicitly identifies the NASA scoring function
because RUL errors are asymmetric and late predictions are more
operationally costly.

The exact implementation should be kept in a dedicated metric module and
unit-tested against known examples.

## Detection lead time

Also report:

``` text
Δt = failure_cycle - alert_cycle
```

Do not substitute RUL accuracy for early-warning performance; they
answer different questions.

------------------------------------------------------------------------

# 15. Evaluation design

## 15.1 Detection metrics

Report:

-   Precision
-   Recall
-   F1
-   ROC-AUC
-   false-alarm rate
-   detection lead time

## 15.2 RUL metrics

Report:

-   MAE
-   RMSE
-   NASA score

## 15.3 Required baseline comparison

At minimum compare:

1.  Fixed-threshold autoencoder.
2.  Dynamic-threshold autoencoder.
3.  Regime-conditioned autoencoder / threshold variant as it becomes
    available.
4.  Final conformal-calibrated regime-conditioned model.

If RUL models are implemented:

5.  LSTM/regression baseline.
6.  Optional joint AE + RUL model.

## 15.4 Metrics should not be cherry-picked

Every experiment should use the same evaluation protocol.

Results should be stored in a structured table such as:

  -------------------------------------------------------------------------------------------------------------
  Experiment   Dataset   Model   Threshold     Precision   Recall     F1   ROC-AUC    MAE   RMSE    NASA   Lead
                                                                                                   Score   Time
  ------------ --------- ------- ----------- ----------- -------- ------ --------- ------ ------ ------- ------

  -------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 16. Required experiments

## Experiment A --- preprocessing sanity check

Datasets:

-   FD001
-   FD002

Check:

-   no NaNs after preprocessing,
-   expected columns,
-   cycle monotonicity,
-   engine grouping,
-   no engine overlap between splits,
-   normalization correctness,
-   window shape.

## Experiment B --- vanilla AE

Train on healthy FD001 cycles.

Measure:

-   reconstruction loss,
-   reconstruction-error distribution,
-   fixed threshold,
-   Precision/Recall/F1,
-   ROC-AUC,
-   lead time where applicable.

## Experiment C --- dynamic threshold precursor

Compare:

``` text
fixed threshold vs dynamic threshold
```

Primary early evidence:

-   false-alarm reduction,
-   precision,
-   recall,
-   F1,
-   lead time.

## Experiment D --- regime conditioning

Start with FD002 because it has multiple operating conditions.

Compare:

``` text
vanilla AE
vs.
regime-conditioned AE
```

Evaluate whether operating-condition variation is better handled.

## Experiment E --- conformal threshold

Compare:

``` text
fixed threshold
dynamic heuristic threshold
conformal-calibrated threshold
```

Focus on:

-   false-alarm behavior,
-   coverage/calibration behavior,
-   detection performance,
-   robustness across operating regimes.

## Experiment F --- sensor attribution

For detected anomalies:

-   calculate per-sensor reconstruction contribution,
-   rank top sensors,
-   visualize contributions.

## Experiment G --- cross-FD generalization

Eventually evaluate:

``` text
train on one/multiple FD datasets
       ↓
test across FD001–FD004
```

The exact train/test configuration should be declared explicitly.

Do not claim cross-FD generalization from a single FD001 experiment.

## Experiment H --- optional joint RUL model

Only after the anomaly pipeline is stable:

``` text
shared latent representation
      ├──> reconstruction
      └──> RUL
```

Compare against separate baselines.

------------------------------------------------------------------------

# 17. Data leakage rules

These are non-negotiable.

## Rule 1 --- Split by engine, not random rows

Do not allow windows from the same engine trajectory to appear across
train and validation/test splits unless the experimental design
explicitly calls for a within-engine setup.

Preferred:

``` text
engine IDs
   ↓
train engines
validation engines
test engines
```

## Rule 2 --- Fit preprocessing only on allowed training data

For normalization:

``` text
fit scaler → training data only
transform → validation/test
```

If a condition-specific calibration is used, define exactly which data
are permitted.

## Rule 3 --- Thresholds cannot use test labels

Never choose a threshold because it gives the best test-set F1.

## Rule 4 --- Conformal calibration needs a separate calibration set

Do not reuse training observations as the claimed held-out calibration
set.

## Rule 5 --- RUL ground truth is evaluation-only

Test failure information must not enter model input or threshold
selection.

------------------------------------------------------------------------

# 18. Repository structure

The following is the recommended repository structure for the project.

``` text
predictive-maintenance-cmapps/
│
├── README.md
├── AI_CONTEXT.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── base.yaml
│   ├── fd001.yaml
│   ├── fd002.yaml
│   ├── fd003.yaml
│   ├── fd004.yaml
│   ├── baseline_ae.yaml
│   ├── dynamic_threshold.yaml
│   └── conformal_regime_ae.yaml
│
├── data/
│   ├── raw/
│   │   └── CMAPSS/
│   │       ├── FD001/
│   │       ├── FD002/
│   │       ├── FD003/
│   │       └── FD004/
│   │
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_baseline_autoencoder.ipynb
│   ├── 04_dynamic_threshold.ipynb
│   ├── 05_regime_conditioning.ipynb
│   ├── 06_conformal_threshold.ipynb
│   ├── 07_sensor_attribution.ipynb
│   └── 08_final_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── schema.py
│   │   ├── splits.py
│   │   ├── rul.py
│   │   ├── healthy_region.py
│   │   ├── normalization.py
│   │   └── windows.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── autoencoder.py
│   │   ├── regime_encoder.py
│   │   ├── conditioned_autoencoder.py
│   │   └── rul_head.py
│   │
│   ├── anomaly/
│   │   ├── __init__.py
│   │   ├── reconstruction.py
│   │   ├── fixed_threshold.py
│   │   ├── dynamic_threshold.py
│   │   ├── conformal.py
│   │   ├── event_detection.py
│   │   └── attribution.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_autoencoder.py
│   │   ├── train_joint.py
│   │   ├── losses.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── classification_metrics.py
│   │   ├── rul_metrics.py
│   │   ├── nasa_score.py
│   │   ├── lead_time.py
│   │   └── evaluation_runner.py
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── run_baseline.py
│   │   ├── run_dynamic_threshold.py
│   │   ├── run_regime_conditioned.py
│   │   └── run_conformal.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── logging.py
│       ├── io.py
│       └── plotting.py
│
├── tests/
│   ├── test_schema.py
│   ├── test_rul.py
│   ├── test_splits.py
│   ├── test_normalization.py
│   ├── test_windows.py
│   ├── test_reconstruction.py
│   ├── test_thresholds.py
│   └── test_metrics.py
│
├── experiments/
│   ├── baseline/
│   ├── dynamic_threshold/
│   ├── regime_conditioned/
│   └── conformal/
│
├── results/
│   ├── tables/
│   ├── figures/
│   ├── checkpoints/
│   └── logs/
│
├── docs/
│   ├── literature/
│   ├── novelty/
│   ├── architecture/
│   ├── experiments/
│   └── report/
│
└── dashboard/
    ├── app.py
    └── components/
```

## Repository rules

### `data/raw`

Never modify raw C-MAPSS files.

### `data/interim`

Use for cleaned/intermediate artifacts that can be regenerated.

### `data/processed`

Use for reusable processed datasets/arrays where appropriate.

### `src`

This is the main production/research codebase.

### `notebooks`

Use notebooks for:

-   exploration,
-   visualization,
-   quick experiments,
-   result interpretation.

Do not put critical reusable functionality only in notebooks.

### `tests`

Every important mathematical or preprocessing component should have
tests.

### `experiments`

Keep experiment outputs and metadata separate from source code.

### `results`

Only reproducible outputs should be committed/shared.

### `docs`

Store architecture decisions, literature notes, novelty analysis, and
report material.

------------------------------------------------------------------------

# 19. Recommended module contracts

## `data/loaders.py`

Responsibilities:

-   read C-MAPSS files,
-   return standardized pandas DataFrames,
-   preserve engine/cycle identifiers.

Suggested conceptual API:

``` python
load_train(fd_id) -> DataFrame
load_test(fd_id) -> DataFrame
load_test_rul(fd_id) -> Series
```

## `data/splits.py`

Responsibilities:

-   engine-level splits,
-   deterministic random seed,
-   leakage prevention.

Suggested API:

``` python
split_by_engine(df, ...)
```

## `data/normalization.py`

Responsibilities:

-   fit scaler,
-   transform data,
-   inverse transform when needed,
-   optionally support condition-aware normalization.

Suggested API:

``` python
fit_normalizer(train_df, ...)
transform(df, ...)
```

## `data/windows.py`

Responsibilities:

-   create temporal windows,
-   preserve metadata,
-   avoid crossing engine boundaries.

Suggested output should contain:

``` text
X
engine_id
start_cycle
end_cycle
```

## `models/autoencoder.py`

Responsibilities:

-   baseline AE architecture,
-   encoder,
-   decoder,
-   reconstruction output.

## `models/regime_encoder.py`

Responsibilities:

-   map the three operational settings to a regime vector.

## `models/conditioned_autoencoder.py`

Responsibilities:

-   combine sensor representation and regime representation,
-   expose latent vector,
-   reconstruct the input.

## `anomaly/reconstruction.py`

Responsibilities:

-   total reconstruction error,
-   per-time-step error,
-   per-channel error.

## `anomaly/fixed_threshold.py`

Responsibilities:

-   establish the baseline detector.

## `anomaly/dynamic_threshold.py`

Responsibilities:

-   heuristic dynamic threshold,
-   precursor to conformal threshold.

## `anomaly/conformal.py`

Responsibilities:

-   calibration scores,
-   conformal quantile,
-   dynamic/regime-aware threshold,
-   coverage-related diagnostics.

## `anomaly/attribution.py`

Responsibilities:

-   channel-level contributions,
-   ranking,
-   explanation object.

## `evaluation/classification_metrics.py`

Responsibilities:

-   Precision,
-   Recall,
-   F1,
-   ROC-AUC,
-   false-alarm metrics.

## `evaluation/rul_metrics.py`

Responsibilities:

-   MAE,
-   RMSE.

## `evaluation/nasa_score.py`

Responsibilities:

-   NASA scoring function only.

Keep this function isolated and unit-tested.

## `evaluation/lead_time.py`

Responsibilities:

-   first valid alert,
-   failure cycle,
-   lead time,
-   missed detection handling.

------------------------------------------------------------------------

# 20. Configuration philosophy

Avoid:

``` python
WINDOW_SIZE = 30
LATENT_DIM = 16
EPOCHS = 100
```

spread across multiple files.

Prefer configuration:

``` yaml
dataset:
  name: CMAPSS
  subset: FD001

features:
  settings:
    - setting_1
    - setting_2
    - setting_3
  sensors:
    all: true

window:
  size: 30
  stride: 1

model:
  latent_dim: 16

training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.001
  seed: 42

threshold:
  method: fixed
```

The exact values above are examples, not established project values.
Experimental values must be logged.

------------------------------------------------------------------------

# 21. Experiment tracking

Every experiment should record:

``` text
experiment_id
timestamp
dataset
train/validation/test split
healthy-region definition
window size
stride
normalization method
model architecture
latent dimension
optimizer
learning rate
epochs
batch size
random seed
threshold method
threshold parameters
metrics
checkpoint path
git commit
```

A result without its configuration is not considered reproducible.

------------------------------------------------------------------------

# 22. Recommended experiment naming

Use names such as:

``` text
fd001_ae_fixed_v001
fd001_ae_dynamic_v001
fd002_ae_fixed_v001
fd002_regime_concat_v001
fd002_regime_film_v001
fd002_regime_conformal_v001
```

Avoid vague names such as:

``` text
final_model
best_model
new_model
test2
latest
```

------------------------------------------------------------------------

# 23. Visualization requirements

## Dataset EDA

At minimum:

-   sensor distributions,
-   sensor correlation heatmap,
-   sensor trends over cycle,
-   operating-setting distributions,
-   engine trajectory examples,
-   healthy vs degraded comparisons.

## Anomaly detection

At minimum:

-   reconstruction error vs cycle,
-   threshold vs cycle,
-   actual failure cycle,
-   alert point,
-   top sensor contributions.

Conceptual plot:

``` text
reconstruction error
 ^
 |                         /\  /\ 
 |                ________/  \/  \____
 |--------------- dynamic threshold ----
 |       /
 |______/
 +--------------------------------------> cycle
                         ↑
                       alert
                              ↑
                           failure
```

## RUL

At minimum:

-   predicted vs true RUL,
-   residual distribution,
-   RUL error by engine,
-   optional early/late error analysis.

------------------------------------------------------------------------

# 24. 12-week overall project plan

The Review-1 PPT gives the following high-level 12-week structure:

### Phase 1 --- Weeks 1--2

**Literature Review & Dataset Exploration**

### Phase 2 --- Weeks 3--4

**Data Preprocessing Pipeline**

### Phase 3 --- Weeks 5--6

**Baseline Model Development**

### Phase 4 --- Weeks 7--9

**Autoencoder Design & Training**

### Phase 5 --- Weeks 10--11

**Evaluation & Hyperparameter Tuning**

### Phase 6 --- Week 12

**Final Report & Presentation**

The newer 9-day plan is an accelerated checkpoint inside this broader
plan.

------------------------------------------------------------------------

# 25. 9-day execution plan --- 27 Aug to 5 Sep 2026

## Day 1 --- 27 Aug

### Aryaman

-   Read arXiv 2601.10269.
-   Read dynamic-threshold literature.
-   Draft novelty differentiation.

### Akriti

-   Shared repo.
-   Environment.
-   C-MAPSS FD001--FD004 organization.
-   Folder structure.

### Nethra

-   Patent-style claim skeleton.
-   Start adjacent patent search.

------------------------------------------------------------------------

## Day 2 --- 28 Aug

### Aryaman

-   Finish literature/gap table with citations.

### Akriti

-   EDA:
    -   sensor distributions,
    -   healthy/degraded cycles,
    -   operating-condition counts.

### Nethra

-   Continue patent landscape.
-   Write novelty/IP strategy section.

------------------------------------------------------------------------

## Day 3 --- 29 Aug

### Aryaman

-   Sliding-window generator.
-   Train/validation/test split logic.

### Akriti

-   Normalization pipeline.
-   Per-condition z-score initial approach.
-   Regime embedding input stub.

### Nethra

-   Piecewise-linear/capped RUL generator.

------------------------------------------------------------------------

## Day 4 --- 30 Aug

### Aryaman

-   Unit-test preprocessing on FD001.
-   Verify shapes and leakage-free splits.

### Akriti

-   Cross-check on FD002.
-   Validate six-condition handling.

### Nethra

-   Baseline AE architecture/design document.

------------------------------------------------------------------------

## Day 5 --- 31 Aug

### Aryaman

-   Implement/train vanilla AE on FD001 healthy cycles.
-   Reconstruction error.
-   Fixed-threshold baseline.

### Akriti

-   Support reconstruction-error/fixed-threshold implementation.

### Nethra

-   Evaluation harness:
    -   Precision,
    -   Recall,
    -   F1,
    -   ROC-AUC,
    -   RMSE,
    -   MAE,
    -   NASA score.

------------------------------------------------------------------------

## Day 6 --- 1 Sep

### Aryaman

-   Tune AE.
-   Log experiments.

### Akriti

-   Implement first dynamic-threshold precursor.

### Nethra

-   Run baseline evaluation.
-   Tables.
-   Reconstruction-error plots.

------------------------------------------------------------------------

## Day 7 --- 2 Sep

### Aryaman

-   Integrate dynamic threshold.
-   Compare false alarms vs fixed threshold.

### Akriti

-   Regime embedding:
    -   concatenation or small MLP,
    -   three operational settings.

### Nethra

-   Start Review-2 report:
    -   problem,
    -   gap table,
    -   novelty,
    -   dataset,
    -   initial results.

------------------------------------------------------------------------

## Day 8 --- 3 Sep

### Aryaman

-   Bug-fix.
-   Re-run experiments.
-   Finalize plots/tables.

### Akriti

-   Cross-validate regime embedding does not break FD001.

### Nethra

-   Finish report draft.
-   Build 6--8 slide Review-2 deck.

------------------------------------------------------------------------

## Day 9 --- 4 Sep

### Full team

-   Rehearse.
-   Verify every number is reproducible from code.
-   Proofread report/deck.
-   Prepare demo/notebook.
-   Prepare speaking parts.
-   Prepare novelty/patentability Q&A.

## 5 Sep

**Submission / Review-2**

All three present.

------------------------------------------------------------------------

# 26. What must be complete by 5 September

The action-plan document defines the checkpoint as:

1.  Finalized differentiated novelty statement.
2.  Patent-style claim skeleton.
3.  Gap table with 15+ cited papers.
4.  Leak-free preprocessing pipeline validated on at least FD001 and
    FD002.
5.  Trained baseline autoencoder.
6.  Fixed-threshold detection results.
7.  Precision/Recall/F1/ROC-AUC results.
8.  RMSE/MAE results where RUL is evaluated.
9.  First dynamic-threshold module.
10. Measurable comparison against fixed threshold.
11. Report/deck/demo materials.

The following are **not required to be fully finished by this
checkpoint** according to the action plan:

-   fully wired regime conditioning,
-   conformal calibration,
-   sensor-attribution map,
-   full cross-FD001--FD004 generalization.

Those remain part of the larger 12-week engineering/research plan.

------------------------------------------------------------------------

# 27. Definition of done for the core ML pipeline

A component is considered done only when:

### Data

-   [ ] Raw files are unchanged.
-   [ ] Columns are standardized.
-   [ ] Engine IDs and cycles are preserved.
-   [ ] Split is engine-level.
-   [ ] No train/test engine leakage.
-   [ ] Normalization is fit only on allowed data.
-   [ ] Window generation never crosses engine boundaries.

### Baseline AE

-   [ ] Model trains successfully.
-   [ ] Reconstruction loss decreases.
-   [ ] Reconstruction outputs have expected shape.
-   [ ] Healthy reconstruction error is characterized.
-   [ ] Fixed threshold is calibrated without test labels.
-   [ ] Event-level alerts can be generated.

### Dynamic threshold

-   [ ] Threshold changes appropriately with context.
-   [ ] Threshold logic is deterministic/reproducible.
-   [ ] False-alarm rate is measured.
-   [ ] Comparison with fixed threshold uses identical evaluation data.

### Regime conditioning

-   [ ] Three operational settings are encoded.
-   [ ] Regime representation can be inspected.
-   [ ] Model accepts regime information.
-   [ ] FD002 can be processed without special-case hacks.

### Conformal component

-   [ ] Calibration set is separated from training.
-   [ ] Nonconformity scores are computed correctly.
-   [ ] Quantile/calibration logic is unit-tested.
-   [ ] Threshold can vary with regime as designed.
-   [ ] Calibration diagnostics are reported.
-   [ ] Statistical claims are worded conservatively.

### Attribution

-   [ ] Per-channel errors are retained.
-   [ ] Top contributing sensors can be ranked.
-   [ ] Attribution is tied to the alert/window.
-   [ ] Dashboard/plots can display attribution.

------------------------------------------------------------------------

# 28. Common failure modes the coding assistant should prevent

## Failure 1 --- random row splitting

Bad:

``` python
train_test_split(df)
```

on individual rows from the same engine.

Why bad:

The model may see nearly identical neighboring cycles from the same
trajectory in train and test.

Preferred:

Split by engine ID before windowing.

------------------------------------------------------------------------

## Failure 2 --- fitting the scaler on all data

Bad:

``` python
scaler.fit(all_data)
```

Why bad:

Validation/test information enters preprocessing.

Preferred:

``` python
scaler.fit(train_data)
```

then transform validation/test.

------------------------------------------------------------------------

## Failure 3 --- threshold tuned on test data

Bad:

``` python
for threshold in thresholds:
    evaluate_on_test()
choose_best()
```

Why bad:

This leaks test information.

Preferred:

Tune/calibrate threshold on a training/calibration/validation protocol
and evaluate once on test.

------------------------------------------------------------------------

## Failure 4 --- treating operational settings as ordinary sensors

This can erase the very operating-regime information the novelty
mechanism is designed to exploit.

Keep settings explicitly represented.

------------------------------------------------------------------------

## Failure 5 --- only reporting average reconstruction loss

Average reconstruction loss does not demonstrate early-warning
usefulness.

Also inspect:

-   reconstruction-error trajectory,
-   threshold crossing,
-   false alarms,
-   lead time,
-   sensor contributions.

------------------------------------------------------------------------

## Failure 6 --- calling a heuristic threshold "conformal"

A percentile or rolling standard-deviation threshold is not
automatically conformal.

Use precise terminology.

------------------------------------------------------------------------

## Failure 7 --- claiming statistical guarantees without calibration validation

Do not write:

> "The system guarantees 5% false alarms."

unless the exact assumptions, calibration procedure, and
empirical/statistical validation support that statement.

Prefer:

> "The conformal calibration is designed to provide the specified
> coverage under its stated assumptions."

------------------------------------------------------------------------

## Failure 8 --- overcomplicating the first model

Do not start with:

``` text
Transformer + BiLSTM + attention + VAE + GAN + conformal prediction
```

The project needs an attributable research progression.

Start:

``` text
vanilla AE
→ dynamic threshold
→ regime conditioning
→ conformal calibration
→ attribution
→ optional joint RUL
```

------------------------------------------------------------------------

# 29. Git and collaboration rules

## Branching

Recommended:

``` text
main
develop
feature/data-pipeline
feature/baseline-ae
feature/dynamic-threshold
feature/regime-conditioning
feature/conformal
feature/evaluation
feature/dashboard
```

## Commit naming

Prefer:

``` text
feat: add engine-level split
feat: implement baseline autoencoder
feat: add fixed threshold detector
feat: add dynamic threshold precursor
feat: add regime embedding
feat: add conformal calibration
fix: prevent window crossing engine boundaries
test: add RUL metric tests
docs: update novelty methodology
```

## Every ML experiment commit should make it possible to answer:

-   What changed?
-   Why?
-   Which dataset?
-   Which configuration?
-   Which metrics?
-   What was the previous baseline?

------------------------------------------------------------------------

# 30. Literature context

The Review-1 deck lists these core references:

1.  Al Bataineh, A., Mairaj, A., & Kaur, D. (2020). *Autoencoder based
    semi-supervised anomaly detection in turbofan engines.*
    International Journal of Advanced Computer Science and Applications,
    11(11).
2.  Muneer, A., Taib, S. M., Naseer, S., Ali, R. F., & Aziz, I. A.
    (2021). *Deep-Learning Based Prognosis Approach for Remaining Useful
    Life Prediction of Turbofan Engine.* Symmetry, 13(10).
3.  Deng, S. et al. (2021). *Remaining Useful Life Estimation of
    Aircraft Engines Using a Joint Deep Learning Model Based on TCNN and
    Transformer.* Computational Intelligence and Neuroscience.
4.  Chao, M. A., Kulkarni, C., Goebel, K., & Fink, O. (2021). *Aircraft
    engine run-to-failure dataset under real flight conditions for
    prognostics and diagnostics.* Data, 6(1). This is associated with
    N-CMAPSS rather than the C-MAPSS dataset used as the core project
    dataset.
5.  Li, X., Zhang, W., Ma, H., Luo, Z., & Li, X. (2021). *Degradation
    alignment in remaining useful life prediction using deep
    cycle-consistent learning.* IEEE Transactions on Neural Networks and
    Learning Systems, 33(10), 5480--5491.
6.  Zhang, W. et al. (2021). *An integrated deep multiscale feature
    fusion network for aeroengine remaining useful life prediction with
    multisensor data.* Knowledge-Based Systems, 235, 107652.

The action-plan PDF additionally highlights recent methodological
context including:

-   a 2025 multi-model comparative study involving Autoencoders, LSTMs
    and Gaussian Process Regression,
-   a 2025 convolutional autoencoder + attention LSTM model,
-   a 2025 Transformer-LSTM model with Bayesian optimization and
    maintenance scheduling,
-   a 2026 BiGRU/self-attention + stacked denoising autoencoder
    approach,
-   a 2024 asymmetric RUL loss based on prediction-vector-angle ideas,
-   a 2024 dynamic-threshold denoising-autoencoder method,
-   a 2025 uncertainty-informed dynamic thresholding method,
-   a 2025 benchmark on uncertainty quantification for deep-learning
    prognostics,
-   a 2026 arXiv paper on unsupervised LSTM autoencoders for early fault
    detection on C-MAPSS,
-   a 2025 broad review of AI methods for anomaly detection and RUL.

Before citing any paper in the final academic report, verify the full
bibliographic metadata and original publisher/source.

------------------------------------------------------------------------

# 31. Closest prior-art concern

The action plan specifically highlights a 2026 arXiv paper on early
fault detection using unsupervised LSTM autoencoders on C-MAPSS.

Its described approach includes:

-   removing operating-condition effects through regression-based
    normalization,
-   LSTM autoencoding,
-   an adaptive data-driven threshold.

Therefore, the final project should clearly differentiate:

``` text
Prior-art-like:
operating-condition normalization
+
LSTM AE
+
adaptive threshold
```

from the proposed direction:

``` text
regime-conditioned AE
+
conformal-calibrated dynamic threshold
+
sensor-level attribution
```

Do not describe "dynamic threshold on C-MAPSS" alone as the project's
novelty.

------------------------------------------------------------------------

# 32. Patent/IP context

The action-plan document states that the proposed novelty should be
framed as a technical system/method rather than merely "using deep
learning."

Its patent-style skeleton is conceptually:

1.  Receive multivariate sensor + operating-condition data.
2.  Generate a regime embedding from operating-condition data.
3.  Condition autoencoder encoding/decoding using the regime embedding.
4.  Compute reconstruction error.
5.  Compute a dynamic detection threshold using conformal calibration.
6.  Decompose reconstruction error into sensor-channel contributions.
7.  Generate an early-failure alert with sensor-attribution information.

The action plan also identifies adjacent patents, including:

-   a turbomachine monitoring system using anomaly detection corrected
    by a wear factor,
-   aircraft-engine operation methods involving dynamic/adjustable
    thresholds,
-   health-monitoring systems for complex systems.

The team should perform an actual patent search before making any
definitive novelty claim.

## Public disclosure warning

The action plan explicitly recommends keeping dated lab notes/Git
history and contacting the institution's IP/Innovation cell before
public disclosure of the specific novelty.

The coding assistant should therefore:

-   avoid suggesting that the team publish the complete novel mechanism
    publicly before the IP strategy is settled,
-   keep the repository private if the team is pursuing the stated
    patent path,
-   preserve commit history,
-   preserve experiment dates,
-   preserve architecture decisions.

This is project/IP workflow guidance from the team's action plan, not
legal advice.

------------------------------------------------------------------------

# 33. What the final system should demonstrate

The ideal end-to-end demonstration is:

``` text
Select engine
    ↓
show sensor/operational history
    ↓
run model
    ↓
calculate reconstruction error
    ↓
show dynamic threshold
    ↓
detect abnormal degradation
    ↓
show alert cycle
    ↓
show estimated lead time
    ↓
show top contributing sensors
    ↓
optionally show RUL estimate
```

A strong dashboard should allow:

-   engine selection,
-   cycle timeline,
-   reconstruction-error curve,
-   threshold curve,
-   alert marker,
-   actual failure marker where evaluation ground truth is available,
-   sensor contribution ranking,
-   RUL prediction if implemented.

------------------------------------------------------------------------

# 34. Suggested AI-assistant prompts by role

## If Aryaman says:

> "I am Aryaman. Implement the reconstruction error module."

The assistant should: - inspect the repository structure, - identify the
model output shape, - preserve engine/window metadata, - implement
scalar and per-channel reconstruction error, - add unit tests, - avoid
changing the model architecture unnecessarily.

## If Aryaman says:

> "I am Aryaman. Fix this data leakage issue."

The assistant should: - inspect split logic, - ensure splitting occurs
by engine, - ensure scalers/thresholds are fit only on permitted data, -
identify exactly where future information enters, - propose the smallest
safe fix, - add a regression test.

## If Aryaman says:

> "I am Aryaman. Implement regime conditioning."

The assistant should: - preserve the baseline AE, - add a regime encoder
for the three settings, - start with concatenation or a simple
conditioning mechanism, - ensure tensor shapes are explicit, - make
conditioning optional through configuration, - compare conditioned vs
unconditioned versions.

## If Akriti says:

> "I am Akriti. Fix preprocessing."

The assistant should focus on: - raw data loading, - schema, -
cleaning, - normalization, - engine-level splits, - windows, -
FD001--FD004 compatibility, - regime inputs.

## If Nethra says:

> "I am Nethra. Add evaluation."

The assistant should focus on: - event-level anomaly metrics, - RUL
metrics, - NASA score, - lead time, - tables, - reproducible plots, - no
test-set threshold tuning.

------------------------------------------------------------------------

# 35. Minimum viable implementation order

If the team is stuck, follow this order exactly.

### Stage 1 --- Data

``` text
load → schema → split → normalize → window
```

### Stage 2 --- Baseline

``` text
healthy windows → AE → reconstruction error → fixed threshold
```

### Stage 3 --- Evaluation

``` text
alerts → Precision/Recall/F1/ROC-AUC
failure cycle → lead time
RUL → MAE/RMSE/NASA
```

### Stage 4 --- Dynamic threshold

``` text
fixed threshold
      ↓
adaptive threshold
```

### Stage 5 --- Regime conditioning

``` text
settings → regime embedding → AE
```

### Stage 6 --- Conformal calibration

``` text
healthy calibration → nonconformity scores → calibrated threshold
```

### Stage 7 --- Sensor attribution

``` text
reconstruction tensor → per-channel error → ranked sensors
```

### Stage 8 --- Optional RUL joint head

``` text
shared latent → RUL head → asymmetric loss
```

### Stage 9 --- Deployment/dashboard

``` text
trained model → inference pipeline → dashboard
```

------------------------------------------------------------------------

# 36. Priority matrix

  Component                     Priority   Required for Sep 5?   Owner
  ----------------------------- ---------- --------------------- ----------------
  C-MAPSS ingestion             P0         Yes                   Akriti
  Engine-level split            P0         Yes                   Aryaman
  Normalization                 P0         Yes                   Akriti
  Sliding windows               P0         Yes                   Aryaman
  RUL labels                    P1         Yes for RUL work      Nethra
  Baseline AE                   P0         Yes                   Aryaman
  Reconstruction error          P0         Yes                   Aryaman/Akriti
  Fixed threshold               P0         Yes                   Aryaman
  Evaluation harness            P0         Yes                   Nethra
  Dynamic threshold precursor   P0         Yes                   Akriti/Aryaman
  Regime embedding              P1         Prototype             Akriti/Aryaman
  Conformal threshold           P1         No                    Aryaman
  Sensor attribution            P1         No                    Nethra/Aryaman
  Joint RUL head                P2         No                    Aryaman
  Cross-FD generalization       P1         No                    All
  Dashboard                     P2         Deliverable           Nethra
  Patent/IP documentation       P0         Yes                   Nethra

------------------------------------------------------------------------

# 37. Practical coding standards

Use:

-   Python 3.x,
-   type hints where useful,
-   docstrings for public functions/classes,
-   deterministic seeds,
-   structured logging,
-   configuration-driven experiments,
-   pytest tests,
-   NumPy/Pandas for data processing,
-   the selected deep-learning framework consistently throughout the
    project.

Avoid:

-   global mutable state,
-   hidden data transformations,
-   duplicated preprocessing code,
-   notebook-only model definitions,
-   unexplained magic numbers,
-   test-set tuning,
-   hard-coded paths,
-   hard-coded FD001 assumptions.

------------------------------------------------------------------------

# 38. When generating code, the assistant should ask/verify these first

Before implementing a non-trivial model component, check:

1.  Which FD dataset?
2.  What is the current dataframe schema?
3.  What are the train/validation/calibration/test boundaries?
4.  What is the window size?
5.  Are operational settings included as model inputs?
6.  What normalization is being used?
7.  Is this baseline or proposed model?
8.  What is the expected tensor shape?
9.  What threshold/calibration method is currently active?
10. Which metrics are being used?
11. Where should the output/checkpoint be saved?
12. Is the requested code changing a research assumption?

If enough information already exists in `AI_CONTEXT.md`, do not
repeatedly ask the user for it.

------------------------------------------------------------------------

# 39. Reviewer's likely questions

The assistant should help the team prepare for:

### Why C-MAPSS?

Because it provides standardized simulated turbofan degradation
trajectories suitable for predictive-maintenance benchmarking.

### Why autoencoders?

They can learn reconstruction behavior from healthy data and provide an
unsupervised/semi-supervised anomaly score when failures are difficult
to label.

### Why not use a standard classifier?

The intended early-warning setting emphasizes learning normal behavior
and identifying deviation rather than requiring complete labeled fault
classes.

### Why is a fixed threshold insufficient?

Operating conditions can change the reconstruction-error distribution,
creating false alarms.

### Why regime conditioning?

FD002--FD004 include multiple operating conditions; a shared model
should distinguish operational variation from actual degradation.

### Why conformal calibration?

It provides a principled calibration mechanism for anomaly thresholds
and can support statistical coverage statements under its assumptions.

### Why sensor attribution?

Maintenance decisions need actionable information about which channels
are driving the anomaly.

### What is actually novel?

Not any single ingredient. The working novelty hypothesis is the
combination of: - regime-conditioned AE, - conformal-calibrated dynamic
thresholding, - sensor-level attribution, for turbofan early-failure
detection.

### Is the combination definitely patentable?

No claim should be made without a proper prior-art/patent search and
institutional/IP review.

### Why not implement everything immediately?

Because the project needs an interpretable progression:

``` text
baseline → improvement → novelty component → final integrated system
```

------------------------------------------------------------------------

# 40. Final AI coding assistant checklist

Before modifying code, verify:

-   [ ] I know which team member is asking.
-   [ ] I know which component they own.
-   [ ] I know whether this is baseline or proposed methodology.
-   [ ] I will not introduce leakage.
-   [ ] I will preserve engine/cycle metadata.
-   [ ] I will keep experiments reproducible.
-   [ ] I will not hard-code dataset-specific assumptions unnecessarily.
-   [ ] I will add tests for important logic.
-   [ ] I will log configuration and results.
-   [ ] I will not claim statistical guarantees without proper
    calibration.
-   [ ] I will not claim novelty/patentability as established fact.
-   [ ] I will not unnecessarily replace the project's chosen
    architecture.
-   [ ] I will keep modules reusable outside notebooks.
-   [ ] I will explain any research-methodology change before
    implementing it.

------------------------------------------------------------------------

# 41. Canonical project flow

The complete intended system can be remembered as:

``` text
NASA C-MAPSS FD001–FD004
          │
          ▼
   Data ingestion
          │
          ▼
 Engine-level splitting
          │
          ▼
 Healthy-region selection
          │
          ▼
   Normalization
          │
          ▼
 Sliding-window sequences
          │
          ├───────────────────────┐
          │                       │
          ▼                       ▼
 Sensor representation      3 operating settings
          │                       │
          │                       ▼
          │                Regime embedding
          │                       │
          └───────────┬───────────┘
                      ▼
          Regime-conditioned AE
                      │
             ┌────────┴────────┐
             ▼                 ▼
     Reconstruction        Shared latent
        error                  │
             │                 └──> Optional RUL head
             ▼
    Conformal calibration
             │
             ▼
 Dynamic regime-aware threshold
             │
             ▼
       Early alert
             │
       ┌─────┴───────────┐
       ▼                 ▼
 Lead-time metric   Sensor attribution
       │                 │
       └────────┬────────┘
                ▼
        Health dashboard
                │
                ▼
      Predictive maintenance
```

------------------------------------------------------------------------

# 42. One-paragraph project summary for an AI model

This is a three-person academic predictive-maintenance project using
NASA C-MAPSS FD001--FD004. The original Review-1 proposal is a deep
autoencoder trained on healthy-region multivariate sensor windows, with
reconstruction error used for early-failure anomaly detection and a
separate RUL evaluation path. The literature/action-plan review
concludes that this baseline is mainstream, so the project is being
upgraded toward a regime-conditioned autoencoder using the three
operational settings, a conformal-calibrated dynamic
reconstruction-error threshold, and per-sensor anomaly attribution. A
joint latent-space RUL head with an asymmetric loss is a stretch goal.
The immediate 27 Aug--5 Sep 2026 checkpoint prioritizes a leakage-free
preprocessing pipeline on FD001/FD002, a vanilla AE, fixed-threshold
baseline metrics, and a first dynamic-threshold improvement. Aryaman
owns the core ML/integration pipeline, Akriti owns
data/EDA/normalization/regime-input work, and Nethra owns evaluation,
literature/IP, reporting, and dashboard/demo work. The coding assistant
must preserve reproducibility, prevent data leakage, distinguish
baseline from proposed components, and avoid unsupported
novelty/patentability claims.

------------------------------------------------------------------------

# 43. Source provenance

This context document was synthesized from the team's two uploaded
project materials:

1.  **Predictive_Maintenance_Aircraft_Engines_Review1.pptx** ---
    Review-1 problem statement, workflow, dataset, metrics, outcomes,
    timeline, and initial references.
2.  **Novelty_Gaps_LitReview_ActionPlan.pdf** --- current research-gap
    analysis, proposed novelty, patent-style claim skeleton, recent
    literature context, and 27 Aug--5 Sep 2026 execution plan.

Where this document adds repository organization, coding conventions,
module boundaries, experiment naming, or assistant behavior rules, those
are **recommended engineering structures created for this context
file**, not claims that those exact structures were present in the
source documents.
