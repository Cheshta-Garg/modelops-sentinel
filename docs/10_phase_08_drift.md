# Stage 8: Statistical Drift Monitoring & Distribution Shift Detection

## 1. Architectural Positioning & Philosophy

In **ModelOps Sentinel**, **Data Drift Monitoring** is a core operational capability designed to continuously protect production machine learning models from silent degradation caused by real-world distribution changes.

Machine learning models are trained under the foundational assumption that production inference inputs are independently and identically distributed ($i.i.d.$) relative to the training data. In production environments, this assumption inevitably breaks down due to macroeconomic volatility, changing user demographics, sensor wear, upstream data schema shifts, and seasonal trends.

### Lightweight ModelOps vs. Generic MLOps
Generic MLOps platforms often introduce massive external monitoring clusters (e.g., heavyweight Spark jobs, complex third-party SaaS agents, or monolithic orchestrators) that add immense operational overhead and latency. 

**ModelOps Sentinel** takes a production-grade, lightweight approach:
1. **Reference Statistics are Model-Version Artifacts**: When a model version is registered or promoted, its reference baseline statistics (quantiles, deciles, mean, standard deviation, and sample reservoirs) are computed directly from its training/validation dataset and persisted as a first-class `reference_stats` entity attached to the immutable `ModelVersion`.
2. **Decoupled Asynchronous Evaluation**: The online inference path remains blazing fast (< 2ms) and is never blocked by distribution computations. Inferences are logged, and background workers (`evaluate_drift_task`) periodically evaluate incoming feature windows against the stored reference baseline.
3. **Structured Drift Events**: Whenever feature distributions deviate beyond calibrated thresholds, granular `drift_events` are permanently recorded with the exact feature, test method, score, threshold, severity, and timestamp.
4. **Actionable API**: Engineers and automated deployment controllers can query `GET /api/v1/models/{id}/drift` at any time to inspect real-time drift state, evaluate canary promotions, or trigger automated rollbacks.

```
+----------------------------------------------------------------------------------------------------+
|                                    Model Registration & Staging                                     |
|                                                                                                    |
|   Training / Validation Dataset ------> extract_reference_statistics()                            |
|                                                    |                                               |
|                                                    v                                               |
|                                  ReferenceStats (ModelVersion Metadata)                            |
+----------------------------------------------------------------------------------------------------+
                                                     |
                                                     | (Golden Baseline Profile)
                                                     v
+-------------------------------------+   +----------------------------------------------------------+
|          Online Traffic             |   |                 Background Drift Worker                  |
|                                     |   |                                                          |
| Client Request                      |   | Worker periodically evaluates recent inference window:   |
|       |                             |   |                                                          |
|       v                             |   | 1. Pulls reference profile (ReferenceStats)              |
| Fast InferenceEngine (< 2ms)        |   | 2. Pulls recent production features (InferenceRequests)  |
|       |                             |   | 3. Computes:                                             |
|       +--> Log InferenceRequest     |   |    * PSI (Population Stability Index)                    |
|            (features, pred, latency)|   |    * Two-Sample Kolmogorov-Smirnov (KS) Test             |
|                                     |   | 4. Compares scores against calibrated thresholds         |
|                                     |   | 5. Persists drift_events (feature, method, score, etc.)  |
+-------------------------------------+   +----------------------------------------------------------+
                                                     |
                                                     v
                                      +------------------------------+
                                      |   GET /api/v1/models/{id}/drift|
                                      |                               |
                                      | Returns real-time health,     |
                                      | summary & recent drift events |
                                      +------------------------------+
```

---

## 2. Core Concepts Explained

### 1. Data Drift (Covariate Shift)
**Data drift** (specifically **covariate shift**) occurs when the marginal probability distribution of input features $P(X)$ changes over time between the training/reference period and real-time production inference, while the conditional ground-truth relationship $P(Y \mid X)$ remains unchanged.

Mathematically:
$$P_{production}(X) \neq P_{reference}(X) \quad \text{while} \quad P_{production}(Y \mid X) = P_{reference}(Y \mid X)$$

**Real-World Examples**:
- **Financial Credit Risk**: High inflation or interest rate hikes shift applicant debt-to-income distributions significantly upward and lower average credit scores compared to historical training data.
- **Industrial Process Control (e.g., SWaT)**: Sensor calibration drift causes tank water levels (`LIT101`) or pipe flow rates (`FIT101`) to register higher baseline readings under normal physical conditions.
- **E-Commerce Recommenders**: Seasonal holidays (e.g., Black Friday) shift customer purchase frequency and browsing dwell time distributions.

---

### 2. Reference Distribution
The **reference distribution** represents the "golden baseline" data profile captured from the model's training, validation, or benchmark verification set. 

Rather than storing massive multi-gigabyte raw training datasets in relational storage, ModelOps Sentinel extracts a compact, mathematically sufficient statistical profile:
- **Decile Quantiles**: 10 equally-spaced quantiles ($0\%, 10\%, \dots, 100\%$) that define bin boundaries for non-parametric binning.
- **Parametric Moments**: Mean, standard deviation, minimum, maximum, and median ($p_{25}, p_{50}, p_{75}$).
- **Sample Reservoir**: A deterministic uniform subsample (up to 500 representative scalar values per continuous feature) enabling exact non-parametric empirical distribution comparisons (KS tests).

---

### 3. Population Stability Index (PSI)
The **Population Stability Index (PSI)** is an industry-standard metric originated in financial risk modeling to measure how much a variable has shifted between a reference population and a production population.

#### Formula:
$$PSI = \sum_{i=1}^{B} \left( \text{Actual}_i\% - \text{Expected}_i\% \right) \times \ln\left( \frac{\text{Actual}_i\%}{\text{Expected}_i\%} \right)$$

Where:
- $B$ is the number of bins (typically 10 decile bins derived from the reference distribution).
- $\text{Expected}_i\%$ is the proportion of reference observations falling in bin $i$.
- $\text{Actual}_i\%$ is the proportion of production inference observations falling in bin $i$.

#### Intuition:
Notice that each bin's contribution is the product of two terms:
1. Difference in proportions: $(\text{Actual}_i\% - \text{Expected}_i\%)$
2. Log ratio of proportions: $\ln(\text{Actual}_i\% / \text{Expected}_i\%)$

If $\text{Actual}_i\% > \text{Expected}_i\%$, both terms are positive, yielding a positive contribution. If $\text{Actual}_i\% < \text{Expected}_i\%$, both terms are negative, their product is positive, and it again adds a positive contribution. If proportions are identical, the log term is $\ln(1) = 0$, yielding a contribution of 0. Thus, $PSI \ge 0$, and higher values represent greater divergence.

#### Standard Interpretation Thresholds:
| PSI Score Range | Severity | Operational Interpretation |
|---|---|---|
| $PSI < 0.10$ | **NONE** | No significant distribution change. Model environment is stable. |
| $0.10 \le PSI < 0.25$ | **MEDIUM** | Moderate distribution shift. Recommend observation and feature inspection. |
| $PSI \ge 0.25$ | **HIGH** | Significant distribution shift. Flag drift event, trigger alerts, consider retraining or canary rollback. |

---

### 4. Two-Sample Kolmogorov-Smirnov (KS) Test
The **Two-Sample Kolmogorov-Smirnov (KS) Test** is a non-parametric statistical hypothesis test that compares the continuous cumulative distribution functions (CDFs) of two independent samples.

#### Formula:
$$D = \sup_{x} \left| F_{reference}(x) - F_{production}(x) \right|$$

Where:
- $F_{reference}(x)$ is the empirical cumulative distribution function (ECDF) of the reference baseline.
- $F_{production}(x)$ is the ECDF of incoming production inferences.
- $D$ is the **KS statistic**, representing the maximum vertical distance between the two cumulative distribution curves.
- The test produces a $p\text{-value}$ assessing the Null Hypothesis:
  - $H_0$: The reference and production samples are drawn from the exact same continuous distribution.
  - $H_1$: The two samples are drawn from different distributions.

#### When to use KS Test vs. PSI:
- **KS Test**: Highly sensitive to subtle shifts in distribution shape, median shifts, or changes in variance for **strictly continuous numerical features**. Does not require pre-partitioning into artificial bins.
- **PSI**: Highly resilient, industry-standard metric that can be applied to both binned continuous features and discrete categorical variables.

---

### 5. Threshold Selection & Severity Levels
Threshold selection balances **sensitivity to true degradation** against **alert fatigue**.

In ModelOps Sentinel, a feature is officially classified as drifted when its calculated score exceeds the configured operational threshold:
- **PSI Drift**: Flagged if $PSI \ge 0.25$.
- **KS Test Drift**: Flagged if $p\text{-value} < 0.05$ (statistical significance) **AND** $D \ge 0.20$ (practical effect size).

The platform assigns a four-tier severity:
- `NONE`: Score below warning cutoff ($PSI < 0.10$, or $D < 0.10$ / $p \ge 0.05$).
- `LOW` / `MEDIUM`: Moderate shift ($0.10 \le PSI < 0.25$, or $0.10 \le D < 0.25$).
- `HIGH`: Critical shift ($PSI \ge 0.25$, or $D \ge 0.25$ with $p < 0.001$).

---

### 6. False Positives & Small-Sample Sparsity
A critical challenge in production drift monitoring is **false positive alerts** caused by small sample sizes and empty bins.

#### The Empty Bin Hazard in PSI:
If a small production window (e.g., $N=40$ inferences) is divided into 10 decile bins, several bins will naturally contain zero observations ($\text{Actual}_i\% = 0$). 
In naive implementations:
- If $0$ is simply clamped to an arbitrary small number like $\epsilon = 10^{-4}$, the log ratio $\ln(10^{-4} / 0.10) \approx -6.91$ creates an artificial spike of $(10^{-4} - 0.10) \times (-6.91) \approx +0.69$ in a single bin!
- This causes the calculated PSI to falsely spike above $0.35$ (HIGH drift) even when the production sample was drawn from the *exact identical distribution* as the training data.

#### ModelOps Sentinel Solutions:
1. **Adaptive Bin Sizing**: If the production sample size $N < 80$, the bin count dynamically scales down using $\min(\text{num\_bins}, \max(4, N // 8))$ to ensure adequate sample density across all partitions.
2. **Laplace Smoothing (Pseudo-Counts)**: Proportions are smoothed using:
   $$P_i = \frac{count_i + 1}{N + B}$$
   This mathematically prevents zero divisions while maintaining true divergence characteristics.
3. **Minimum Sample Guardrails**: If fewer than 5 production inferences are available, evaluation is gracefully bypassed rather than generating spurious alerts.

---

### 7. Why Drift Does Not Automatically Mean Model Failure
> [!IMPORTANT]
> **Data drift is an early warning indicator, NOT proof of catastrophic model failure.**

A common rookie mistake is immediately taking a model offline the instant any feature shows drift. In production, feature drift does not always cause prediction degradation for several key reasons:

1. **Uninformative / Low-Attribution Features**:
   - Modern tabular models (e.g., Random Forests, XGBoost) use dozens or hundreds of features. If an uninformative feature with near-zero tree split importance (or near-zero SHAP value) drifts significantly, the model's actual predictions and classification accuracy remain completely unaffected.
2. **Monotonic Shifts Preserving Decision Boundaries**:
   - In tree-based models, if feature values shift from $[100, 200]$ to $[300, 400]$, but all tree split thresholds for that feature were at $X > 50$, all shifted data points will cleanly route down the exact same tree branches, yielding identical predictions.
3. **Invariant Decision Margins (Classification Saturation)**:
   - In logistic or sigmoid models, if an applicant's annual income increases from \$250,000 to \$500,000, their probability of loan repayment is already saturated at $> 0.99$. The distribution shift has zero impact on the final decision outcome.
4. **Covariate Shift with Consistent Concepts**:
   - The inputs shifted, but the underlying law governing the target $P(Y \mid X)$ remains sound. The model may generalize smoothly to the new input region if the training set had reasonable support or regularization.

**Operational Best Practice**:
Drift detection should trigger:
- Logging a `drift_event`.
- Increased monitoring and prioritization of ground-truth label collection.
- Validation against canary deployments.
- Targeted retraining only if model performance metrics (accuracy, F1, ROC-AUC) or output prediction distributions also degrade.

---

## 3. Data Model & Storage

### The `drift_events` Table Schema

The `drift_events` table permanently records individual feature evaluations:

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` (PK) | Unique event identifier. |
| `model_version_id`| `UUID` (FK) | Foreign key referencing the evaluated `ModelVersion`. |
| `feature` | `VARCHAR(100)` | Name of the feature evaluated (e.g. `credit_score`). |
| `method` | `VARCHAR(50)` | Statistical test method used (`PSI` or `KS_TEST`). |
| `score` | `FLOAT` | Calculated score ($PSI$ value or $D$ statistic). |
| `threshold` | `FLOAT` | Threshold used to classify drift (e.g. `0.25`). |
| `severity` | `VARCHAR(20)` | Classified severity (`NONE`, `LOW`, `MEDIUM`, `HIGH`). |
| `detected_at` | `DATETIME(TZ)` | Exact UTC timestamp of evaluation. |
| `drift_detected` | `BOOLEAN` | Whether the score breached the threshold. |
| `overall_drift_score` | `FLOAT` | Global drift metric across window. |
| `feature_drift_details`| `JSONB` | Granular bin contributions, ECDF differences, and sample metrics. |

---

## 4. API Specification

### `GET /api/v1/models/{id}/drift`
Retrieves the comprehensive statistical drift state and historical drift events for a registered model.

#### Request:
```http
GET /api/v1/models/33210f41-e8dd-4323-899a-e1a054164862/drift HTTP/1.1
Host: localhost:8000
Authorization: Bearer <JWT_ACCESS_TOKEN>
```

#### Response (`HTTP 200 OK`):
```json
{
  "model_id": "33210f41-e8dd-4323-899a-e1a054164862",
  "model_name": "credit_risk_sentinel",
  "version_tag": "v1.0.0",
  "reference_stats_configured": true,
  "drift_detected": true,
  "summary": {
    "total_evaluations": 12,
    "drifted_features_count": 5,
    "highest_severity": "HIGH"
  },
  "events": [
    {
      "id": "7ca64700-1c39-4467-bc5b-43d8389369f1",
      "feature": "credit_score",
      "method": "KS_TEST",
      "score": 0.921,
      "threshold": 0.25,
      "severity": "HIGH",
      "drift_detected": true,
      "detected_at": "2026-11-01T08:59:52.124500Z"
    },
    {
      "id": "e305e9ca-9ecb-4375-bebe-08f9725f3c1a",
      "feature": "annual_income",
      "method": "KS_TEST",
      "score": 0.283,
      "threshold": 0.25,
      "severity": "HIGH",
      "drift_detected": true,
      "detected_at": "2026-11-01T08:59:52.126800Z"
    },
    {
      "id": "98e4f55a-4bbf-474c-8f19-90d238c9da11",
      "feature": "loan_amount",
      "method": "KS_TEST",
      "score": 0.175,
      "threshold": 0.25,
      "severity": "NONE",
      "drift_detected": false,
      "detected_at": "2026-11-01T08:59:52.128100Z"
    }
  ]
}
```

---

## 5. Controlled Distribution Shift Simulation

The automated simulation script [`scripts/simulate_drift.py`](file:///d:/ModelOps%20Sentinel/scripts/simulate_drift.py) provides end-to-end verification of distribution shift detection:

1. **Baseline Registration**:
   - Generates 600 reference customer profiles ($credit\_score \sim \mathcal{N}(710, 45)$, $annual\_income \sim \mathcal{N}(75000, 15000)$, $debt\_to\_income \sim \mathcal{N}(0.28, 0.08)$).
   - Trains and registers a Scikit-Learn `RandomForestClassifier` with `ReferenceStats` attached to version `v1.0.0`.
2. **Phase 1: Normal In-Distribution Traffic**:
   - Generates 60 inference requests sampled from the exact same distributions as reference.
   - Triggers drift evaluation (`PSI` and `KS_TEST`).
   - **Verification**: Confirms `drift_detected == False`, with low PSI scores ($< 0.08$) and small KS statistics.
3. **Phase 2: Controlled Covariate Shift**:
   - Injects macroeconomic distress shift:
     - $credit\_score \sim \mathcal{N}(610, 45)$ (dropped by 100 points, $-2.2\sigma$)
     - $debt\_to\_income \sim \mathcal{N}(0.48, 0.09)$ (increased debt ratio, $+2.5\sigma$)
     - $annual\_income \sim \mathcal{N}(62000, 12000)$ (moderate decrease)
     - $loan\_amount$ left unshifted as an un-drifted control.
   - Sends 60 shifted production inferences.
   - Triggers drift evaluation.
   - **Verification**: Confirms `drift_detected == True`, with $credit\_score$ $PSI = 2.27$ (HIGH), $KS = 0.92$ (HIGH), and $loan\_amount$ remains unflagged ($NONE$).
4. **API Verification**:
   - Issues `GET /api/v1/models/{id}/drift` and verifies payload integrity and summary metrics.

---

## 6. Test Suite & Verification Results

All tests have been implemented and verified in [`tests/test_drift.py`](file:///d:/ModelOps%20Sentinel/tests/test_drift.py):

| Test Case | Description | Result |
|---|---|---|
| `test_psi_identical_distribution` | Normal samples from identical distributions produce $PSI < 0.10$ and `NONE` severity | **PASSED** |
| `test_psi_shifted_distribution` | $+2\sigma$ shifted samples produce $PSI > 0.25$ and `HIGH` severity | **PASSED** |
| `test_ks_identical_distribution` | Identical continuous distributions produce $D < 0.15$, $p > 0.05$, `NONE` severity | **PASSED** |
| `test_ks_shifted_distribution` | Shifted continuous distribution produces $D > 0.25$, $p < 0.001$, `HIGH` severity | **PASSED** |
| `test_extract_reference_statistics` | Profiles deciles, parametric moments, and sample reservoir across features | **PASSED** |
| `test_drift_evaluation_and_events_recording` | Evaluates production requests against reference stats and persists `drift_events` | **PASSED** |
| `test_get_model_drift_api` | Verifies `GET /api/v1/models/{id}/drift` response schema, summary, and events | **PASSED** |
| `test_worker_evaluate_drift_task` | Verifies decoupled asynchronous worker execution via arq task | **PASSED** |
