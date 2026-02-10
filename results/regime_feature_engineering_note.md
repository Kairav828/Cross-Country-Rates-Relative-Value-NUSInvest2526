# Regime Feature Engineering: Statistical Challenges and Design Choices

## 1. Data Frequency Mismatch and Temporal Persistence

### Problem Statement
Policy rate spreads (Fed-ECB, Fed-BoJ) update quarterly while other features (MOVE, DXY, CESI) update daily, creating a sampling imbalance: $T_{\text{policy}} = 4 \text{ per year} \ll T_{\text{macro}} = 252 \text{ per year}$.

### Statistical Implications
The Gaussian HMM assumes continuous feature evolution within regimes. When policy spreads remain constant for ~63 consecutive trading days, the model interprets this as:
$$\text{Var}(\Delta \text{spread}_t | \text{regime}_i) \approx 0$$

This degenerates the emission covariance matrix $\boldsymbol{\Sigma}_i$ (near-singular with one eigenvalue → 0). In testing, we found that instead of overfitting to noise, the EM algorithm maximized likelihood by "latching" onto these zero-variance periods. This produced very long regimes with minimal switching (persisting for the entire inter-meeting duration), effectively ignoring the daily volatility in other macro features we intended to capture.

### Attempted Solutions and Failures

**First Differencing**: Differencing policy spreads yields a sequence dominated by zeros (95% of days) punctuated by discrete jumps at FOMC/ECB meetings (5% of days). The resulting distribution is multimodal (spike at zero + fat tails), violating the Gaussian emission assumption. The HMM identifies spurious regimes around individual policy decisions rather than persistent macro states.

**Cycle Extraction (HP Filter)**: Decomposing policy spreads via Hodrick-Prescott filter (λ=1600 for quarterly data upsampled to daily) produced smooth cycles but lost information about policy divergence timing. The resulting feature had minimal variance and contributed negligibly to regime classification.

### Decision
Drop policy spreads entirely to ensure temporal homogeneity in the observation model. This sacrifices direct monetary policy information but prevents spurious regime fragmentation.

---

## 2. Multicollinearity and Information Redundancy

### Problem Statement
Expanding to 11 features created excessive correlation within subsets:
- Volatility cluster: MOVE, VIX exhibited ρ = 0.25, while FX overnight implied volatility between USD and EUR exhibited ρ = 0.69
- Curve cluster: Individual country 2s10s slopes showed ρ = 0.37
- Macro cluster: CESI z-scores across countries with ρ ∈ [0.1, 0.2]

### Statistical Implications
When eigenvalues of the within-regime covariance matrix $\boldsymbol{\Sigma}_i$ span multiple orders of magnitude ($\lambda_{\text{max}} / \lambda_{\text{min}} > 10^4$), numerical instability emerges:

$$\text{det}(\boldsymbol{\Sigma}_i) = \prod_{j=1}^p \lambda_j \to 0$$

The Baum-Welch EM algorithm's convergence becomes path-dependent, and BIC comparison loses reliability (overfitting to correlation structure rather than regime dynamics). Redundant features inflate the parameter count without adding regime-discriminating information.

### Trade-offs
- **Keep all features**: Retain maximum information but risk numerical instability and overfitting (320 parameters for 4-state, 11-feature model)
- **Manual feature selection**: Drop features subjectively (e.g., keep MOVE, drop VIX) but lose potentially regime-relevant information
- **PCA dimensionality reduction**: Collapse correlated groups systematically but sacrifice interpretability

---

## 3. Dimensionality Reduction via Principal Components Analysis

### Methodology
Collapsed correlated feature groups using PCA:
- **Curve signal**: PC1 of (US, EUR, JPY, AUD) 2s10s changes → captures global curve steepening/flattening
- **Macro surprise**: PC1 of (USD, EUR, JPY, AUD) CESI z-scores → represents synchronized growth momentum

### Statistical Justification
If individual features $x_{i,t}$ are noisy proxies of a latent factor $\xi_t$:
$$x_{i,t} = \beta_i \xi_t + \epsilon_{i,t}, \quad \epsilon_{i,t} \sim \mathcal{N}(0, \sigma_i^2)$$

Then PC1 estimates the maximum-variance projection:
$$\widehat{\xi}_t = \arg\max_{\mathbf{w}} \mathbf{w}^T \text{Cov}(\mathbf{X}) \mathbf{w} \quad \text{s.t.} \quad \|\mathbf{w}\| = 1$$

For regime classification, this recovers the **common component** (global risk-on/risk-off) while discarding country-specific idiosyncrasies. We interpret PC1 not as a mechanical linear combination, but as an estimated latent ‘global’ factor (global curve shift / synchronized surprises) that is more stable than any single country series.

### Drawbacks and Limitations
1. **Information loss**: PC1 captures 32-36% of variance → remainder discarded (may contain regime-relevant country divergences, e.g., EUR-USD policy desynchronization)
2. **Interpretation complexity**: PC1 loadings are linear combinations, not directly observable economic quantities (hard to explain "curve signal = 0.54×US + 0.51×EUR + ...")
3. **Temporal instability**: PCA loadings estimated on full sample may shift in rolling windows due to structural changes (e.g., ECB QE altered EUR weight in curve signal post-2015)
4. **Sign ambiguity**: PC1 direction is arbitrary (eigenvector sign convention) → must verify economic interpretation matches intuition (positive = steepening vs flattening)
5. **Non-stationarity of loadings**: Assumes correlation structure is constant over time, but crises alter cross-country relationships (e.g., EUR-USD curve correlation flipped during sovereign debt crisis)

---

## 4. Feature Scaling and Initialization Bias

### Problem Statement
The Baum-Welch (EM) algorithm used to train HMMs is a local optimizer sensitive to initialization. `hmmlearn` initializes model parameters using **K-Means clustering** on the dataset. K-Means relies on Euclidean distance, meaning features with larger variances dominate the initial cluster assignments.

If features are unscaled, components with naturally small units are effectively ignored during initialization. While the EM algorithm *can* eventually learn to handle different scales (by adapting the covariance matrix $\boldsymbol{\Sigma}_i$), a poor initialization often traps the model in a suboptimal local minimum where regimes are poorly separated.

### Implemented Solution
Imposed bias on the initialization via target standard deviations: MOVE changes (3.0), DXY changes (1.0), curve signal (2.0), macro surprise (1.5).

### Economic Rationale
Although all features are sampled daily, they represent different types of market phenomena. We explicitly prioritize volatility because we are modeling **Risk Regimes**, not just **Business Cycles**:

1.  **MOVE changes (Target $\sigma=3.0$ = Primary Driver)**: 
    *   **Logic**: A "Regime Change" in our trading context is defined primarily by a breakdown in liquidity and risk tolerance. Volatility is the most direct proxy for this.
    *   **Effect**: By keeping this variance high (relative to others), we force the HMM to prioritize separating "Crisis" (High Vol) from "Calm" (Low Vol) states above all else. This preserves the natural hierarchy observed in the unscaled data.

2.  **Curve Signal (Target $\sigma=2.0$ = Secondary Driver)**:
    *   **Logic**: The shape of the yield curve (Inverted vs. Steep) is the second most important determinant of relative value strategy performance.
    *   **Effect**: This differentiates the "Calm" regimes. Once the model has separated High vs. Low volatility (via MOVE), this feature helps split the Low Volatility periods into "Reflation/Steepening" vs. "Late Cycle/Flattening."

3.  **DXY & Macro Surprise (Target $\sigma=1.0 - 1.5$ = Context)**:
    *   **Logic**: These provide context (is the crisis USD-led? is growth slowing?) but are not the primary definitions of the regime.
    *   **Effect**: Scaling these lower prevents the model from creating regimes based purely on idiosyncratic FX moves or hearing "noise" in economic data releases, ensuring they only act as modifiers to the dominant Vol/Curve states.

### Statistical Trade-offs
**Advantages**:
- **Guided Convergence**: Steers the non-convex optimization toward an economically interpretable solution (volatility regimes) rather than a mathematical curiosity.
- **Robustness**: Prevents low-variance noise from creating spurious clusters during the critical initialization phase.

**Disadvantages**:
1.  **Initialization Dependence**: We are explicitly biasing the model. If the "true" maximum likelihood solution does not align with volatility, we are forcing the model away from it.
2.  **Subjectivity**: The specific ratios (3:2) are heuristic engineered features, not learned parameters.
3.  **Transient Effect**: Mathematically, scaling does not change the objective function of the HMM (the likelihood function is scale-invariant for Gaussian emissions). The effect is purely on the *trajectory* of the optimizer, meaning we rely on the EM algorithm getting "stuck" in the specific local optimum we pointed it toward.

---

## 5. Stationarity Enforcement and Structural Breaks

### Requirement
HMM assumes stationary emissions within each regime:
$$p(\mathbf{x}_t | s_t = i) = \mathcal{N}(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) \quad \forall t$$

If features are I(1) non-stationary, the model spuriously attributes trends to regime switches rather than persistent dynamics within states.

### Applied Transformations
| Feature          | Transformation                  | Rationale                                  |
|------------------|---------------------------------|--------------------------------------------|
| MOVE changes     | First difference                | MOVE level has unit root (volatility regimes shift mean) |
| DXY changes      | First difference                | DXY level trends (secular USD strength/weakness cycles) |
| Curve signal     | PC1 of first-differenced slopes | Individual 2s10s levels are I(1) (rate level drift) |
| Macro surprise   | PC1 of rolling z-scores         | CESI bounded but time-varying variance |

### Structural Breaks and Model Limitations
**Identified breaks**: 2008-09 (GFC Co-movement shift), 2020-03 (Variance explosion).

**Implications**: The model assumes time-invariant parameters. Structural breaks violate this. We mitigate this by prioritizing **interpretability** (fixed regime labels) over adaptability, accepting that the model may misclassify transition periods during structural breaks.

---

## 6. Model Selection and Complexity-Fit Trade-off

### Bayesian Information Criterion
BIC formula: $\text{BIC} = -2\log L + k \log n$ where $k$ is the number of free parameters.

Tested 2, 3, 4 states. Lower BIC selected 4-state model.

### Notes on Parameter Counting
For a full-covariance Gaussian HMM with $N$ states and $M$ features:
$$k = \underbrace{N(N-1)}_{\text{TransMat}} + \underbrace{NM}_{\text{Means}} + \underbrace{N \frac{M(M+1)}{2}}_{\text{Covs}} + \underbrace{(N-1)}_{\text{StartProb}}$$

*   **Implementation Note**: Our current BIC calculation (68 parameters for 4 states) omits the $N-1$ initial state probabilities. The theoretical count is 71. Given $n \approx 5000$, this difference ($\approx 3 \log 5000 \approx 25$) is visible but negligible compared to the likelihood differences between state counts, so the model selection decision remains valid.

### Interpretation Challenges
**4-state model selected, but economic clarity decreases**:
- 2 states: Crisis vs Calm (too simple).
- 3 states: Crisis, Normal, Risk-on (intuitive).
- 4 states: Lowest BIC, but splits the "Calm" state into two subtle variants based on macro signals.

### Alternative Selection Criteria Not Used
- **AIC**: Penalizes complexity less, likely to suggest even more states (overfitting).
- **Cross-validation**: Difficult due to the latent nature of regimes (no ground truth labels).

---

## 7. Summary of Statistical Assumptions and Limitations

### Core Assumptions
1. **Markovian dynamics**: $P(s_t | s_{1:t-1}) = P(s_t | s_{t-1})$
2. **Gaussian emissions**: Features are normally distributed within regimes.
3. **Time-invariant parameters**: Regime properties are constant over 20 years.

### Known Violations and Implications
**Heavy tails**: Crisis features exhibit excess kurtosis. The Gaussian assumption forces the model to switch regimes rapidly during extreme events to fit the tails, potentially interpreting single outliers as regime shifts.

**Autocorrelation**: Features show slight autocorrelation. This violates the assumption that observations are conditionally independent given the state, potentially inflating the likelihood and leading BIC to select too many states.

**Feature incompleteness**: Omitting policy rates means the model is blind to gradual monetary tightening cycles, capturing only the resulting volatility outcomes.