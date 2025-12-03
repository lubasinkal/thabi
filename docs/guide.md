
GEMAct Data Input Guide
1. Frequency (Number of Claims)
- Distribution name: e.g., 'poisson', 'binomial', 'negativebinomial'.
- Parameters: depend on distribution.
  - Poisson: {'mu': expected number of claims}
  - Binomial: {'n': trials, 'p': probability}
- Optional: threshold (analysis threshold for reported claims).
Example:
from gemact.lossmodel import Frequency
frequency = Frequency(dist='poisson', par={'mu': 7}, threshold=0)
2. Severity (Claim Size Distribution)
- Distribution name: e.g., 'gamma', 'lognormal', 'pareto2', 'genpareto'.
- Parameters: dictionary of distribution parameters.
  - Gamma: {'a': shape}
  - Lognormal: {'scale': value, 'shape': sigma}
  - Pareto2: {'scale': K, 'shape': q}
  - GenPareto: {'loc': 0, 'scale': σ, 'c': ξ}
Example (Pareto tail above $25k):
from gemact.lossmodel import Severity
severity = Severity(dist='pareto2', par={'scale': 25000, 'shape': 1.5})
3. Policy Structure (Reinsurance Layers)
- Layer object defines deductible, cover, aggregate limits, reinstatements.
- Parameters:
  - deductible (priority)
  - cover (limit per claim)
  - aggr_cover (aggregate limit)
  - aggr_deductible (aggregate deductible)
  - n_reinst (number of reinstatements)
  - reinst_percentage (premium percentage for reinstatement)
  - share (quota share participation)
Example (475k xs 25k layer):
from gemact.lossmodel import PolicyStructure, Layer
policy = PolicyStructure(layers=Layer(deductible=25000, cover=475000))
4. Loss Model (Aggregate Distribution)
- Inputs: frequency, severity, policy structure.
- Computation method: 'fft', 'recursive', 'mc'.
- Discretization method: 'massdispersal', 'upper_discretisation', 'lower_discretisation', 'localmoments'.
- Nodes: number of discretization nodes (n_aggr_dist_nodes).
- Simulation size: if Monte Carlo (n_sim).
Example:
from gemact.lossmodel import LossModel
lm = LossModel(
    frequency=frequency,
    severity=severity,
    policystructure=policy,
    aggr_loss_dist_method='fft',
    sev_discr_method='massdispersal',
    n_aggr_dist_nodes=2**17
)
5. Outputs
- Pure premium: lm.pure_premium_dist[0]
- Moments: lm.mean(), lm.var(), lm.skewness()
- Quantiles: lm.ppf(q=[0.95])
- Simulation: lm.rvs(n)
6. Climate Data Integration
- Frequency μ: adjust based on rainfall extremes.
- Severity q: adjust based on climate regime.
- Threshold K: set to damage threshold (e.g., $25k).
- Run scenarios by plugging climate-adjusted μ and q into inputs.
