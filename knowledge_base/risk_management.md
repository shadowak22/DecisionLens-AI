# Risk Management and Failure Analysis Frameworks

## 1. Risk Assessment Matrix (Likelihood vs. Impact)
Structured categorization of vulnerabilities across 5x5 matrices:
- **Low Likelihood, Low Impact**: Accept and monitor routinely.
- **High Likelihood, Low Impact**: Mitigate through automated procedures, error logging, and standard guardrails.
- **Low Likelihood, High Impact (Black Swan)**: Formulate business continuity plans, disaster recovery protocols, insurance, and architectural decoupling.
- **High Likelihood, High Impact (Critical Red Zone)**: Unacceptable risk profile; immediate redesign, pivot, or strategic pause required.

## 2. Failure Mode and Effects Analysis (FMEA)
FMEA calculates a Risk Priority Number (RPN) for every potential point of failure:
$$RPN = \text{Severity (S)} \times \text{Occurrence (O)} \times \text{Detection (D)}$$
- **Severity (1-10)**: How catastrophic is the consequence to users, revenue, or compliance?
- **Occurrence (1-10)**: What is the estimated frequency of this failure mode occurring?
- **Detection (1-10)**: How difficult is it to identify the fault before downstream damage occurs?
Items with $RPN > 150$ demand mandatory pre-launch mitigation architectures.

## 3. Four T's of Risk Response
Organizations must explicitly classify their stance towards each identified threat:
- **Tolerate**: Accept the residual risk when mitigation costs exceed potential exposure.
- **Treat**: Implement safeguards, redundancies, rate-limiting, and validation controls to lower risk.
- **Transfer**: Shift liability via third-party service agreements, cyber insurance, or cloud vendor SLAs.
- **Terminate**: Discontinue high-risk features, unvalidated integrations, or non-compliant dependencies.

## 4. Blast Radius Containment
Architectural best practices to isolate failures:
- **Circuit Breakers & Graceful Degradation**: Isolate third-party API dependencies so outages do not cascade into core services.
- **Staged Phased Rollouts (Canary Deployments)**: Expose 5% of users to new changes initially before full fleet deployment.
- **Dead Man's Switches & Instant Rollback**: Automated thresholds that revert changes if error budgets or latency SLOs are breached.
