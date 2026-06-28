# ALSPC-DPO structured audit

Adaptive Linguistic-Signal Preference Calibration DPO (ALSPC-DPO) trains a structured audit model to output `correctness`, `error_type`, and an evidence-grounded `rationale` for a candidate finding label.

The adaptive margin is:

```text
m(x) = clip(m0 + alpha_neg * S_neg_norm + alpha_imp * S_imp_norm, m_min, m_max)
```

The manuscript-selected margin-search configuration was:

```text
m0 = 0.1202
alpha_neg = 1.1497
alpha_imp = 0.3247
m_min = 0.0
m_max = 0.7819
```

Use `scripts/build_alspc_margin_data.py` to attach margins to preference data before ALSPC-DPO training.
