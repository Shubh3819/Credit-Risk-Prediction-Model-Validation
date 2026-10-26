CREDIT RISK MODEL — PRELIMINARY EVALUATION LIMITATIONS

Models:
- Logistic Regression baseline.
- XGBoost classifier.
- Both models use the same feature set and the same stratified
  train/test split.

Target:
- target_default was preserved as provided.
- The data dictionary does not define the target observation window,
  default event date, or exact target-generation rule.

Repayment features:
- Months 1-3 repayment aggregates were included because repayment
  feature engineering is required by the assignment.
- Their temporal relationship to target_default is not established.
- Therefore, these results must not be described as leakage-free.

Bureau features:
- Bureau records opened after loan origination were excluded from
  the origination-cutoff aggregate.
- Bureau reporting dates for outstanding balances and delinquency
  are unavailable, so reporting-time leakage cannot be ruled out.

Collections:
- Collections events were excluded because event dates include
  records before and after associated loan origination.

Preprocessing:
- Numeric imputation, scaling, and categorical encoding were fitted
  using training data through sklearn pipelines.
- The test set was used for initial model evaluation.
- The probability threshold of 0.5 is a reference threshold, not
  an optimized business decision threshold.

Interpretation:
- Performance metrics are preliminary and conditional on the
  undocumented synthetic target definition and timing assumptions.
- Feature importance indicates predictive association, not causality.