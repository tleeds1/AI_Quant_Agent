# Nightly Eval Scorecard
Generated at: `2026-08-31 15:30:54`

## Summary Metrics

- **Precision**: `1.000`
- **Recall**: `1.000`
- **Total Fixtures**: `8`
- **Citation Precision (V3, deterministic)**: `1.000`
- **Citation Recall (V3, deterministic)**: `1.000`
- **Citation Fixtures**: `5`

> Every number above is measured against a small, hand-curated fixture set with unambiguous ground truth (architecture.md §10.2's own caveat) -- this proves the verifier's deterministic logic is correct against a controlled set, not real-world precision/recall against a live model or a large labelled corpus. No live `ANTHROPIC_API_KEY` was used to produce these numbers.

## Fixtures Breakdown

| Fixture Name | Expected | Actual | P/R Status | Caught By Layer | Matched |
|---|---|---|---|---|---|
| clean_pass | `PASS` | `PASS` | `PASS == PASS` | None (Passed) | ✅ |
| hallucinated_number | `FAIL` | `FAIL` | `FAIL == FAIL` | V2 | ✅ |
| dangling_evidence_id | `FAIL` | `FAIL` | `FAIL == FAIL` | V1 | ✅ |
| unresolved_metric_ref | `FAIL` | `FAIL` | `FAIL == FAIL` | V1 | ✅ |
| r001_extreme_risk_buy | `FAIL` | `FAIL` | `FAIL == FAIL` | V4 | ✅ |
| r008_guaranteed_language | `FAIL` | `FAIL` | `FAIL == FAIL` | V4 | ✅ |
| unsupported_claim | `FAIL` | `FAIL` | `FAIL == FAIL` | V5 | ✅ |
| contradictory_claims | `FAIL` | `FAIL` | `FAIL == FAIL` | V5 | ✅ |
