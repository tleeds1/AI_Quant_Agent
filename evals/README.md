# Evals

Golden traces, adversarial suite and fixture portfolios described in `docs/architecture.md` §10.
`make eval` produces a scorecard (markdown + JSON). Populated starting M4 (Verifier) through M6
(Observability, evals, hardening).

**M4 status:** the verifier's own golden/flawed-answer fixture set (hallucination probes,
rule-breach fixtures, entailment-critic fixtures) and the precision/recall measurement it's
scored against live under `tests/unit/evals/`, not here. `pyproject.toml`'s `testpaths=["tests"]`
means pytest never looks in this directory, and `make eval*` targets are still stubs (see
`Makefile`) — a real, CI-wired scorecard CLI reading/writing this directory is explicitly M6 scope
(`docs/guideline.md`'s M6 DoD: "all §10.4 gates wired into CI"). M4 built the underlying
measurement functions (`hallucinated_number_rate`, `precision_recall`) and proved them against a
real fixture set with actual `pytest` assertions that already run under `make check` today —
`this` directory becomes the real, CI-wired home for that fixture set and scorecard starting M6,
which should migrate `tests/unit/evals/fixtures.py`'s content here rather than duplicating it.
