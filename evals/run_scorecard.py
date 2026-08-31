import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

from evals.citation_fixtures import CITATION_FIXTURES
from evals.fixtures import GOLDEN_FIXTURES
from tests.unit.evals.citation_harness import citation_precision_recall
from tests.unit.evals.harness import precision_recall, run_fixture

from quantagent.llm.prompts import PromptLoader


async def main() -> None:
    prompts = PromptLoader()
    print("Running Golden Fixtures scorecard...")
    runs = []
    for fixture in GOLDEN_FIXTURES:
        run = await run_fixture(fixture, prompts=prompts)
        runs.append(run)
        print(
            f"Fixture: {fixture.name} -> "
            f"Expected: {fixture.expected_verdict}, Actual: {run.report.verdict}"
        )

    precision, recall = precision_recall(runs)

    print("\nRunning Citation Precision fixtures (V3, deterministic, no live model)...")
    citation_precision, citation_recall = citation_precision_recall(CITATION_FIXTURES)

    # Compile results
    fixtures_list: list[dict[str, Any]] = []

    for r in runs:
        # Check if actual verdict matched expected
        passed_expectation = r.report.verdict == r.fixture.expected_verdict

        # Check if correct layer caught it
        caught_correctly = True
        if r.fixture.expected_layer is not None:
            caught_correctly = any(
                check.layer == r.fixture.expected_layer and check.verdict == "FAIL"
                for check in r.results
            )

        matched = passed_expectation and caught_correctly

        fixtures_list.append(
            {
                "name": r.fixture.name,
                "expected_verdict": r.fixture.expected_verdict,
                "actual_verdict": r.report.verdict,
                "expected_layer": (
                    str(r.fixture.expected_layer) if r.fixture.expected_layer else None
                ),
                "matched": matched,
                "results_summary": [
                    {
                        "layer": check.layer,
                        "check_id": check.check_id,
                        "verdict": check.verdict,
                        "message": check.message,
                    }
                    for check in r.results
                ],
            }
        )

    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_fixtures": len(GOLDEN_FIXTURES),
            "precision": precision,
            "recall": recall,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "citation_fixtures": len(CITATION_FIXTURES),
        },
        "fixtures": fixtures_list,
    }

    # Write JSON
    os.makedirs("evals", exist_ok=True)
    json_path = "evals/scorecard.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON scorecard to {json_path}")

    # Write Markdown
    md_path = "evals/scorecard.md"
    md_lines = [
        "# Nightly Eval Scorecard",
        f"Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Summary Metrics",
        "",
        f"- **Precision**: `{precision:.3f}`",
        f"- **Recall**: `{recall:.3f}`",
        f"- **Total Fixtures**: `{len(GOLDEN_FIXTURES)}`",
        f"- **Citation Precision (V3, deterministic)**: `{citation_precision:.3f}`",
        f"- **Citation Recall (V3, deterministic)**: `{citation_recall:.3f}`",
        f"- **Citation Fixtures**: `{len(CITATION_FIXTURES)}`",
        "",
        "> Every number above is measured against a small, hand-curated fixture set with "
        "unambiguous ground truth (architecture.md §10.2's own caveat) -- this proves the "
        "verifier's deterministic logic is correct against a controlled set, not real-world "
        "precision/recall against a live model or a large labelled corpus. No live "
        "`ANTHROPIC_API_KEY` was used to produce these numbers.",
        "",
        "## Fixtures Breakdown",
        "",
        "| Fixture Name | Expected | Actual | P/R Status | Caught By Layer | Matched |",
        "|---|---|---|---|---|---|",
    ]
    for fix in fixtures_list:
        caught_layers = set(
            res["layer"] for res in fix["results_summary"] if res["verdict"] == "FAIL"
        )
        caught_str = ", ".join(caught_layers) if caught_layers else "None (Passed)"
        status_icon = "✅" if fix["matched"] else "❌"
        md_lines.append(
            f"| {fix['name']} | `{fix['expected_verdict']}` | `{fix['actual_verdict']}` | "
            f"`{fix['expected_verdict']} == {fix['actual_verdict']}` | "
            f"{caught_str} | {status_icon} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved Markdown scorecard to {md_path}")

    # Exit with code 0 on success, 1 on failure
    all_matched = all(f["matched"] for f in fixtures_list)
    if all_matched:
        print("\nAll fixtures passed expectations!")
        sys.exit(0)
    else:
        print("\nSome fixtures did not meet expectations!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
