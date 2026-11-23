"""
Phase 5: module boundary and deployment-contract guards.

Layer rule (one-directional):

    data_loader → features → plots
                         ↘
    features → dashboard ← plots
    features → analyst  → dashboard (pack only; no CSV/Plotly)
    app → dashboard / analyst / features / plots / data_loader (display only)
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from dashboard import run_briefing
from data_loader import CACHE_TTL_HOURS, DOWNLOAD_ATTEMPTS, DOWNLOAD_RETRY_DELAYS_SECONDS
from features import AnalysisError, validate_cost_bps

ROOT = Path(__file__).resolve().parent.parent
LAYER_MODULES = ("data_loader", "features", "plots", "analyst", "dashboard", "app")

# Allowed top-level imports between project modules (outside ``if __name__``).
ALLOWED: dict[str, frozenset[str]] = {
    "data_loader": frozenset(),
    "features": frozenset({"data_loader"}),
    "plots": frozenset(),  # may use PORTFOLIO_LABEL string constant only — no features import
    "analyst": frozenset({"features"}),
    "dashboard": frozenset({"analyst", "data_loader", "features", "plots"}),
    "app": frozenset({"analyst", "dashboard", "data_loader", "features", "plots"}),
}


def _top_level_project_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.If):
            # Skip ``if __name__ == "__main__":`` blocks (smoke imports only).
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom) and child.module in LAYER_MODULES:
                found.add(child.module)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    root = alias.name.split(".", 1)[0]
                    if root in LAYER_MODULES:
                        found.add(root)
    return found


class ModuleBoundaryTests(unittest.TestCase):
    def test_layer_imports_stay_one_directional(self) -> None:
        for name, allowed in ALLOWED.items():
            path = ROOT / f"{name}.py"
            imports = _top_level_project_imports(path)
            unexpected = imports - allowed
            self.assertFalse(
                unexpected,
                f"{name}.py imports {sorted(unexpected)}; allowed={sorted(allowed)}",
            )

    def test_plots_does_not_import_data_or_features(self) -> None:
        source = (ROOT / "plots.py").read_text(encoding="utf-8")
        self.assertNotIn("import data_loader", source)
        self.assertNotIn("from data_loader", source)
        self.assertNotIn("from features", source)
        self.assertNotIn("import features", source)

    def test_analyst_does_not_load_prices_or_draw(self) -> None:
        source = (ROOT / "analyst.py").read_text(encoding="utf-8")
        self.assertNotIn("import data_loader", source)
        self.assertNotIn("from data_loader", source)
        self.assertNotIn("import plots", source)
        self.assertNotIn("from plots", source)
        self.assertNotIn("matplotlib", source)
        self.assertNotIn("plotly", source)


class InputValidationTests(unittest.TestCase):
    def test_cost_bps_out_of_range_fails_before_load(self) -> None:
        with self.assertRaises(AnalysisError):
            run_briefing(offline=True, preset="Max", cost_bps=10_000)

    def test_validate_cost_bps_rejects_non_numeric(self) -> None:
        with self.assertRaises(AnalysisError):
            validate_cost_bps("abc")  # type: ignore[arg-type]

    def test_blank_llm_key_is_treated_as_absent(self) -> None:
        briefing = run_briefing(
            offline=True,
            preset="Max",
            enable_llm_analyst=True,
            llm_api_key="   ",
        )
        self.assertIsNotNone(briefing.analyst)
        self.assertIsNotNone(briefing.analyst.llm)
        self.assertEqual(briefing.analyst.llm.claims, ())
        self.assertTrue(
            any("no API key" in note for note in briefing.analyst.llm.missing_notes)
        )


class CacheAndRetryContractTests(unittest.TestCase):
    def test_download_retries_are_bounded_and_documented(self) -> None:
        self.assertEqual(DOWNLOAD_ATTEMPTS, 3)
        self.assertEqual(len(DOWNLOAD_RETRY_DELAYS_SECONDS), DOWNLOAD_ATTEMPTS - 1)
        self.assertEqual(CACHE_TTL_HOURS, 12)
        self.assertTrue(all(delay > 0 for delay in DOWNLOAD_RETRY_DELAYS_SECONDS))


if __name__ == "__main__":
    unittest.main()
