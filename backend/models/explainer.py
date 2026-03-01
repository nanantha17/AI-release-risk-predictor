"""
Risk Explainer
──────────────
Computes SHAP-style feature attribution for interpretable risk factors.
Uses sklearn's feature_importances_ combined with deviation-from-baseline analysis.
"""

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


BASELINES = {
    "test_coverage": 80.0,
    "code_coverage": 75.0,
    "branch_coverage": 70.0,
    "past_defects_total": 5.0,
    "critical_defects": 0.5,
    "defect_resolution_rate": 90.0,
    "cyclomatic_complexity": 8.0,
    "lines_of_code_changed": 2000.0,
    "num_contributors": 4.0,
    "build_success_rate": 95.0,
    "avg_pr_review_time_hours": 24.0,
    "open_issues": 10.0,
}

HIGHER_IS_RISKIER = {
    "past_defects_total", "critical_defects", "cyclomatic_complexity",
    "lines_of_code_changed", "avg_pr_review_time_hours", "open_issues",
}
LOWER_IS_RISKIER = {
    "test_coverage", "code_coverage", "branch_coverage",
    "defect_resolution_rate", "build_success_rate",
}

FEATURE_META = {
    "test_coverage": ("Test Coverage", "% of code covered by tests"),
    "code_coverage": ("Code Coverage", "% of code lines executed during tests"),
    "branch_coverage": ("Branch Coverage", "% of code branches tested"),
    "past_defects_total": ("Historical Defects", "Total defects found in past 3 releases"),
    "critical_defects": ("Critical Defects", "P0/blocker defects in current release"),
    "defect_resolution_rate": ("Defect Resolution Rate", "% of defects fixed before release"),
    "cyclomatic_complexity": ("Code Complexity", "Average cyclomatic complexity score"),
    "lines_of_code_changed": ("Changeset Size", "Lines of code modified in this release"),
    "num_contributors": ("Team Size", "Number of engineers contributing"),
    "build_success_rate": ("CI Build Success", "% of CI pipeline runs that passed"),
    "avg_pr_review_time_hours": ("PR Review Time", "Average hours to review and merge PRs"),
    "open_issues": ("Open Issues", "Unresolved issues at release time"),
}


def _severity(impact: float) -> str:
    a = abs(impact)
    if a < 0.1: return "low"
    if a < 0.25: return "medium"
    if a < 0.5: return "high"
    return "critical"


class RiskExplainer:
    """Explains which features contribute most to the predicted risk."""

    def explain(self, metrics, prediction: dict) -> list:
        from pydantic import BaseModel
        factors = []

        metric_values = {
            "test_coverage": metrics.test_coverage,
            "code_coverage": metrics.code_coverage,
            "branch_coverage": metrics.branch_coverage,
            "past_defects_total": metrics.past_defects_total,
            "critical_defects": metrics.critical_defects,
            "defect_resolution_rate": metrics.defect_resolution_rate,
            "cyclomatic_complexity": metrics.cyclomatic_complexity,
            "lines_of_code_changed": metrics.lines_of_code_changed,
            "num_contributors": metrics.num_contributors,
            "build_success_rate": metrics.build_success_rate,
            "avg_pr_review_time_hours": metrics.avg_pr_review_time_hours,
            "open_issues": metrics.open_issues,
        }

        for feat, value in metric_values.items():
            baseline = BASELINES[feat]
            label, desc = FEATURE_META[feat]
            if feat == "num_contributors":    # misleading that more contributors may lead to more risk
                continue
            if feat in HIGHER_IS_RISKIER:
                # Normalized deviation: how much above baseline?
                # ── FIX 2: Guard against zero baseline AND zero value
                if baseline <= 0 and value <= 0:
                    impact = 0.0  # both zero = at baseline = no impact
                elif baseline <= 0:
                    impact = float(np.clip(np.log1p(value) * 0.3, 0, 0.8))
                else:
                    norm_val = np.log1p(value) / np.log1p(max(baseline, 1))
                    impact = float(np.clip((norm_val - 1.0) * 0.5, -0.5, 0.8))
            else:  # lower is riskier
                deficit = (baseline - value) / baseline
                impact = float(np.clip(deficit * 0.8, -0.3, 0.8))

            if abs(impact) < 0.03:
                continue  # skip negligible factors

            direction = "↑ increases" if impact > 0 else "↓ decreases"
            detail = f"Current: {value:.1f} (baseline: {baseline:.1f}). {direction} release risk."

            from pydantic import BaseModel as BM
            # Use a dict since we return List[RiskFactor] from pydantic
            factors.append({
                "name": label,
                "impact": round(impact, 3),
                "description": f"{desc}. {detail}",
                "severity": _severity(impact),
            })

        # Sort by absolute impact descending, take top 8
        factors.sort(key=lambda f: abs(f["impact"]), reverse=True)
        return factors[:8]
