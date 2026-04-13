"""
Release Risk Predictor - FastAPI Backend
Uses PyTorch + Transformers + Scikit-learn to predict software release risk.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import uvicorn #fastapi
import logging
import time

# Local modules
from models.risk_model import RiskPredictor
from models.explainer import RiskExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Release Risk Predictor API",
    description="Predicts software release risk using ML metrics analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models at startup
predictor = RiskPredictor()
explainer = RiskExplainer()

# ─── Schemas ────────────────────────────────────────────────────────────────

class ReleaseMetrics(BaseModel):
    package_name: str = Field(..., description="Name of the software package")
    version: str = Field(..., description="Release version string")

    # Code quality metrics (0-100 scale)
    test_coverage: float = Field(..., ge=0, le=100, description="Test coverage %")
    code_coverage: float = Field(..., ge=0, le=100, description="Code coverage %")
    branch_coverage: float = Field(..., ge=0, le=100, description="Branch coverage %")

    # Defect history
    past_defects_total: int = Field(..., ge=0, description="Total defects in last 3 releases")
    critical_defects: int = Field(..., ge=0, description="Critical/P0 defects")
    defect_resolution_rate: float = Field(..., ge=0, le=100, description="% defects resolved before release")

    # Code complexity
    cyclomatic_complexity: float = Field(..., ge=0, description="Average cyclomatic complexity")
    lines_of_code_changed: int = Field(..., ge=0, description="LOC changed in this release")
    num_contributors: int = Field(..., ge=1, description="Number of contributors")

    # CI/CD signals
    build_success_rate: float = Field(..., ge=0, le=100, description="CI build success rate %")
    avg_pr_review_time_hours: float = Field(..., ge=0, description="Avg PR review time in hours")
    open_issues: int = Field(..., ge=0, description="Open issues at release time")

    # Optional free-text notes
    release_notes: Optional[str] = Field(None, description="Release notes or changelog text")


class RiskFactor(BaseModel):
    name: str
    impact: float  # -1.0 to 1.0
    description: str
    severity: str  # low / medium / high / critical


class PredictionResponse(BaseModel):
    package_name: str
    version: str
    risk_score: float         # 0-100
    risk_level: str           # LOW / MEDIUM / HIGH / CRITICAL
    confidence: float         # 0-1
    risk_factors: List[RiskFactor]
    recommendations: List[str]
    ml_breakdown: dict
    processing_time_ms: float


class BatchRequest(BaseModel):
    releases: List[ReleaseMetrics]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", models_loaded=True, version="1.0.0")


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(metrics: ReleaseMetrics):
    """Predict release risk for a single package."""
    start = time.time()
    try:
        result = predictor.predict(metrics)
        factors = explainer.explain(metrics, result)
        recs = _generate_recommendations(metrics, result, factors)
        elapsed = (time.time() - start) * 1000

        return PredictionResponse(
            package_name=metrics.package_name,
            version=metrics.version,
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            confidence=result["confidence"],
            risk_factors=factors,
            recommendations=recs,
            ml_breakdown=result["breakdown"],
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    """Predict risk for multiple releases at once."""
    results = []
    for metrics in request.releases:
        result = predictor.predict(metrics)
        factors = explainer.explain(metrics, result)
        recs = _generate_recommendations(metrics, result, factors)
        results.append({
            "package_name": metrics.package_name,
            "version": metrics.version,
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "confidence": result["confidence"],
        })
    return {"results": results, "count": len(results)}


@app.get("/metrics/schema")
def get_metrics_schema():
    """Returns the expected input schema with descriptions."""
    return ReleaseMetrics.schema()


@app.get("/examples")
def get_examples():
    """Return example payloads for testing."""
    return {
        "low_risk": {
            "package_name": "auth-service",
            "version": "2.3.1",
            "test_coverage": 92, "code_coverage": 88, "branch_coverage": 85,
            "past_defects_total": 3, "critical_defects": 0, "defect_resolution_rate": 100,
            "cyclomatic_complexity": 4.2, "lines_of_code_changed": 320, "num_contributors": 3,
            "build_success_rate": 98, "avg_pr_review_time_hours": 4, "open_issues": 2,
            "release_notes": "Minor bug fixes and performance improvements"
        },
        "high_risk": {
            "package_name": "payment-gateway",
            "version": "4.0.0",
            "test_coverage": 45, "code_coverage": 38, "branch_coverage": 30,
            "past_defects_total": 24, "critical_defects": 5, "defect_resolution_rate": 60,
            "cyclomatic_complexity": 22.7, "lines_of_code_changed": 15000, "num_contributors": 18,
            "build_success_rate": 72, "avg_pr_review_time_hours": 72, "open_issues": 87,
            "release_notes": "Major rewrite of payment processing core"
        }
    }


# ─── Helpers ────────────────────────────────────────────────────────────────

def _generate_recommendations(metrics: ReleaseMetrics, result: dict, factors: List[RiskFactor]) -> List[str]:
    recs = []
    if metrics.test_coverage < 70:
        recs.append(f"🧪 Increase test coverage from {metrics.test_coverage:.0f}% to at least 80% before release.")
    if metrics.critical_defects > 0:
        recs.append(f"🚨 Resolve all {metrics.critical_defects} critical defect(s) before shipping.")
    if metrics.build_success_rate < 85:
        recs.append(f"🔧 CI build success rate is {metrics.build_success_rate:.0f}% — investigate flaky tests.")
    if metrics.cyclomatic_complexity > 15:
        recs.append(f"📊 High cyclomatic complexity ({metrics.cyclomatic_complexity:.1f}). Refactor complex modules.")
    if metrics.open_issues > 50:
        recs.append(f"📋 {metrics.open_issues} open issues — triage and close/defer before release.")
    if metrics.avg_pr_review_time_hours > 48:
        recs.append("⏱ Slow PR review cycle (>48h). Enforce SLAs to improve code quality signals.")
    if metrics.lines_of_code_changed > 10000:
        recs.append("📦 Large changeset (>10k LOC). Consider splitting into smaller incremental releases.")
    if metrics.defect_resolution_rate < 80:
        recs.append(f"🐛 Only {metrics.defect_resolution_rate:.0f}% of defects resolved. Address backlog before release.")
    if not recs:
        recs.append("✅ Metrics look healthy. Proceed with standard release validation.")
    return recs


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
