"""
Risk Prediction Model
─────────────────────
Ensemble of:
1. PyTorch deep neural network (tabular features)
2. Scikit-learn GradientBoosting (structured metrics)
3. HuggingFace DistilBERT (release notes NLP)

Final risk score = weighted average of all three.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Feature Engineering ────────────────────────────────────────────────────

FEATURE_NAMES = [
    "test_coverage", "code_coverage", "branch_coverage",
    "past_defects_total", "critical_defects", "defect_resolution_rate",
    "cyclomatic_complexity", "lines_of_code_changed", "num_contributors",
    "build_success_rate", "avg_pr_review_time_hours", "open_issues",
    # Derived features
    "coverage_avg", "defect_density", "review_burden",
    "change_risk_index", "quality_gate_score",
]


def extract_features(metrics) -> np.ndarray:
    """Convert ReleaseMetrics into a flat feature vector."""
    raw = np.array([
        metrics.test_coverage,
        metrics.code_coverage,
        metrics.branch_coverage,
        metrics.past_defects_total,
        metrics.critical_defects,
        metrics.defect_resolution_rate,
        metrics.cyclomatic_complexity,
        metrics.lines_of_code_changed,
        metrics.num_contributors,
        metrics.build_success_rate,
        metrics.avg_pr_review_time_hours,
        metrics.open_issues,
    ], dtype=np.float32)

    # Derived features
    coverage_avg      = (metrics.test_coverage + metrics.code_coverage + metrics.branch_coverage) / 3
    defect_density    = metrics.past_defects_total / max(metrics.lines_of_code_changed / 1000, 1)
    review_burden     = metrics.lines_of_code_changed / max(metrics.num_contributors, 1)
    change_risk_index = np.log1p(metrics.lines_of_code_changed) * (metrics.cyclomatic_complexity / 10)
    quality_gate_score = (
        (coverage_avg / 100) * 0.4 +
        (metrics.build_success_rate / 100) * 0.3 +
        (metrics.defect_resolution_rate / 100) * 0.3
    ) * 100

    derived = np.array([
        coverage_avg, defect_density, review_burden,
        change_risk_index, quality_gate_score,
    ], dtype=np.float32)

    return np.concatenate([raw, derived])


# ─── PyTorch Neural Network ──────────────────────────────────────────────────

class RiskNet(nn.Module):
    """
    Deep tabular network for release risk regression.
    Architecture: Input → BatchNorm → residual blocks → Risk Score [0,1]
    """

    def __init__(self, input_dim: int = 17):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Wider network to better separate low/high risk
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.block4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x).squeeze(-1)


# ─── Scikit-learn Ensemble ───────────────────────────────────────────────────

def build_sklearn_model():
    from sklearn.ensemble import GradientBoostingRegressor
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        ))
    ])


# ─── NLP Sentiment Scorer ───────────────────────────────────────────────────

class ReleaseNotesSentiment:
    HIGH_RISK_KEYWORDS = {
        "rewrite", "major", "breaking", "migration", "deprecated", "removed",
        "critical", "hotfix", "emergency", "vulnerability", "security patch",
        "regression", "rollback", "urgent", "incident", "overhaul", "redesign",
        "refactor", "risky", "unstable", "failure", "outage", "blocker"
    }
    LOW_RISK_KEYWORDS = {
        "minor", "patch", "fix", "improvement", "optimization", "cleanup",
        "docs", "documentation", "typo", "style", "polish", "trivial",
        "small", "cosmetic", "update", "bump"
    }

    def __init__(self):
        self._transformer = None
        self._try_load_transformer()

    def _try_load_transformer(self):
        try:
            from transformers import pipeline
            self._transformer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
            )
            logger.info("Transformer sentiment model loaded.")
        except Exception as e:
            logger.warning(f"Transformer not available, using keyword fallback: {e}")

    # Software-domain override: if text is clearly low-risk by keyword signal,
    # skip DistilBERT entirely (it was trained on movie reviews and misreads
    # software terms like "fix", "bug", "optimization" as negative sentiment)
    STRONG_LOW_RISK  = {"minor", "patch", "trivial", "cosmetic", "typo",
                        "docs", "documentation", "bump", "polish", "small"}
    STRONG_HIGH_RISK = {"breaking", "emergency", "hotfix", "vulnerability",
                        "incident", "rollback", "outage", "blocker", "critical",
                        "regression", "rewrite", "overhaul"}

    def score(self, text: Optional[str]) -> float:
        if not text or text.strip().lower() in ("string", ""):
            return 0.25  # neutral default

        text_lower = text.lower()
        high_hits        = sum(1 for kw in self.HIGH_RISK_KEYWORDS  if kw in text_lower)
        low_hits         = sum(1 for kw in self.LOW_RISK_KEYWORDS   if kw in text_lower)
        strong_high_hits = sum(1 for kw in self.STRONG_HIGH_RISK    if kw in text_lower)
        strong_low_hits  = sum(1 for kw in self.STRONG_LOW_RISK     if kw in text_lower)

        # Keyword-only score: anchored at 0.20, pushed up by risk words, down by safe words
        keyword_score = float(np.clip(
            0.20 + (high_hits * 0.13) - (low_hits * 0.06), 0.0, 1.0
        ))

        # ── Short-circuit: trust keywords over DistilBERT for clear cases ──
        # DistilBERT was trained on movie reviews; it misreads software terms
        # like "bug", "fix", "optimization" as negative with high confidence.
        if strong_high_hits >= 1 and strong_low_hits == 0:
            # Clearly high-risk language — trust keywords, skip transformer
            return float(np.clip(keyword_score + 0.20, 0.0, 1.0))

        if strong_low_hits >= 1 and strong_high_hits == 0:
            # Clearly low-risk language — trust keywords, skip transformer
            return float(np.clip(keyword_score, 0.0, 0.35))

        if self._transformer is None:
            return keyword_score

        # ── Ambiguous text: use transformer but cap its influence ──────────
        try:
            result = self._transformer(text[:512])[0]
            label  = result["label"]
            conf   = result["score"]

            logger.debug(f"NLP: label={label}, conf={conf:.3f}, kw_score={keyword_score:.3f}")

            if label == "NEGATIVE":
                # Only trust transformer negativity if keywords also signal risk
                # Without keyword support, DistilBERT false-positives on SW text
                keyword_support = high_hits / max(high_hits + low_hits + 1, 1)
                transformer_risk = 0.20 + 0.55 * conf * (0.3 + 0.7 * keyword_support)
            else:
                # POSITIVE label → low risk, scale by confidence
                transformer_risk = max(0.02, 0.20 - 0.16 * conf)

            # Keyword score always gets at least 50% weight
            t_weight = min(0.50, 0.30 + 0.25 * conf)
            k_weight = 1.0 - t_weight
            blended  = t_weight * transformer_risk + k_weight * keyword_score
            return float(np.clip(blended, 0.0, 1.0))

        except Exception:
            return keyword_score


# ─── Synthetic Data Generator ────────────────────────────────────────────────

def _generate_synthetic_data(N: int = 6000, seed: int = 42):
    """
    Generate synthetic training data with realistic distribution
    including deliberate extreme HIGH and LOW risk edge cases.
    """
    np.random.seed(seed)

    # ── Base distribution (N samples) ──────────────────────────
    X = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float32)
    X[:, 0]  = np.random.beta(5, 2, N) * 100          # test_coverage
    X[:, 1]  = np.random.beta(5, 2, N) * 100          # code_coverage
    X[:, 2]  = np.random.beta(4, 2, N) * 100          # branch_coverage
    X[:, 3]  = np.random.poisson(5, N).astype(float)  # past_defects_total
    X[:, 4]  = np.random.poisson(1, N).astype(float)  # critical_defects
    X[:, 5]  = np.random.beta(7, 2, N) * 100          # defect_resolution_rate
    X[:, 6]  = np.random.exponential(5, N)             # cyclomatic_complexity
    X[:, 7]  = np.random.exponential(3000, N)          # lines_of_code_changed
    X[:, 8]  = np.random.randint(1, 20, N).astype(float)  # num_contributors
    X[:, 9]  = np.random.beta(7, 2, N) * 100          # build_success_rate
    X[:, 10] = np.random.exponential(12, N)            # pr_review_time
    X[:, 11] = np.random.poisson(10, N).astype(float) # open_issues

    # ── Extreme HIGH risk (500 samples) ────────────────────────
    H = 500
    Xh = np.zeros((H, len(FEATURE_NAMES)), dtype=np.float32)
    Xh[:, 0]  = np.random.uniform(5,  45, H)    # very low test coverage
    Xh[:, 1]  = np.random.uniform(5,  40, H)    # very low code coverage
    Xh[:, 2]  = np.random.uniform(5,  35, H)    # very low branch coverage
    Xh[:, 3]  = np.random.uniform(15, 40, H)    # many historical defects
    Xh[:, 4]  = np.random.uniform(3,  15, H)    # many critical defects
    Xh[:, 5]  = np.random.uniform(20, 60, H)    # low resolution rate
    Xh[:, 6]  = np.random.uniform(18, 55, H)    # high complexity
    Xh[:, 7]  = np.random.uniform(8000, 30000, H)  # massive changeset
    Xh[:, 8]  = np.random.uniform(10, 25, H)    # large team (coordination risk)
    Xh[:, 9]  = np.random.uniform(45, 75, H)    # low build success
    Xh[:, 10] = np.random.uniform(48, 120, H)   # very slow PR review
    Xh[:, 11] = np.random.uniform(60, 250, H)   # many open issues
    yh = np.random.uniform(0.75, 1.0, H).astype(np.float32)

    # ── Extreme LOW risk (500 samples) ─────────────────────────
    L = 500
    Xl = np.zeros((L, len(FEATURE_NAMES)), dtype=np.float32)
    Xl[:, 0]  = np.random.uniform(85, 100, L)   # high test coverage
    Xl[:, 1]  = np.random.uniform(80, 100, L)   # high code coverage
    Xl[:, 2]  = np.random.uniform(75, 100, L)   # high branch coverage
    Xl[:, 3]  = np.random.uniform(0,  4,   L)   # few historical defects
    Xl[:, 4]  = np.zeros(L)                      # zero critical defects
    Xl[:, 5]  = np.random.uniform(90, 100, L)   # high resolution rate
    Xl[:, 6]  = np.random.uniform(1,  6,   L)   # low complexity
    Xl[:, 7]  = np.random.uniform(50, 1500, L)  # small changeset
    Xl[:, 8]  = np.random.uniform(1,  6,   L)   # small focused team
    Xl[:, 9]  = np.random.uniform(93, 100, L)   # high build success
    Xl[:, 10] = np.random.uniform(1,  12,  L)   # fast PR review
    Xl[:, 11] = np.random.uniform(0,  8,   L)   # few open issues
    yl = np.random.uniform(0.0, 0.12, L).astype(np.float32)

    # ── Combine all splits ──────────────────────────────────────
    X_all = np.concatenate([X, Xh, Xl], axis=0)
    N_all = len(X_all)

    # ── Compute derived features for ALL rows ───────────────────
    X_all[:, 12] = (X_all[:, 0] + X_all[:, 1] + X_all[:, 2]) / 3
    X_all[:, 13] = X_all[:, 3] / np.maximum(X_all[:, 7] / 1000, 0.1)
    X_all[:, 14] = X_all[:, 7] / np.maximum(X_all[:, 8], 1)
    X_all[:, 15] = np.log1p(X_all[:, 7]) * (X_all[:, 6] / 10)
    X_all[:, 16] = (
        (X_all[:, 12] / 100) * 0.4 +
        (X_all[:, 9]  / 100) * 0.3 +
        (X_all[:, 5]  / 100) * 0.3
    ) * 100

    # ── Base labels ─────────────────────────────────────────────
    y_base = (
        (1 - X[:, 12] / 100)        * 0.22 +
        np.minimum(X[:, 4] / 10, 1) * 0.22 +
        (1 - X[:, 5]  / 100)        * 0.16 +
        np.minimum(X[:, 6] / 30, 1) * 0.14 +
        (1 - X[:, 9]  / 100)        * 0.14 +
        np.minimum(X[:, 11] / 100, 1) * 0.12
    ).astype(np.float32)
    y_base = np.clip(
        y_base + np.random.normal(0, 0.04, N).astype(np.float32),
        0, 1
    )

    y_all = np.concatenate([y_base, yh, yl], axis=0)

    # Shuffle
    idx = np.random.permutation(N_all)
    return X_all[idx], y_all[idx]


# ─── Main Risk Predictor ────────────────────────────────────────────────────

class RiskPredictor:
    """
    Ensemble risk predictor combining:
    - PyTorch RiskNet      (40%)
    - Scikit-learn GBM     (40%)
    - NLP release notes    (20%)
    """

    WEIGHTS = {"pytorch": 0.40, "sklearn": 0.40, "nlp": 0.20}

    MODEL_DIR     = "./saved_models"
    SKLEARN_PATH  = "./saved_models/sklearn_gbm.pkl"
    PYTORCH_PATH  = "./saved_models/risknet.pt"

    def __init__(self):
        import os, joblib
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.risk_net      = RiskNet().to(self.device)
        self.sklearn_model = build_sklearn_model()
        self.nlp_scorer    = ReleaseNotesSentiment()

        if (os.path.exists(self.SKLEARN_PATH) and
                os.path.exists(self.PYTORCH_PATH)):
            # ── Fast path: load pre-trained weights from disk ──
            self.sklearn_model = joblib.load(self.SKLEARN_PATH)
            self.risk_net.load_state_dict(
                torch.load(self.PYTORCH_PATH, map_location=self.device)
            )
            self.risk_net.eval()
            logger.info("Models loaded from disk — skipping training")
        else:
            # ── Slow path: train and then save ─────────────────
            logger.info("No saved models found — training from scratch...")
            self._pretrain_on_synthetic()
            os.makedirs(self.MODEL_DIR, exist_ok=True)
            joblib.dump(self.sklearn_model, self.SKLEARN_PATH)
            torch.save(self.risk_net.state_dict(), self.PYTORCH_PATH)
            logger.info(f"Models saved to {self.MODEL_DIR}")

        logger.info(f"RiskPredictor ready on {self.device}")

    def _pretrain_on_synthetic(self):
        X, y = _generate_synthetic_data(N=6000)
        self.sklearn_model.fit(X, y)
        self._train_pytorch(X, y)

    def _train_pytorch(self, X: np.ndarray, y: np.ndarray, epochs: int = 80):
        self.risk_net.train()

        optimizer = torch.optim.AdamW(
            self.risk_net.parameters(), lr=5e-4, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )

        X_t = torch.tensor(X).to(self.device)
        y_t = torch.tensor(y).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True, drop_last=False
        )

        best_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.risk_net(xb)

                # Combined loss: MSE + L1 + focal penalty on extremes
                mse  = F.mse_loss(pred, yb)
                l1   = F.l1_loss(pred, yb)
                # Penalise predictions far from true extremes harder
                extreme_mask = (yb > 0.7) | (yb < 0.15)
                if extreme_mask.any():
                    extreme_loss = F.mse_loss(pred[extreme_mask], yb[extreme_mask])
                else:
                    extreme_loss = torch.tensor(0.0).to(self.device)

                loss = mse + 0.3 * l1 + 0.5 * extreme_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.risk_net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Step scheduler once per epoch (CosineAnnealingLR)
            scheduler.step()

            if epoch_loss < best_loss:
                best_loss  = epoch_loss
                best_state = {k: v.clone() for k, v in self.risk_net.state_dict().items()}

        # Restore best weights
        if best_state:
            self.risk_net.load_state_dict(best_state)
        self.risk_net.eval()

    def predict(self, metrics) -> dict:
        features = extract_features(metrics)

        # ── PyTorch ──────────────────────────────────────────────
        x = torch.tensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pt_score = float(self.risk_net(x).cpu().item())

        # ── Sklearn ──────────────────────────────────────────────
        sk_raw   = self.sklearn_model.predict(features.reshape(1, -1))[0]
        sk_score = float(np.clip(sk_raw, 0, 1))

        # ── NLP ──────────────────────────────────────────────────
        nlp_score = self.nlp_scorer.score(metrics.release_notes)

        # ── Ensemble ─────────────────────────────────────────────
        final_score = (
            self.WEIGHTS["pytorch"] * pt_score +
            self.WEIGHTS["sklearn"] * sk_score +
            self.WEIGHTS["nlp"]     * nlp_score
        )

        # Confidence = inverse of model disagreement
        scores      = [pt_score, sk_score, nlp_score]
        disagreement = float(np.std(scores))
        confidence   = float(np.clip(1.0 - disagreement * 2, 0.5, 0.99))

        risk_score = round(final_score * 100, 1)

        return {
            "risk_score":  risk_score,
            "risk_level":  self._risk_level(risk_score),
            "confidence":  round(confidence, 3),
            "breakdown": {
                "pytorch_neural_net":       round(pt_score  * 100, 1),
                "sklearn_gradient_boost":   round(sk_score  * 100, 1),
                "nlp_release_notes":        round(nlp_score * 100, 1),
                "ensemble_weights":         self.WEIGHTS,
            },
        }

    @staticmethod
    def _risk_level(score: float) -> str:
        if score < 25:  return "LOW"
        if score < 50:  return "MEDIUM"
        if score < 75:  return "HIGH"
        return "CRITICAL"
