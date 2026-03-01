import { useState, useCallback } from "react";

const API_BASE = "http://localhost:8000";

// ── Design tokens ──────────────────────────────────────────────────────────
const COLORS = {
  bg: "#0a0d14",
  surface: "#111827",
  surfaceHigh: "#1a2235",
  border: "#1e2d45",
  accent: "#00d4aa",
  accentDim: "#00a882",
  accentGlow: "rgba(0, 212, 170, 0.15)",
  text: "#e2e8f0",
  textMuted: "#6b7a99",
  LOW: "#22c55e",
  MEDIUM: "#f59e0b",
  HIGH: "#f97316",
  CRITICAL: "#ef4444",
};

const RISK_CONFIG = {
  LOW: { color: COLORS.LOW, bg: "rgba(34,197,94,0.1)", label: "Low Risk", icon: "✓" },
  MEDIUM: { color: COLORS.MEDIUM, bg: "rgba(245,158,11,0.1)", label: "Medium Risk", icon: "⚠" },
  HIGH: { color: COLORS.HIGH, bg: "rgba(249,115,22,0.1)", label: "High Risk", icon: "▲" },
  CRITICAL: { color: COLORS.CRITICAL, bg: "rgba(239,68,68,0.1)", label: "Critical Risk", icon: "✕" },
};

// ── Helpers ─────────────────────────────────────────────────────────────────
const defaultMetrics = {
  package_name: "my-service",
  version: "1.0.0",
  test_coverage: 75,
  code_coverage: 70,
  branch_coverage: 65,
  past_defects_total: 8,
  critical_defects: 1,
  defect_resolution_rate: 85,
  cyclomatic_complexity: 9,
  lines_of_code_changed: 4000,
  num_contributors: 5,
  build_success_rate: 90,
  avg_pr_review_time_hours: 18,
  open_issues: 15,
  release_notes: "Performance improvements and bug fixes for the v1 release.",
};

const FIELD_GROUPS = [
  {
    title: "Package Info",
    icon: "📦",
    fields: [
      { key: "package_name", label: "Package Name", type: "text" },
      { key: "version", label: "Version", type: "text" },
    ],
  },
  {
    title: "Coverage Metrics",
    icon: "🧪",
    fields: [
      { key: "test_coverage", label: "Test Coverage %", type: "number", min: 0, max: 100 },
      { key: "code_coverage", label: "Code Coverage %", type: "number", min: 0, max: 100 },
      { key: "branch_coverage", label: "Branch Coverage %", type: "number", min: 0, max: 100 },
    ],
  },
  {
    title: "Defect History",
    icon: "🐛",
    fields: [
      { key: "past_defects_total", label: "Past Defects (3 releases)", type: "number", min: 0 },
      { key: "critical_defects", label: "Critical Defects", type: "number", min: 0 },
      { key: "defect_resolution_rate", label: "Resolution Rate %", type: "number", min: 0, max: 100 },
    ],
  },
  {
    title: "Code Quality",
    icon: "📊",
    fields: [
      { key: "cyclomatic_complexity", label: "Cyclomatic Complexity", type: "number", min: 0 },
      { key: "lines_of_code_changed", label: "LOC Changed", type: "number", min: 0 },
      { key: "num_contributors", label: "Contributors", type: "number", min: 1 },
    ],
  },
  {
    title: "CI/CD Signals",
    icon: "🔧",
    fields: [
      { key: "build_success_rate", label: "Build Success Rate %", type: "number", min: 0, max: 100 },
      { key: "avg_pr_review_time_hours", label: "Avg PR Review Time (hrs)", type: "number", min: 0 },
      { key: "open_issues", label: "Open Issues", type: "number", min: 0 },
    ],
  },
];

// ── Sub-components ───────────────────────────────────────────────────────────

function RiskGauge({ score, level }) {
  const cfg = RISK_CONFIG[level] || RISK_CONFIG.MEDIUM;
  const angle = (score / 100) * 180 - 90;

  return (
    <div style={{ textAlign: "center", padding: "24px 0" }}>
      <svg width="200" height="110" viewBox="0 0 200 110" style={{ overflow: "visible" }}>
        {/* Track */}
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke={COLORS.border}
          strokeWidth="14"
          strokeLinecap="round"
        />
        {/* Fill */}
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke={cfg.color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={`${(score / 100) * 251.2} 251.2`}
          style={{ filter: `drop-shadow(0 0 8px ${cfg.color}80)` }}
        />
        {/* Needle */}
        <g transform={`rotate(${angle}, 100, 100)`}>
          <line x1="100" y1="100" x2="100" y2="30" stroke={cfg.color} strokeWidth="2.5" strokeLinecap="round" />
          <circle cx="100" cy="100" r="5" fill={cfg.color} />
        </g>
        {/* Labels */}
        <text x="15" y="116" fill={COLORS.textMuted} fontSize="10">0</text>
        <text x="94" y="16" fill={COLORS.textMuted} fontSize="10">50</text>
        <text x="180" y="116" fill={COLORS.textMuted} fontSize="10">100</text>
      </svg>
      <div style={{ fontSize: 52, fontWeight: 900, color: cfg.color, lineHeight: 1, fontFamily: "monospace" }}>
        {score}
      </div>
      <div style={{
        display: "inline-flex", alignItems: "center", gap: 6,
        marginTop: 8, padding: "4px 16px", borderRadius: 20,
        background: cfg.bg, color: cfg.color, fontWeight: 700, fontSize: 14,
      }}>
        <span>{cfg.icon}</span>
        <span>{cfg.label}</span>
      </div>
    </div>
  );
}

function ModelBreakdownBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: COLORS.textMuted, fontSize: 12 }}>{label}</span>
        <span style={{ color: COLORS.text, fontSize: 12, fontWeight: 600 }}>{value.toFixed(1)}</span>
      </div>
      <div style={{ height: 6, background: COLORS.border, borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${value}%`, background: color,
          borderRadius: 3, transition: "width 0.8s ease",
          boxShadow: `0 0 6px ${color}80`,
        }} />
      </div>
    </div>
  );
}

function RiskFactorBar({ factor }) {
  const impact = factor.impact;
  const isPositive = impact > 0;
  const width = Math.abs(impact) * 100;
  const color = { low: COLORS.accent, medium: COLORS.MEDIUM, high: COLORS.HIGH, critical: COLORS.CRITICAL }[factor.severity] || COLORS.textMuted;

  return (
    <div style={{
      background: COLORS.surfaceHigh, borderRadius: 10, padding: "12px 16px",
      border: `1px solid ${COLORS.border}`, marginBottom: 8,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ color: COLORS.text, fontWeight: 600, fontSize: 13 }}>{factor.name}</span>
        <span style={{
          fontSize: 11, padding: "2px 8px", borderRadius: 10, fontWeight: 700,
          background: `${color}20`, color,
        }}>
          {factor.severity.toUpperCase()}
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <div style={{ flex: 1, height: 4, background: COLORS.border, borderRadius: 2, position: "relative" }}>
          <div style={{
            position: "absolute",
            left: isPositive ? "50%" : `${50 - width / 2}%`,
            width: `${width / 2}%`,
            height: "100%",
            background: color,
            borderRadius: 2,
          }} />
          <div style={{ position: "absolute", left: "50%", top: -2, width: 1, height: 8, background: COLORS.textMuted }} />
        </div>
        <span style={{ color, fontSize: 11, fontWeight: 700, minWidth: 40, textAlign: "right" }}>
          {isPositive ? "+" : ""}{(impact * 100).toFixed(0)}%
        </span>
      </div>
      <p style={{ color: COLORS.textMuted, fontSize: 11, margin: 0 }}>{factor.description}</p>
    </div>
  );
}

// ── Main App ────────────────────────────────────────────────────────────────

export default function App() {
  const [metrics, setMetrics] = useState(defaultMetrics);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("results");

  const handleChange = useCallback((key, val) => {
    setMetrics(prev => ({ ...prev, [key]: val }));
  }, []);

  const predict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        ...metrics,
        test_coverage: +metrics.test_coverage,
        code_coverage: +metrics.code_coverage,
        branch_coverage: +metrics.branch_coverage,
        past_defects_total: +metrics.past_defects_total,
        critical_defects: +metrics.critical_defects,
        defect_resolution_rate: +metrics.defect_resolution_rate,
        cyclomatic_complexity: +metrics.cyclomatic_complexity,
        lines_of_code_changed: +metrics.lines_of_code_changed,
        num_contributors: +metrics.num_contributors,
        build_success_rate: +metrics.build_success_rate,
        avg_pr_review_time_hours: +metrics.avg_pr_review_time_hours,
        open_issues: +metrics.open_issues,
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }

      const data = await res.json();
      setResult(data);
      setActiveTab("results");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [metrics]);

  const loadExample = async (type) => {
    try {
      const res = await fetch(`${API_BASE}/examples`);
      const examples = await res.json();
      setMetrics(prev => ({ ...prev, ...examples[type] }));
    } catch {
      const ex = type === "low_risk" ? {
        package_name: "auth-service", version: "2.3.1",
        test_coverage: 92, code_coverage: 88, branch_coverage: 85,
        past_defects_total: 3, critical_defects: 0, defect_resolution_rate: 100,
        cyclomatic_complexity: 4.2, lines_of_code_changed: 320, num_contributors: 3,
        build_success_rate: 98, avg_pr_review_time_hours: 4, open_issues: 2,
        release_notes: "Minor bug fixes and performance improvements",
      } : {
        package_name: "payment-gateway", version: "4.0.0",
        test_coverage: 45, code_coverage: 38, branch_coverage: 30,
        past_defects_total: 24, critical_defects: 5, defect_resolution_rate: 60,
        cyclomatic_complexity: 22.7, lines_of_code_changed: 15000, num_contributors: 18,
        build_success_rate: 72, avg_pr_review_time_hours: 72, open_issues: 87,
        release_notes: "Major rewrite of payment processing core",
      };
      setMetrics(prev => ({ ...prev, ...ex }));
    }
  };

  return (
    <div style={{
      minHeight: "100vh", background: COLORS.bg, color: COLORS.text,
      fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
        * { box-sizing: border-box; }
        input, textarea { outline: none; }
        input:focus, textarea:focus { border-color: ${COLORS.accent} !important; box-shadow: 0 0 0 2px ${COLORS.accentGlow} !important; }
        ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: ${COLORS.bg}; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 3px; }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.5 } }
        @keyframes slideIn { from { opacity:0; transform:translateY(10px) } to { opacity:1; transform:translateY(0) } }
        @keyframes spin { to { transform: rotate(360deg) } }
      `}</style>

      {/* Header */}
      <header style={{
        borderBottom: `1px solid ${COLORS.border}`,
        padding: "20px 32px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: `linear-gradient(90deg, ${COLORS.surface}, ${COLORS.bg})`,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8, background: COLORS.accent,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, boxShadow: `0 0 20px ${COLORS.accentGlow}`,
          }}>⬡</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: 1, fontFamily: "'Space Grotesk', sans-serif" }}>
              RELEASE RISK PREDICTOR
            </div>
            <div style={{ color: COLORS.textMuted, fontSize: 10, letterSpacing: 2 }}>
              PYTORCH · TRANSFORMERS · SCIKIT-LEARN
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => loadExample("low_risk")} style={btnStyle(false, "small")}>↓ Low Risk Example</button>
          <button onClick={() => loadExample("high_risk")} style={btnStyle(false, "small")}>↓ High Risk Example</button>
        </div>
      </header>

      <div style={{ display: "flex", height: "calc(100vh - 77px)" }}>

        {/* Left Panel – Form */}
        <div style={{
          width: 380, flexShrink: 0, overflowY: "auto",
          borderRight: `1px solid ${COLORS.border}`, padding: "24px",
          background: COLORS.surface,
        }}>
          {FIELD_GROUPS.map(group => (
            <div key={group.title} style={{ marginBottom: 24 }}>
              <div style={{
                display: "flex", alignItems: "center", gap: 8, marginBottom: 12,
                paddingBottom: 8, borderBottom: `1px solid ${COLORS.border}`,
              }}>
                <span>{group.icon}</span>
                <span style={{ color: COLORS.accent, fontSize: 11, fontWeight: 700, letterSpacing: 2 }}>
                  {group.title.toUpperCase()}
                </span>
              </div>
              {group.fields.map(f => (
                <div key={f.key} style={{ marginBottom: 12 }}>
                  <label style={{ display: "block", color: COLORS.textMuted, fontSize: 10, marginBottom: 4, letterSpacing: 1 }}>
                    {f.label.toUpperCase()}
                  </label>
                  <input
                    type={f.type}
                    min={f.min}
                    max={f.max}
                    value={metrics[f.key]}
                    onChange={e => handleChange(f.key, f.type === "number" ? +e.target.value : e.target.value)}
                    style={{
                      width: "100%", background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                      borderRadius: 6, color: COLORS.text, padding: "8px 12px",
                      fontSize: 13, fontFamily: "inherit", transition: "border-color 0.2s",
                    }}
                  />
                </div>
              ))}
            </div>
          ))}

          {/* Release Notes */}
          <div style={{ marginBottom: 24 }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 8, marginBottom: 12,
              paddingBottom: 8, borderBottom: `1px solid ${COLORS.border}`,
            }}>
              <span>📝</span>
              <span style={{ color: COLORS.accent, fontSize: 11, fontWeight: 700, letterSpacing: 2 }}>
                RELEASE NOTES (NLP)
              </span>
            </div>
            <textarea
              value={metrics.release_notes || ""}
              onChange={e => handleChange("release_notes", e.target.value)}
              rows={4}
              placeholder="Describe what changed in this release..."
              style={{
                width: "100%", background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                borderRadius: 6, color: COLORS.text, padding: "8px 12px",
                fontSize: 12, fontFamily: "inherit", resize: "vertical", lineHeight: 1.6,
              }}
            />
          </div>

          <button
            onClick={predict}
            disabled={loading}
            style={{
              width: "100%", padding: "14px", borderRadius: 8, fontWeight: 700,
              fontSize: 14, letterSpacing: 2, cursor: loading ? "wait" : "pointer",
              background: loading ? COLORS.border : COLORS.accent,
              color: loading ? COLORS.textMuted : COLORS.bg,
              border: "none", transition: "all 0.2s",
              fontFamily: "inherit",
              boxShadow: loading ? "none" : `0 0 20px ${COLORS.accentGlow}`,
            }}
          >
            {loading ? "ANALYZING..." : "▶ PREDICT RISK"}
          </button>
        </div>

        {/* Right Panel – Results */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px" }}>
          {error && (
            <div style={{
              background: "rgba(239,68,68,0.1)", border: "1px solid #ef4444",
              borderRadius: 10, padding: "16px", marginBottom: 20, color: "#ef4444",
            }}>
              <strong>Error:</strong> {error}
              <div style={{ fontSize: 12, marginTop: 6, color: COLORS.textMuted }}>
                Make sure the backend is running: <code>uvicorn main:app --reload</code>
              </div>
            </div>
          )}

          {!result && !loading && (
            <div style={{
              display: "flex", flexDirection: "column", alignItems: "center",
              justifyContent: "center", height: "80%", color: COLORS.textMuted,
              gap: 12,
            }}>
              <div style={{ fontSize: 64, opacity: 0.3 }}>⬡</div>
              <div style={{ fontSize: 14 }}>Configure metrics and click Predict Risk</div>
              <div style={{ fontSize: 12, opacity: 0.6 }}>
                Ensemble model: PyTorch Neural Net + Gradient Boost + DistilBERT NLP
              </div>
            </div>
          )}

          {loading && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "80%", gap: 16 }}>
              <div style={{
                width: 48, height: 48, border: `3px solid ${COLORS.border}`,
                borderTopColor: COLORS.accent, borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
              }} />
              <div style={{ color: COLORS.textMuted, fontSize: 13 }}>Running ensemble inference...</div>
            </div>
          )}

          {result && !loading && (
            <div style={{ animation: "slideIn 0.4s ease" }}>
              {/* Package header */}
              <div style={{ marginBottom: 24 }}>
                <h2 style={{ margin: 0, fontFamily: "'Space Grotesk', sans-serif", fontSize: 22 }}>
                  {result.package_name} <span style={{ color: COLORS.textMuted, fontWeight: 400 }}>v{result.version}</span>
                </h2>
                <div style={{ color: COLORS.textMuted, fontSize: 12, marginTop: 4 }}>
                  Processed in {result.processing_time_ms}ms · Confidence {(result.confidence * 100).toFixed(0)}%
                </div>
              </div>

              {/* Top row */}
              <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 20, marginBottom: 20 }}>
                {/* Gauge */}
                <div style={{ background: COLORS.surface, borderRadius: 14, border: `1px solid ${COLORS.border}`, padding: "16px 8px" }}>
                  <div style={{ fontSize: 10, color: COLORS.textMuted, letterSpacing: 2, textAlign: "center", marginBottom: 4 }}>
                    RISK SCORE
                  </div>
                  <RiskGauge score={result.risk_score} level={result.risk_level} />
                  <div style={{
                    textAlign: "center", fontSize: 11, color: COLORS.textMuted, marginTop: 8,
                  }}>
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Model breakdown */}
                <div style={{ background: COLORS.surface, borderRadius: 14, border: `1px solid ${COLORS.border}`, padding: 20 }}>
                  <div style={{ fontSize: 10, color: COLORS.textMuted, letterSpacing: 2, marginBottom: 16 }}>
                    MODEL ENSEMBLE BREAKDOWN
                  </div>
                  <ModelBreakdownBar label="🧠 PyTorch Neural Net (40%)" value={result.ml_breakdown.pytorch_neural_net} color={COLORS.accent} />
                  <ModelBreakdownBar label="🌲 Gradient Boost / Sklearn (40%)" value={result.ml_breakdown.sklearn_gradient_boost} color="#818cf8" />
                  <ModelBreakdownBar label="📝 DistilBERT NLP (20%)" value={result.ml_breakdown.nlp_release_notes} color="#f472b6" />

                  <div style={{ marginTop: 20, padding: "12px 16px", background: COLORS.bg, borderRadius: 8 }}>
                    <div style={{ fontSize: 10, color: COLORS.textMuted, marginBottom: 8 }}>ARCHITECTURE</div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {["RiskNet(17→128→64→32→1)", "GradientBoostingRegressor(n=200)", "DistilBERT-SST2"].map(tag => (
                        <span key={tag} style={{
                          fontSize: 10, padding: "3px 8px", background: COLORS.border,
                          borderRadius: 4, color: COLORS.textMuted, fontFamily: "monospace",
                        }}>{tag}</span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Tabs */}
              <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
                {["factors", "recommendations"].map(tab => (
                  <button key={tab} onClick={() => setActiveTab(tab)} style={{
                    padding: "8px 20px", borderRadius: 8, border: "none", cursor: "pointer",
                    fontFamily: "inherit", fontSize: 12, fontWeight: 600, letterSpacing: 1,
                    background: activeTab === tab ? COLORS.accent : COLORS.surface,
                    color: activeTab === tab ? COLORS.bg : COLORS.textMuted,
                    transition: "all 0.2s",
                  }}>
                    {tab === "factors" ? "⬡ RISK FACTORS" : "💡 RECOMMENDATIONS"}
                  </button>
                ))}
              </div>

              {activeTab === "factors" && (
                <div>
                  {result.risk_factors.length === 0 && (
                    <div style={{ color: COLORS.textMuted, textAlign: "center", padding: 40 }}>No significant risk factors detected.</div>
                  )}
                  {result.risk_factors.map((f, i) => (
                    <RiskFactorBar key={i} factor={f} />
                  ))}
                </div>
              )}

              {activeTab === "recommendations" && (
                <div>
                  {result.recommendations.map((rec, i) => (
                    <div key={i} style={{
                      background: COLORS.surface, borderRadius: 10,
                      border: `1px solid ${COLORS.border}`, padding: "14px 18px",
                      marginBottom: 10, fontSize: 13, lineHeight: 1.6,
                      animation: `slideIn 0.3s ease ${i * 0.05}s both`,
                    }}>
                      {rec}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function btnStyle(primary, size = "normal") {
  return {
    padding: size === "small" ? "6px 14px" : "10px 20px",
    borderRadius: 6,
    border: `1px solid ${primary ? COLORS.accent : COLORS.border}`,
    background: primary ? COLORS.accent : "transparent",
    color: primary ? COLORS.bg : COLORS.textMuted,
    cursor: "pointer",
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: 1,
    fontFamily: "inherit",
    transition: "all 0.2s",
  };
}
