"""
MCP Orchestration Layer — Release Risk Signal Collector
═══════════════════════════════════════════════════════
Architecture:
  - GitHub MCP Server     → real data from public repos
  - JIRA MCP Server       → mock realistic data
  - Smartsheet MCP Server → mock realistic data
  - Claude API            → orchestrates all tool calls
  - Output               → 13 signals → Risk Predictor

Usage:
    python mcp_orchestrator.py --repo microsoft/vscode --version v1.89.0
    python mcp_orchestrator.py --repo pytorch/pytorch --version v2.3.0

Requirements:
    pip install anthropic requests python-dotenv

Environment:
    ANTHROPIC_API_KEY=your-key
    GITHUB_TOKEN=your-github-token  (optional, raises rate limits)
"""

import os
import json
import math
import random
import argparse
import logging
import requests
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv
import anthropic

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — SIGNAL SCHEMA
# ═══════════════════════════════════════════════════════════════

@dataclass
class ReleaseSignals:
    """
    13 signals mapped to your existing Risk Predictor schema.

    Schedule (4) + Quality (4) + Process (3) + Unstructured (2) = 13
    """
    # ── Schedule signals ─────────────────────────────────────
    milestone_trend:           float = 0.5   # 0=on_track, 0.5=slipping, 1=critical
    dependency_closure_rate:   float = 80.0  # % linked issues closed vs planned
    days_since_last_update:    int   = 3     # days since last milestone update
    critical_path_float:       float = 5.0   # days of float remaining

    # ── Quality signals ──────────────────────────────────────
    defect_density_trend:      float = 2.0   # new defects per week (rolling 4wk avg)
    test_coverage_trajectory:  float = 0.0   # +ve improving, -ve declining (% per week)
    defect_escape_rate:        float = 5.0   # % defects found post-gate
    p1_p2_open_count:          int   = 0     # open P1/P2 defect count

    # ── Process signals ──────────────────────────────────────
    ci_build_pass_rate_trend:  float = 95.0  # % CI runs passing (rolling 2wk)
    code_churn_rate:           float = 1.0   # recent churn vs 90-day baseline (ratio)
    pr_merge_velocity:         float = 5.0   # PRs merged per week (rolling 4wk)

    # ── Unstructured signals ─────────────────────────────────
    release_note_sentiment:    float = 0.25  # 0=low_risk, 1=high_risk (NLP)
    standup_risk_language:     float = 0.2   # 0=clean, 1=high_risk_terms detected

    # ── Metadata ─────────────────────────────────────────────
    repo:        str = ""
    version:     str = ""
    sources:     dict = field(default_factory=dict)
    warnings:    list = field(default_factory=list)
    collected_at: str = ""


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — GITHUB MCP SERVER (real data)
# ═══════════════════════════════════════════════════════════════

class GitHubMCPServer:
    """
    Pulls real process signals from public GitHub repos.
    Uses GitHub REST API v3 — no authentication required for public repos
    (60 req/hr unauthenticated, 5000 req/hr with GITHUB_TOKEN).
    """

    BASE = "https://api.github.com"

    def __init__(self, token: str = ""):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
            logger.info("GitHub: authenticated (5000 req/hr)")
        else:
            logger.info("GitHub: unauthenticated (60 req/hr)")

    def _get(self, path: str, params: dict = None) -> dict | list:
        url = f"{self.BASE}{path}"
        r   = requests.get(url, headers=self.headers,
                           params=params or {}, timeout=15)
        if r.status_code == 404:
            raise ValueError(f"GitHub 404: {url}")
        r.raise_for_status()
        return r.json()

    # ── Tool: get_ci_build_pass_rate ────────────────────────
    def get_ci_build_pass_rate(self, repo: str,
                               days: int = 14) -> dict:
        """
        Returns CI build pass rate % over the last N days.
        Pulls GitHub Actions workflow runs.
        """
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() + "Z"
        try:
            data  = self._get(f"/repos/{repo}/actions/runs",
                              {"per_page": 100, "created": f">={since}"})
            runs  = data.get("workflow_runs", [])
            if not runs:
                return {"pass_rate": 95.0, "total_runs": 0,
                        "source": "github_actions", "note": "no runs found"}

            total    = len(runs)
            passed   = sum(1 for r in runs
                          if r.get("conclusion") == "success")
            pass_rate = round(passed / total * 100, 1)

            return {
                "pass_rate":  pass_rate,
                "total_runs": total,
                "passed":     passed,
                "failed":     total - passed,
                "source":     "github_actions",
                "days":       days,
            }
        except Exception as e:
            logger.warning(f"CI pass rate failed: {e}")
            return {"pass_rate": 95.0, "total_runs": 0,
                    "source": "github_actions", "error": str(e)}

    # ── Tool: get_pr_merge_velocity ─────────────────────────
    def get_pr_merge_velocity(self, repo: str,
                              weeks: int = 4) -> dict:
        """
        Returns PRs merged per week over the last N weeks.
        """
        since = (datetime.now(timezone.utc) - timedelta(weeks=weeks)).isoformat() + "Z"
        try:
            prs = self._get(f"/repos/{repo}/pulls",
                            {"state": "closed", "per_page": 100,
                             "sort": "updated", "direction": "desc"})

            merged = [p for p in prs
                      if p.get("merged_at") and
                         p["merged_at"] >= since]

            velocity = round(len(merged) / weeks, 1)

            # Compute trend: compare last 2 weeks vs previous 2 weeks
            midpoint  = (datetime.now(timezone.utc) - timedelta(weeks=weeks/2)).isoformat() + "Z"
            recent    = [p for p in merged if p["merged_at"] >= midpoint]
            older     = [p for p in merged if p["merged_at"] <  midpoint]
            trend     = "increasing" if len(recent) >= len(older) else "declining"

            return {
                "velocity_per_week": velocity,
                "total_merged":      len(merged),
                "trend":             trend,
                "recent_2wk":        len(recent),
                "prior_2wk":         len(older),
                "source":            "github_pulls",
                "weeks":             weeks,
            }
        except Exception as e:
            logger.warning(f"PR velocity failed: {e}")
            return {"velocity_per_week": 5.0, "source": "github_pulls",
                    "error": str(e)}

    # ── Tool: get_code_churn_rate ───────────────────────────
    def get_code_churn_rate(self, repo: str,
                            recent_days: int = 14,
                            baseline_days: int = 90) -> dict:
        """
        Returns code churn ratio: recent additions+deletions vs baseline.
        Ratio > 1.5 = high churn (risky), < 0.8 = low activity.
        """
        try:
            # Get recent commits
            since_recent   = (datetime.now(timezone.utc) -
                               timedelta(days=recent_days)).isoformat() + "Z"
            since_baseline = (datetime.now(timezone.utc) -
                               timedelta(days=baseline_days)).isoformat() + "Z"

            recent_commits = self._get(f"/repos/{repo}/commits",
                                       {"since": since_recent,
                                        "per_page": 100})
            all_commits    = self._get(f"/repos/{repo}/commits",
                                       {"since": since_baseline,
                                        "per_page": 100})

            recent_count   = len(recent_commits)
            baseline_avg   = len(all_commits) / (baseline_days / recent_days)
            churn_ratio    = round(recent_count / max(baseline_avg, 1), 2)

            return {
                "churn_ratio":       churn_ratio,
                "recent_commits":    recent_count,
                "baseline_avg":      round(baseline_avg, 1),
                "recent_days":       recent_days,
                "baseline_days":     baseline_days,
                "interpretation":    (
                    "high_churn"    if churn_ratio > 1.5 else
                    "normal"        if churn_ratio > 0.8 else
                    "low_activity"
                ),
                "source": "github_commits",
            }
        except Exception as e:
            logger.warning(f"Code churn failed: {e}")
            return {"churn_ratio": 1.0, "source": "github_commits",
                    "error": str(e)}

    # ── Tool: get_standup_risk_language ─────────────────────
    def get_standup_risk_language(self, repo: str,
                                  days: int = 14) -> dict:
        """
        Scans recent PR comments and commit messages for risk language.
        Returns risk score 0-1 based on flagged term frequency.
        """
        RISK_TERMS = {
            "high":   ["blocker", "blocked", "critical", "urgent", "breaking",
                       "regression", "outage", "incident", "hotfix", "emergency",
                       "failed", "failure", "crash", "corrupted", "escalation"],
            "medium": ["concern", "risk", "delay", "slip", "behind", "stuck",
                       "issue", "problem", "difficult", "challenge", "worry",
                       "unclear", "unknown", "waiting", "dependency"],
            "low":    ["fixed", "resolved", "done", "complete", "shipped",
                       "merged", "approved", "good", "clean", "passing"],
        }

        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() + "Z"
        texts = []

        try:
            # Pull commit messages
            commits = self._get(f"/repos/{repo}/commits",
                                {"since": since, "per_page": 50})
            for c in commits:
                msg = c.get("commit", {}).get("message", "")
                texts.append(msg.lower())

            # Pull recent PR comments (issues API covers PR comments too)
            comments = self._get(f"/repos/{repo}/issues/comments",
                                  {"since": since, "per_page": 50})
            for c in comments:
                texts.append(c.get("body", "").lower())

        except Exception as e:
            logger.warning(f"Comment scan failed: {e}")
            return {"risk_score": 0.2, "source": "github_comments",
                    "error": str(e)}

        full_text  = " ".join(texts)
        high_hits  = sum(full_text.count(t) for t in RISK_TERMS["high"])
        medium_hits= sum(full_text.count(t) for t in RISK_TERMS["medium"])
        low_hits   = sum(full_text.count(t) for t in RISK_TERMS["low"])
        total_words= max(len(full_text.split()), 1)

        # Weighted score: high=0.15, medium=0.05, low=-0.03 per hit per 100 words
        norm        = total_words / 100
        risk_score  = (high_hits * 0.15 + medium_hits * 0.05
                       - low_hits * 0.03) / norm
        risk_score  = round(max(0.0, min(1.0, risk_score + 0.15)), 3)

        flagged = ([t for t in RISK_TERMS["high"] if t in full_text][:5])

        return {
            "risk_score":       risk_score,
            "high_risk_hits":   high_hits,
            "medium_risk_hits": medium_hits,
            "positive_hits":    low_hits,
            "flagged_terms":    flagged,
            "texts_scanned":    len(texts),
            "source":           "github_comments",
            "days":             days,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — JIRA MOCK MCP SERVER
# ═══════════════════════════════════════════════════════════════

class JIRAMockMCPServer:
    """
    Realistic mock JIRA server.
    Returns synthetic but plausible defect and schedule signals.
    Seed based on repo name for reproducibility.
    """

    def __init__(self, repo: str):
        # Seed random from repo name so same repo = same mock data
        random.seed(hash(repo) % 10000)
        self._scenario = self._pick_scenario()
        logger.info(f"JIRA Mock: scenario={self._scenario}")

    def _pick_scenario(self) -> str:
        return random.choice([
            "healthy", "healthy", "slipping",
            "at_risk", "critical"
        ])

    def get_defect_signals(self, version: str) -> dict:
        """Returns defect quality signals for a release version."""
        scenarios = {
            "healthy":  {"p1_p2": 0, "density": 1.2, "escape": 2.1,
                         "resolution": 96.0},
            "slipping": {"p1_p2": 2, "density": 3.8, "escape": 8.5,
                         "resolution": 78.0},
            "at_risk":  {"p1_p2": 4, "density": 6.2, "escape": 14.0,
                         "resolution": 61.0},
            "critical": {"p1_p2": 8, "density": 11.5, "escape": 22.0,
                         "resolution": 44.0},
        }
        s = scenarios.get(self._scenario, scenarios["healthy"])

        # Add realistic noise
        noise = lambda x, pct: round(x * (1 + random.uniform(-pct, pct)), 1)

        return {
            "version":             version,
            "p1_p2_open":          s["p1_p2"] + random.randint(0, 2),
            "defect_density_week": noise(s["density"], 0.2),
            "defect_escape_rate":  noise(s["escape"], 0.15),
            "resolution_rate":     noise(s["resolution"], 0.05),
            "total_open_bugs":     s["p1_p2"] * 4 + random.randint(5, 20),
            "bugs_this_week":      random.randint(2, 12),
            "scenario":            self._scenario,
            "source":              "jira_mock",
        }

    def get_schedule_signals(self, version: str) -> dict:
        """Returns milestone and dependency schedule signals."""
        scenarios = {
            "healthy":  {"trend": 0.1, "closure": 92.0, "float": 8.0,
                         "days_stale": 1},
            "slipping": {"trend": 0.5, "closure": 68.0, "float": 3.0,
                         "days_stale": 4},
            "at_risk":  {"trend": 0.7, "closure": 51.0, "float": 1.0,
                         "days_stale": 7},
            "critical": {"trend": 0.9, "closure": 32.0, "float": 0.0,
                         "days_stale": 12},
        }
        s = scenarios.get(self._scenario, scenarios["healthy"])

        noise = lambda x, pct: round(x * (1 + random.uniform(-pct, pct)), 2)

        trend_label = (
            "on_track"   if s["trend"] < 0.3 else
            "slipping"   if s["trend"] < 0.6 else
            "at_risk"    if s["trend"] < 0.8 else
            "critical"
        )

        return {
            "version":                version,
            "milestone_trend_score":  noise(s["trend"], 0.1),
            "milestone_trend_label":  trend_label,
            "dependency_closure_pct": noise(s["closure"], 0.08),
            "critical_path_float_days": max(0, noise(s["float"], 0.2)),
            "days_since_last_update": s["days_stale"] + random.randint(0, 2),
            "milestones_at_risk":     random.randint(0, 3) if s["trend"] > 0.4 else 0,
            "scenario":               self._scenario,
            "source":                 "jira_mock",
        }

    def get_test_coverage_trajectory(self) -> dict:
        """Returns test coverage trend over last 4 weeks."""
        trajectories = {
            "healthy":  [78, 80, 82, 85],
            "slipping": [85, 83, 80, 78],
            "at_risk":  [82, 78, 73, 68],
            "critical": [75, 68, 60, 54],
        }
        weekly = trajectories.get(self._scenario, [80, 80, 80, 80])
        weekly = [w + random.randint(-2, 2) for w in weekly]
        delta  = weekly[-1] - weekly[0]

        return {
            "weekly_coverage":   weekly,
            "current_coverage":  weekly[-1],
            "trajectory_delta":  round(delta, 1),
            "trajectory_label":  (
                "improving" if delta > 2 else
                "stable"    if delta > -2 else
                "declining"
            ),
            "source": "jira_mock",
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — SMARTSHEET MOCK MCP SERVER
# ═══════════════════════════════════════════════════════════════

class SmartsheetMockMCPServer:
    """
    Realistic mock Smartsheet server.
    Provides schedule and process tracking signals.
    """

    def __init__(self, repo: str):
        random.seed(hash(repo + "ss") % 10000)

    def get_program_health_signals(self, version: str) -> dict:
        """Returns schedule and program health signals from tracker sheet."""
        # Simulate a realistic Smartsheet program tracker row
        on_time_pct  = random.uniform(55, 98)
        risk_items   = random.randint(0, 8)
        red_items    = random.randint(0, min(risk_items, 3))

        return {
            "version":              version,
            "on_time_pct":          round(on_time_pct, 1),
            "risk_items":           risk_items,
            "red_status_items":     red_items,
            "yellow_status_items":  risk_items - red_items,
            "last_updated_days_ago": random.randint(0, 5),
            "source":               "smartsheet_mock",
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — MCP TOOL DEFINITIONS (for Claude API)
# ═══════════════════════════════════════════════════════════════

MCP_TOOLS = [
    {
        "name": "github_get_ci_pass_rate",
        "description": "Get CI/CD build pass rate from GitHub Actions for a repo over the last N days",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo":  {"type": "string",
                          "description": "GitHub repo in owner/name format"},
                "days":  {"type": "integer",
                          "description": "Number of days to look back",
                          "default": 14},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "github_get_pr_velocity",
        "description": "Get PR merge velocity (PRs merged per week) from GitHub",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo":  {"type": "string"},
                "weeks": {"type": "integer", "default": 4},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "github_get_code_churn",
        "description": "Get code churn rate comparing recent vs baseline commit activity",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo":          {"type": "string"},
                "recent_days":   {"type": "integer", "default": 14},
                "baseline_days": {"type": "integer", "default": 90},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "github_get_standup_risk",
        "description": "Scan recent PR comments and commit messages for risk language indicators",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "days": {"type": "integer", "default": 14},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "jira_get_defect_signals",
        "description": "Get defect quality signals: P1/P2 count, defect density, escape rate, resolution rate",
        "input_schema": {
            "type": "object",
            "properties": {
                "version": {"type": "string",
                            "description": "Release version label"},
            },
            "required": ["version"],
        },
    },
    {
        "name": "jira_get_schedule_signals",
        "description": "Get schedule signals: milestone trend, dependency closure rate, critical path float",
        "input_schema": {
            "type": "object",
            "properties": {
                "version": {"type": "string"},
            },
            "required": ["version"],
        },
    },
    {
        "name": "jira_get_test_coverage_trajectory",
        "description": "Get test coverage trend over last 4 weeks — improving, stable, or declining",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "smartsheet_get_program_health",
        "description": "Get program health signals from Smartsheet tracker: on-time %, risk items, red/yellow status",
        "input_schema": {
            "type": "object",
            "properties": {
                "version": {"type": "string"},
            },
            "required": ["version"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — MCP ORCHESTRATOR (Claude API)
# ═══════════════════════════════════════════════════════════════

class MCPOrchestrator:
    """
    Uses Claude API to orchestrate MCP tool calls across
    GitHub, JIRA, and Smartsheet servers.

    Claude decides which tools to call, in what order,
    and synthesizes results into 13 risk signals.
    """

    SYSTEM_PROMPT = """You are a release risk signal collector for a software program management tool.

Your job is to collect ALL 13 risk signals by calling the available tools, then synthesize them into a structured JSON response.

Call tools in this order:
1. github_get_ci_pass_rate
2. github_get_pr_velocity
3. github_get_code_churn
4. github_get_standup_risk
5. jira_get_defect_signals
6. jira_get_schedule_signals
7. jira_get_test_coverage_trajectory
8. smartsheet_get_program_health

After ALL tools complete, respond with ONLY a JSON object in this exact format:
{
  "milestone_trend": <float 0-1, 0=on_track 1=critical>,
  "dependency_closure_rate": <float 0-100, % closed>,
  "days_since_last_update": <int>,
  "critical_path_float": <float, days remaining>,
  "defect_density_trend": <float, new defects per week>,
  "test_coverage_trajectory": <float, +ve improving -ve declining>,
  "defect_escape_rate": <float 0-100, % post-gate>,
  "p1_p2_open_count": <int>,
  "ci_build_pass_rate_trend": <float 0-100>,
  "code_churn_rate": <float, ratio vs baseline>,
  "pr_merge_velocity": <float, PRs per week>,
  "release_note_sentiment": <float 0-1>,
  "standup_risk_language": <float 0-1>,
  "summary": "<2 sentence human-readable summary of overall risk>"
}

Be precise. Use the actual numbers from tool results. Do not make up data."""

    def __init__(self, repo: str, version: str):
        self.repo    = repo
        self.version = version
        self.client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Initialize MCP servers
        self.github      = GitHubMCPServer(token=GITHUB_TOKEN)
        self.jira        = JIRAMockMCPServer(repo=repo)
        self.smartsheet  = SmartsheetMockMCPServer(repo=repo)

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return JSON string result."""
        logger.info(f"Executing tool: {tool_name}({tool_input})")

        try:
            if tool_name == "github_get_ci_pass_rate":
                result = self.github.get_ci_build_pass_rate(
                    tool_input["repo"],
                    tool_input.get("days", 14)
                )
            elif tool_name == "github_get_pr_velocity":
                result = self.github.get_pr_merge_velocity(
                    tool_input["repo"],
                    tool_input.get("weeks", 4)
                )
            elif tool_name == "github_get_code_churn":
                result = self.github.get_code_churn_rate(
                    tool_input["repo"],
                    tool_input.get("recent_days", 14),
                    tool_input.get("baseline_days", 90)
                )
            elif tool_name == "github_get_standup_risk":
                result = self.github.get_standup_risk_language(
                    tool_input["repo"],
                    tool_input.get("days", 14)
                )
            elif tool_name == "jira_get_defect_signals":
                result = self.jira.get_defect_signals(
                    tool_input["version"]
                )
            elif tool_name == "jira_get_schedule_signals":
                result = self.jira.get_schedule_signals(
                    tool_input["version"]
                )
            elif tool_name == "jira_get_test_coverage_trajectory":
                result = self.jira.get_test_coverage_trajectory()
            elif tool_name == "smartsheet_get_program_health":
                result = self.smartsheet.get_program_health_signals(
                    tool_input.get("version", self.version)
                )
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            logger.info(f"  → {json.dumps(result)[:120]}...")
            return json.dumps(result)

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})

    def collect(self) -> ReleaseSignals:
        """
        Run Claude-orchestrated tool calls across all MCP servers.
        Returns populated ReleaseSignals object.
        """
        logger.info(f"\n{'═'*55}")
        logger.info(f"MCP Signal Collection: {self.repo} @ {self.version}")
        logger.info(f"{'═'*55}")

        messages = [{
            "role": "user",
            "content": (
                f"Collect all 13 risk signals for repo '{self.repo}' "
                f"version '{self.version}'. "
                f"Call all 8 tools then return the JSON summary."
            )
        }]

        # Agentic loop — Claude calls tools until done
        max_iterations = 15
        iteration      = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\nClaude iteration {iteration}...")

            response = self.client.messages.create(
                model      = "claude-sonnet-4-20250514",
                max_tokens = 2000,
                system     = self.SYSTEM_PROMPT,
                tools      = MCP_TOOLS,
                messages   = messages,
            )

            # Add assistant response to history
            messages.append({
                "role":    "assistant",
                "content": response.content,
            })

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Claude is done — extract JSON from response
                for block in response.content:
                    if hasattr(block, "text"):
                        return self._parse_signals(block.text)
                break

            elif response.stop_reason == "tool_use":
                # Execute all requested tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     result,
                        })

                # Feed results back to Claude
                messages.append({
                    "role":    "user",
                    "content": tool_results,
                })
            else:
                logger.warning(f"Unexpected stop_reason: {response.stop_reason}")
                break

        logger.warning("Max iterations reached — using defaults")
        return ReleaseSignals(repo=self.repo, version=self.version)

    def _parse_signals(self, text: str) -> ReleaseSignals:
        """Parse Claude's JSON response into ReleaseSignals."""
        # Extract JSON block
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.error("No JSON found in Claude response")
            return ReleaseSignals(repo=self.repo, version=self.version)

        try:
            data = json.loads(text[start:end])
            signals = ReleaseSignals(
                repo    = self.repo,
                version = self.version,
                collected_at = datetime.now(timezone.utc).isoformat(),

                # Schedule
                milestone_trend         = float(data.get("milestone_trend", 0.5)),
                dependency_closure_rate = float(data.get("dependency_closure_rate", 80)),
                days_since_last_update  = int(data.get("days_since_last_update", 3)),
                critical_path_float     = float(data.get("critical_path_float", 5)),

                # Quality
                defect_density_trend      = float(data.get("defect_density_trend", 2)),
                test_coverage_trajectory  = float(data.get("test_coverage_trajectory", 0)),
                defect_escape_rate        = float(data.get("defect_escape_rate", 5)),
                p1_p2_open_count          = int(data.get("p1_p2_open_count", 0)),

                # Process
                ci_build_pass_rate_trend  = float(data.get("ci_build_pass_rate_trend", 95)),
                code_churn_rate           = float(data.get("code_churn_rate", 1.0)),
                pr_merge_velocity         = float(data.get("pr_merge_velocity", 5)),

                # Unstructured
                release_note_sentiment    = float(data.get("release_note_sentiment", 0.25)),
                standup_risk_language     = float(data.get("standup_risk_language", 0.2)),

                sources = {
                    "github":      "real — GitHub REST API",
                    "jira":        "mock — synthetic realistic data",
                    "smartsheet":  "mock — synthetic realistic data",
                    "claude":      "claude-sonnet-4-20250514 orchestration",
                },
            )

            summary = data.get("summary", "")
            if summary:
                logger.info(f"\nClaude summary: {summary}")

            return signals

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}\nText: {text[:200]}")
            return ReleaseSignals(repo=self.repo, version=self.version)


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — RISK PREDICTOR BRIDGE
# ═══════════════════════════════════════════════════════════════

def signals_to_risk_metrics(signals: ReleaseSignals) -> dict:
    """
    Map 13 MCP signals → your existing Risk Predictor schema.

    Your predictor expects:
    test_coverage, code_coverage, branch_coverage,
    past_defects_total, critical_defects, defect_resolution_rate,
    cyclomatic_complexity, lines_of_code_changed, num_contributors,
    build_success_rate, avg_pr_review_time_hours, open_issues,
    release_notes
    """

    # Derive test_coverage from trajectory
    # trajectory_delta of +5 → coverage ~85, -10 → coverage ~65
    base_coverage        = 75.0
    test_coverage        = max(20, min(100, base_coverage +
                            signals.test_coverage_trajectory * 2))
    code_coverage        = test_coverage * 0.95
    branch_coverage      = test_coverage * 0.88

    # Defect resolution rate inverse of escape rate
    defect_resolution    = max(0, 100 - signals.defect_escape_rate * 2)

    # Lines of code changed proxy from churn rate
    loc_changed          = int(2000 * signals.code_churn_rate)

    # Open issues from schedule signals
    open_issues          = (signals.p1_p2_open_count * 3 +
                            int(signals.defect_density_trend * 4))

    # Complexity proxy from churn + standup risk
    complexity           = 5 + (signals.code_churn_rate * 4) + \
                           (signals.standup_risk_language * 6)

    # Contributors proxy from PR velocity
    contributors         = max(1, int(signals.pr_merge_velocity * 0.8))

    # PR review time proxy from schedule stress
    pr_review_hours      = 8 + (signals.milestone_trend * 40)

    return {
        "package_name":             signals.repo.replace("/", "-"),
        "version":                  signals.version,
        "test_coverage":            round(test_coverage, 1),
        "code_coverage":            round(code_coverage, 1),
        "branch_coverage":          round(branch_coverage, 1),
        "past_defects_total":       int(signals.defect_density_trend * 4),
        "critical_defects":         signals.p1_p2_open_count,
        "defect_resolution_rate":   round(defect_resolution, 1),
        "cyclomatic_complexity":    round(complexity, 1),
        "lines_of_code_changed":    loc_changed,
        "num_contributors":         contributors,
        "build_success_rate":       signals.ci_build_pass_rate_trend,
        "avg_pr_review_time_hours": round(pr_review_hours, 1),
        "open_issues":              open_issues,
        "release_notes":            (
            f"Milestone trend: {signals.milestone_trend:.2f}. "
            f"Dependency closure: {signals.dependency_closure_rate:.0f}%. "
            f"CI pass rate: {signals.ci_build_pass_rate_trend:.0f}%. "
            f"P1/P2 open: {signals.p1_p2_open_count}. "
            f"Code churn ratio: {signals.code_churn_rate:.2f}."
        ),
    }


def call_risk_predictor(metrics: dict,
                        api_url: str = "http://localhost:8000") -> dict:
    """Call your existing Risk Predictor API with the mapped signals."""
    try:
        r = requests.post(
            f"{api_url}/predict",
            json    = metrics,
            timeout = 30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        logger.warning("Risk Predictor API not running — returning signals only")
        return {"error": "Risk Predictor not running at " + api_url,
                "hint": "Start backend: uvicorn main:app --reload"}
    except Exception as e:
        logger.error(f"Risk Predictor call failed: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MCP Release Risk Signal Collector"
    )
    parser.add_argument("--repo",    default="microsoft/vscode",
                        help="GitHub repo (owner/name)")
    parser.add_argument("--version", default="latest",
                        help="Release version label")
    parser.add_argument("--api",     default="http://localhost:8000",
                        help="Risk Predictor API URL")
    parser.add_argument("--no-predict", action="store_true",
                        help="Collect signals only, skip Risk Predictor call")
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("\nERROR: ANTHROPIC_API_KEY not set")
        print("Set it with: $env:ANTHROPIC_API_KEY='your-key'")
        return

    # ── Step 1: Collect signals via MCP ─────────────────────
    orchestrator = MCPOrchestrator(repo=args.repo, version=args.version)
    signals      = orchestrator.collect()

    # ── Step 2: Print signal summary ────────────────────────
    print(f"\n{'═'*55}")
    print(f"COLLECTED SIGNALS: {args.repo} @ {args.version}")
    print(f"{'═'*55}")
    print(f"\n  SCHEDULE:")
    print(f"    milestone_trend:           {signals.milestone_trend:.2f}  [{signals.sources.get('jira', 'mock')}]")
    print(f"    dependency_closure_rate:   {signals.dependency_closure_rate:.1f}%")
    print(f"    days_since_last_update:    {signals.days_since_last_update}")
    print(f"    critical_path_float:       {signals.critical_path_float:.1f} days")
    print(f"\n  QUALITY:")
    print(f"    defect_density_trend:      {signals.defect_density_trend:.1f}/week")
    print(f"    test_coverage_trajectory:  {signals.test_coverage_trajectory:+.1f}%/week")
    print(f"    defect_escape_rate:        {signals.defect_escape_rate:.1f}%")
    print(f"    p1_p2_open_count:          {signals.p1_p2_open_count}")
    print(f"\n  PROCESS:")
    print(f"    ci_build_pass_rate_trend:  {signals.ci_build_pass_rate_trend:.1f}%  [github_real]")
    print(f"    code_churn_rate:           {signals.code_churn_rate:.2f}x  [github_real]")
    print(f"    pr_merge_velocity:         {signals.pr_merge_velocity:.1f}/week  [github_real]")
    print(f"\n  UNSTRUCTURED:")
    print(f"    release_note_sentiment:    {signals.release_note_sentiment:.3f}")
    print(f"    standup_risk_language:     {signals.standup_risk_language:.3f}  [github_real]")

    if args.no_predict:
        print(f"\n{'═'*55}")
        print("Signals collected. Risk Predictor call skipped (--no-predict)")
        return

    # ── Step 3: Map to Risk Predictor schema ────────────────
    metrics = signals_to_risk_metrics(signals)

    print(f"\n{'═'*55}")
    print("MAPPED TO RISK PREDICTOR SCHEMA")
    print(f"{'═'*55}")
    for k, v in metrics.items():
        if k != "release_notes":
            print(f"  {k:35s}: {v}")

    # ── Step 4: Call Risk Predictor ─────────────────────────
    print(f"\n{'═'*55}")
    print(f"CALLING RISK PREDICTOR: {args.api}/predict")
    print(f"{'═'*55}")

    result = call_risk_predictor(metrics, api_url=args.api)

    if "error" in result:
        print(f"\n  {result['error']}")
        if "hint" in result:
            print(f"  Hint: {result['hint']}")
    else:
        print(f"\n  risk_score:   {result.get('risk_score')}")
        print(f"  risk_level:   {result.get('risk_level')}")
        print(f"  confidence:   {result.get('confidence')}")
        print(f"\n  Top risk factors:")
        for f in result.get("risk_factors", [])[:4]:
            print(f"    [{f['severity'].upper():8s}] {f['name']:25s} impact={f['impact']:+.3f}")
        print(f"\n  Recommendations:")
        for r in result.get("recommendations", [])[:4]:
            print(f"    {r}")

    print(f"\n{'═'*55}\n")


if __name__ == "__main__":
    main()
