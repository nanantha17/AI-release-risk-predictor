"""
Release Risk Predictor — MCP Client
=====================================
Connects GitHub MCP server → Risk Predictor ML engine via Claude.

Flow:
  User asks about a repo/PR
       ↓
  Claude calls GitHub MCP tools (list PRs, get commits, get issues)
       ↓
  Claude calls Risk Predictor MCP tools (score_release_risk)
       ↓
  Claude returns a conversational risk report

Run:
  1. Start FastAPI backend:   uvicorn main:app --port 8000  (in /backend)
  2. Set env vars in .env
  3. Run:  uv run mcp_client.py

.env required:
  ANTHROPIC_API_KEY=sk-ant-...
  CLAUDE_MODEL=claude-opus-4-6
  GITHUB_TOKEN=ghp_...          ← GitHub Personal Access Token
  RISK_API_URL=http://localhost:8000
"""

import asyncio
import sys
import os
import json
import httpx
from dotenv import load_dotenv
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")
RISK_API_URL      = os.getenv("RISK_API_URL", "http://localhost:8000")

assert ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY missing in .env"
assert GITHUB_TOKEN,      "GITHUB_TOKEN missing in .env"

# ── Risk Predictor API bridge ─────────────────────────────────────────────────

async def call_risk_api(payload: dict) -> dict:
    """
    Call the local FastAPI /predict endpoint.
    This bridges the MCP agent to your ML ensemble.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{RISK_API_URL}/predict", json=payload)
        resp.raise_for_status()
        return resp.json()


async def check_risk_api_health() -> bool:
    """Verify the FastAPI backend is running before starting the agent."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{RISK_API_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False


# ── Tool executor ─────────────────────────────────────────────────────────────

async def execute_tool(session: ClientSession, tool_name: str, tool_input: dict) -> str:
    """
    Route tool calls:
    - score_release_risk  → local FastAPI ML engine
    - everything else     → GitHub MCP server
    """
    if tool_name == "score_release_risk":
        # ── Route to your ML engine ──────────────────────────────
        try:
            result = await call_risk_api(tool_input)
            return json.dumps(result, indent=2)
        except httpx.ConnectError:
            return json.dumps({
                "error": f"Risk API unreachable at {RISK_API_URL}. "
                         "Start backend with: uvicorn main:app --port 8000"
            })
        except httpx.HTTPStatusError as e:
            return json.dumps({"error": f"Risk API error {e.response.status_code}: {e.response.text}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    else:
        # ── Route to GitHub MCP server ───────────────────────────
        result = await session.call_tool(tool_name, tool_input)
        if result.content:
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts) if parts else str(result.content)
        return "No result returned from tool."


# ── Agentic loop ──────────────────────────────────────────────────────────────

async def run_agent(session: ClientSession, client: anthropic.Anthropic, user_message: str):
    """
    Multi-turn agentic loop:
    Claude ↔ GitHub MCP tools ↔ Risk Predictor ML engine
    """

    # ── Fetch available GitHub tools from MCP server ─────────────
    tools_response = await session.list_tools()
    github_tools = [
        {
            "name":        t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {"type": "object", "properties": {}}
        }
        for t in tools_response.tools
    ]

    # ── Add Risk Predictor as a custom tool ───────────────────────
    risk_tool = {
        "name": "score_release_risk",
        "description": (
            "Score the release risk of a software patch or repository using an ensemble ML model "
            "(PyTorch neural net + Gradient Boosting + NLP). "
            "Call this after gathering GitHub metrics to get a risk score, confidence level, "
            "top risk factors, and actionable recommendations. "
            "Map GitHub data to these fields as best you can — use 0 or reasonable defaults for missing values."
        ),
        "input_schema": {
            "type": "object",
            "required": [
                "package_name", "version",
                "test_coverage", "code_coverage", "branch_coverage",
                "past_defects_total", "critical_defects", "defect_resolution_rate",
                "cyclomatic_complexity", "lines_of_code_changed", "num_contributors",
                "build_success_rate", "avg_pr_review_time_hours", "open_issues"
            ],
            "properties": {
                "package_name":             {"type": "string",  "description": "Repo or package name"},
                "version":                  {"type": "string",  "description": "Version or PR/commit ref"},
                "test_coverage":            {"type": "number",  "description": "Test coverage % (0-100). Use 70 if unknown."},
                "code_coverage":            {"type": "number",  "description": "Code coverage % (0-100). Use 70 if unknown."},
                "branch_coverage":          {"type": "number",  "description": "Branch coverage % (0-100). Use 65 if unknown."},
                "past_defects_total":       {"type": "integer", "description": "Closed bug issues in last 3 releases"},
                "critical_defects":         {"type": "integer", "description": "Open issues labelled critical/P0/blocker"},
                "defect_resolution_rate":   {"type": "number",  "description": "% closed issues vs total (0-100)"},
                "cyclomatic_complexity":    {"type": "number",  "description": "Avg complexity. Use 10 if unknown."},
                "lines_of_code_changed":    {"type": "integer", "description": "Total lines added + deleted in this release"},
                "num_contributors":         {"type": "integer", "description": "Number of unique contributors"},
                "build_success_rate":       {"type": "number",  "description": "CI pass rate % (0-100). Use 85 if unknown."},
                "avg_pr_review_time_hours": {"type": "number",  "description": "Avg hours from PR open to merge"},
                "open_issues":              {"type": "integer", "description": "Total open issues right now"},
                "release_notes":            {"type": "string",  "description": "Latest commit messages or changelog (optional)"}
            }
        }
    }

    all_tools = github_tools + [risk_tool]

    # ── System prompt ─────────────────────────────────────────────
    system_prompt = """You are a Release Risk Intelligence Agent for software engineering teams.

Your workflow when asked about a GitHub repo or PR:
1. Use GitHub tools to gather: open issues, recent commits, PRs, contributors, CI status
2. Map the GitHub data to release metrics (LOC changed, open issues, contributors, etc.)
3. Call score_release_risk with the mapped metrics to get the ML risk score
4. Present a clear, actionable risk report:
   - Risk score + level (LOW/MEDIUM/HIGH/CRITICAL)
   - Top 3 risk factors driving the score
   - Specific recommendations
   - Go/NoGo recommendation

When mapping GitHub data to risk metrics:
- lines_of_code_changed: sum of additions + deletions from recent commits
- num_contributors: count unique authors in recent commits  
- open_issues: current open issue count
- critical_defects: issues with labels like bug, critical, P0, blocker
- past_defects_total: closed bug issues in the last 30 days
- defect_resolution_rate: (closed issues / total issues) * 100
- avg_pr_review_time_hours: estimate from PR created_at vs merged_at
- Use sensible defaults (test_coverage=70, build_success_rate=85) when CI data is unavailable

Always be concrete — give the actual numbers, not just "high" or "low"."""

    messages = [{"role": "user", "content": user_message}]

    print(f"\n Agent thinking...\n")

    # ── Agentic loop ──────────────────────────────────────────────
    while True:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            tools=all_tools,
            messages=messages,
        )

        # Collect assistant message
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract and print final text response
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\n{'='*60}")
                    print(block.text)
                    print(f"{'='*60}\n")
            break

        if response.stop_reason == "tool_use":
            tool_results = []

            # Fire all tool calls concurrently instead of one at a time
            tasks = [
                execute_tool(session, block.name, block.input)
                for block in tool_calls
            ]
            results = await asyncio.gather(*tasks)

            tool_results = [
                {
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                }
                for block, result in zip(tool_calls, results)
            
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason
            print(f" Unexpected stop_reason: {response.stop_reason}")
            break
    # avoids sequential tool execution:
    if response.stop_reason == "tool_use":
        tool_calls = [b for b in response.content if b.type == "tool_use"]

        # Execute independent tools in parallel
        async def execute_one(block):
            result = await execute_tool(session, block.name, block.input)
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            }

        # asyncio.gather runs all tool calls concurrently
        tool_results = await asyncio.gather(*[execute_one(b) for b in tool_calls])
        messages.append({"role": "user", "content": list(tool_results)})

# ── CLI ───────────────────────────────────────────────────────────────────────

async def main():
    # Check ML backend is running
    print(" Checking Risk Predictor API...")
    healthy = await check_risk_api_health()
    if not healthy:
        print(f"""
  Warning: Risk API not reachable at {RISK_API_URL}
   Start it first:
     cd backend
     uvicorn main:app --port 8000

   Continuing anyway — risk scoring will fail until it's running.
""")
    else:
        print(f" Risk Predictor API is running at {RISK_API_URL}\n")

    # Connect to GitHub MCP server
    github_server = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={
            **os.environ,
            "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN,
        }
    )

    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print(" Connecting to GitHub MCP server...")
    async with stdio_client(github_server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(" GitHub MCP server connected\n")

            # List available GitHub tools
            tools = await session.list_tools()
            print(f" Available GitHub tools: {[t.name for t in tools.tools]}\n")

            # ── Interactive CLI loop ───────────────────────────────
            print("Release Risk Intelligence Agent")
            print("Type 'exit' to quit\n")
            print("Example queries:")
            print("  • Assess the release risk for github.com/owner/repo")
            print("  • What's the risk of merging the latest PRs in owner/repo?")
            print("  • Score the release risk for the main branch of owner/repo\n")

            while True:
                try:
                    user_input = input(">> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break

                await run_agent(session, anthropic_client, user_input)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
