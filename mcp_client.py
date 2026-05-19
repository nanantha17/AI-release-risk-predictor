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
import html
import re
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

# ── API Response Validation ───────────────────────────────────────────────────
def sanitize_text(raw: str, max_length: int = 2000) -> str:
    """Sanitize free-text from GitHub before passing to Claude or ML model."""
    if not isinstance(raw, str):
        return ""
    # Truncate
    raw = raw[:max_length]
    # Remove control characters
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
    # Escape HTML
    raw = html.escape(raw)
    # Prompt injection prevention
    for pattern in ['ignore previous instructions', 'ignore all prior',
                    'system prompt', 'you are now', 'disregard']:
        raw = re.sub(pattern, '[removed]', raw, flags=re.IGNORECASE)
    return raw.strip()


def validate_risk_payload(payload: dict) -> dict:
    """
    Validate and clamp all fields before sending to Risk Predictor ML model.
    Called inside execute_tool() when tool_name == 'score_release_risk'.
    """
    # Numeric range clamps — values outside these are data errors
    numeric_ranges = {
        "test_coverage":            (0.0,   100.0),
        "code_coverage":            (0.0,   100.0),
        "branch_coverage":          (0.0,   100.0),
        "past_defects_total":       (0,     10000),
        "critical_defects":         (0,     1000),
        "defect_resolution_rate":   (0.0,   100.0),
        "cyclomatic_complexity":    (0.0,   200.0),
        "lines_of_code_changed":    (0,     1000000),
        "num_contributors":         (0,     10000),
        "build_success_rate":       (0.0,   100.0),
        "avg_pr_review_time_hours": (0.0,   720.0),  # max 30 days
        "open_issues":              (0,     100000),
    }

    cleaned = dict(payload)  # copy — don't mutate original

    for field, (min_val, max_val) in numeric_ranges.items():
        if field not in cleaned:
            print(f"   [VALIDATION] Missing field: {field} — using safe default")
            cleaned[field] = (min_val + max_val) / 2
            continue

        try:
            val = float(cleaned[field])
            clamped = max(min_val, min(max_val, val))
            if clamped != val:
                print(f"   [VALIDATION] {field} clamped: {val} → {clamped}")
            # Preserve int type for integer fields
            cleaned[field] = int(clamped) if isinstance(min_val, int) else clamped
        except (TypeError, ValueError) as e:
            print(f"   [VALIDATION] {field} invalid value '{cleaned[field]}': {e}")
            cleaned[field] = (min_val + max_val) / 2

    # Sanitize text fields
    if "release_notes" in cleaned:
        cleaned["release_notes"] = sanitize_text(
            cleaned.get("release_notes", ""),
            max_length=2000
        )

    if "package_name" in cleaned:
        # Package name should only contain safe characters
        cleaned["package_name"] = re.sub(
            r'[^a-zA-Z0-9\-_./]', '',
            str(cleaned.get("package_name", "unknown"))
        )[:100]

    return cleaned





# ── Risk Predictor API bridge ─────────────────────────────────────────────────

async def call_risk_api(payload: dict) -> dict:
    """
    Call the local FastAPI /predict endpoint.
    Uses explicit timeout and TLS verification.
    """
    async with httpx.AsyncClient(
        timeout=30,
        verify=True,        # explicit TLS verification — never False
        follow_redirects=False  # don't silently follow redirects
    ) as client:
        resp = await client.post(
            f"{RISK_API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
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
            validated_input = validate_risk_payload(tool_input)
            changed = {k: (tool_input.get(k), validated_input[k])
                       for k in validated_input
                       if validated_input[k] != tool_input.get(k)}
            if changed:
                print(f"   [VALIDATION] Fields modified: {list(changed.keys())}")

            result = await call_risk_api(validated_input)  # use validated, not raw
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

async def run_agent(session: ClientSession,
                    client: anthropic.Anthropic,
                    user_message: str):
    """
    Multi-turn agentic loop:
    Claude ↔ GitHub MCP tools ↔ Risk Predictor ML engine
    """

    # ── Fetch available GitHub tools from MCP server ──────────────
    tools_response = await session.list_tools()
    github_tools = [
        {
            "name":         t.name,
            "description":  t.description or "",
            "input_schema": t.inputSchema if hasattr(t, "inputSchema")
                           else {"type": "object", "properties": {}}
        }
        for t in tools_response.tools
    ]

    # ── Add Risk Predictor as a custom tool ───────────────────────
    risk_tool = {
        "name": "score_release_risk",
        "description": (
            "Score the release risk of a software patch or repository "
            "using an ensemble ML model (PyTorch neural net + Gradient "
            "Boosting + NLP). Call this after gathering GitHub metrics "
            "to get a risk score, confidence level, top risk factors, "
            "and actionable recommendations."
        ),
        "input_schema": {
            "type": "object",
            "required": [
                "package_name", "version",
                "test_coverage", "code_coverage", "branch_coverage",
                "past_defects_total", "critical_defects",
                "defect_resolution_rate", "cyclomatic_complexity",
                "lines_of_code_changed", "num_contributors",
                "build_success_rate", "avg_pr_review_time_hours",
                "open_issues"
            ],
            "properties": {
                "package_name":             {"type": "string"},
                "version":                  {"type": "string"},
                "test_coverage":            {"type": "number"},
                "code_coverage":            {"type": "number"},
                "branch_coverage":          {"type": "number"},
                "past_defects_total":       {"type": "integer"},
                "critical_defects":         {"type": "integer"},
                "defect_resolution_rate":   {"type": "number"},
                "cyclomatic_complexity":    {"type": "number"},
                "lines_of_code_changed":    {"type": "integer"},
                "num_contributors":         {"type": "integer"},
                "build_success_rate":       {"type": "number"},
                "avg_pr_review_time_hours": {"type": "number"},
                "open_issues":              {"type": "integer"},
                "release_notes":            {"type": "string"}
            }
        }
    }

    all_tools = github_tools + [risk_tool]

    # ── System prompt ─────────────────────────────────────────────
    system_prompt = """You are a Release Risk Intelligence Agent.

Your workflow when asked about a GitHub repo or PR:
1. Use GitHub tools to gather: open issues, recent commits, PRs, contributors, CI status
2. Map the GitHub data to release metrics
3. Call score_release_risk with the mapped metrics
4. Present a clear, actionable risk report:
   - Risk score + level (LOW/MEDIUM/HIGH/CRITICAL)
   - Top 3 risk factors
   - Specific recommendations
   - Go/NoGo recommendation

Always be concrete — give actual numbers, not just high/low."""

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

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\n{'='*60}")
                    print(block.text)
                    print(f"{'='*60}\n")
            break

        elif response.stop_reason == "tool_use":
            tool_calls = [b for b in response.content
                         if b.type == "tool_use"]

            for block in tool_calls:
                print(f"   Calling: {block.name}")
                if block.name == "score_release_risk":
                    print(f"      Running ML ensemble...")

            # Execute all tool calls concurrently
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
            ]

            messages.append({"role": "user", "content": tool_results})

        else:
            print(f"Unexpected stop_reason: {response.stop_reason}")
            break
# ── CLI ───────────────────────────────────────────────────────────────────────

async def main():
    # Check ML backend is running
    print(" Checking Risk Predictor API...")
    healthy = await check_risk_api_health()
    if not healthy:
        print(f"""
⚠️  Warning: Risk API not reachable at {RISK_API_URL}
   Start it first:
     cd backend
     uvicorn main:app --port 8000

   Continuing anyway — risk scoring will fail until it's running.
""")
    else:
        print(f"✅ Risk Predictor API is running at {RISK_API_URL}\n")

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
