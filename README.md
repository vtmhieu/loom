# Loom

A minimal CLI coding agent built from scratch as a learning project. The goal is
to understand **agent harness engineering** — the layer of code between a raw LLM
API and a useful, long-running agent — by feeling every problem directly rather than
abstracting it away.

Stack: Python 3.12+, Gemini SDK (raw), Rich, Click. No LangChain, no LlamaIndex.

---

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd loom
uv sync
export GOOGLE_API_KEY=your-key-here   # from aistudio.google.com
```

## Running

```bash
uv run loom
```

```
Options:
  --model TEXT               Gemini model to use.       [default: gemini-2.0-flash-lite]
  --max-tokens INTEGER       Max output tokens.         [default: 8096]
  --compact-threshold FLOAT  Compact at this % of window. [default: 0.70]
  --fresh                    Ignore saved session, start clean.
```

---

## How it works

Loom runs an **agent loop**: the model receives the full conversation history, decides
what to do, calls tools, gets results, and loops until it has nothing left to do.
The conversation history is the model's only memory — there is no other state.

### The loop in one diagram

```
user message
     │
     ▼
┌─────────────────────────────────┐
│  generate_content_stream()      │  ← streams tokens to terminal
│                                 │
│  finish_reason == STOP          │  → done, show response
│  finish_reason == tool_use      │  → execute tools, inject results, loop
│  finish_reason == MAX_TOKENS    │  → raise error
└─────────────────────────────────┘
```

### Tools

| Tool           | What it does |
|----------------|--------------|
| `bash`         | Run a shell command. Returns stdout + stderr + exit code. Timeout: 60s. |
| `read_file`    | Read a file's full contents. |
| `write_file`   | Create or overwrite a file. Creates parent directories automatically. |
| `str_replace`  | Replace an exact unique string in a file. `old_string` must appear exactly once. |
| `spawn_agent`  | Delegate a subtask to a fresh sub-agent with its own context window. |

`str_replace` is the primary editing tool — it only outputs the changed region, while
`write_file` forces the model to regenerate the whole file. For large files, the token
cost difference is significant.

### Context management

Every API call sends the full history. Token usage is printed after each call:

```
  ctx 4,768 tokens (0.45% of window) · 113 out
```

When usage crosses `--compact-threshold`, the harness summarizes old messages and
replaces them with the summary + the last 4 messages verbatim. The model continues
without knowing compaction happened.

### Sub-agents

`spawn_agent` runs a nested agent loop in a fresh context window, seeded with
whatever context the parent passes. The parent only sees the result — not the
sub-agent's full working memory. This isolates complex subtasks and prevents
the parent's context from growing unboundedly on long tasks.

```
parent agent (ctx: 831 tokens)
  └── spawn_agent(task="...", context="...")
        sub-agent (ctx: starts at 670 tokens, grows independently)
        ╚ returns result string
  parent continues (still ~831 tokens + result)
```

### Safety

Destructive bash commands (`rm`, `git reset`, `git push --force`, etc.) trigger a
confirmation prompt before executing. This is enforced in the harness — not just
in the system prompt — so the model cannot bypass it.

```
  ⚠ destructive command: rm hello.py
  Run this? [y/N]
```

### Persistence

After every turn, the full message history is serialized to `.loom/session.json`.
On next launch, Loom offers to resume. `--fresh` starts clean and clears the file.

---

## Project structure

```
src/loom/
  cli.py          # Click entrypoint — flags, client init, resume prompt
  loop.py         # Agent loop: streaming, tool dispatch, compaction trigger,
                  #             sub-agent runner, destructive command gate
  tools.py        # Tool implementations (bash, read_file, write_file, str_replace)
                  # + Gemini FunctionDeclarations
  compaction.py   # Summarize-and-replace context compaction
  state.py        # Checkpoint save/load (serialize Content objects to JSON)
```

---

## What each step taught

| Step | What was built | Core lesson |
|------|---------------|-------------|
| 1 | Bare loop | The message list is the model's entire memory. `stop_reason` drives the loop. |
| 2 | bash + read_file | The tool_use / tool_result dance. Tool results bloat context fast. |
| 3 | write_file + str_replace | Why `str_replace` beats `write_file` for edits: output tokens and error surface. |
| 4 | Token telemetry | Context compounds — turn N pays for all prior turns. Tool results dominate. |
| 5 | System prompt | Prompts grow from observed failures, not imagination. Safety belongs in code, not prose. |
| 6 | Compaction | You destroy information to make room. What you lose depends on how you summarize. |
| 7 | Persistence | Agent state is just the message list. Serialize it and you can resume anywhere. |
| 8 | Streaming | Tokens stream via `generate_content_stream`. Assembly happens chunk by chunk. |
| 9 | Sub-agents | Hierarchical context isolation. Parent pays only for the result, not the subtask's working memory. |
