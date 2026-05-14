import re
import sys
import time

from google import genai
from google.genai import types
from rich.console import Console

from loom.compaction import compact
from loom.state import save
from loom.tools import DECLARATIONS as BASE_DECLARATIONS, dispatch

console = Console()

# Commands that can destroy state irreversibly.
# Enforced in the harness — not left to the system prompt alone.
_DESTRUCTIVE_PATTERNS = [
    r"\brm\b",
    r"\bgit\s+reset\b",
    r"\bgit\s+clean\b",
    r"\bgit\s+push\b.+(-f\b|--force)",
    r"\btruncate\b",
    r"\bdd\b.+of=",
    r"\bmkfs\b",
]

SYSTEM_PROMPT = """
You are Loom, a coding agent running in the user's terminal.

## Tools
You have four tools: bash, read_file, write_file, str_replace.
- Use bash to explore, run tests, install packages, and check command output.
- Use read_file before any edit — never guess the current file content.
- Use str_replace for edits to existing files. Use write_file only to create new files.
- When bash fails, read the full error before retrying. Do not retry the same command unchanged.

## Behavior
- Do not greet the user or introduce yourself. Respond directly to the request.
- Before starting a task with unclear scope, ask one clarifying question.
- Before deleting files, dropping data, or making irreversible changes: state what you are about to do and ask for confirmation.
- Be concise. State what you did and what the result was. Do not narrate what you are about to do.
""".strip()

# Context window sizes in tokens, per model.
CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.0-flash-lite": 1_048_576,
    "gemini-2.0-flash":      1_048_576,
    "gemini-2.5-flash":      1_048_576,
    "gemini-2.5-pro":        1_048_576,
    "gemini-1.5-flash":      1_048_576,
    "gemini-1.5-pro":        2_097_152,
}
DEFAULT_CONTEXT_WINDOW = 1_048_576

# spawn_agent is only given to the parent agent — sub-agents do not get it,
# which prevents infinite recursion.
_SPAWN_AGENT_DECLARATION = types.FunctionDeclaration(
    name="spawn_agent",
    description=(
        "Delegate a self-contained subtask to a fresh sub-agent with its own context window. "
        "Use when a task is isolated enough to run independently — e.g. 'write tests for X', "
        "'refactor module Y', 'investigate error Z'. "
        "The sub-agent has access to bash, read_file, write_file, and str_replace. "
        "Pass only the context the sub-agent actually needs, not the full conversation."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "task": types.Schema(
                type=types.Type.STRING,
                description="The specific task for the sub-agent to complete.",
            ),
            "context": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Relevant context from the current session: file paths, prior results, "
                    "constraints, or anything the sub-agent needs to know."
                ),
            ),
        },
        required=["task"],
    ),
)

# Parent agents get base tools + spawn_agent.
_PARENT_DECLARATIONS = types.Tool(
    function_declarations=[
        *BASE_DECLARATIONS.function_declarations,
        _SPAWN_AGENT_DECLARATION,
    ]
)


def _is_destructive(command: str) -> bool:
    return any(re.search(p, command, re.IGNORECASE) for p in _DESTRUCTIVE_PATTERNS)


def _confirm(command: str) -> bool:
    """Ask the user before running a destructive command. Defaults to No."""
    console.print(f"\n  [bold red]⚠ destructive command:[/bold red] [yellow]{command}[/yellow]")
    try:
        answer = input("  Run this? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        answer = ""
    return answer == "y"


def _extract_text(content: types.Content) -> str:
    return "\n".join(p.text for p in content.parts if p.text)


def _show_usage(prompt_tokens: int, output_tokens: int, model: str) -> None:
    if not prompt_tokens:
        return
    ctx_window = CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
    pct = prompt_tokens / ctx_window * 100
    console.print(
        f"  [dim]ctx {prompt_tokens:,} tokens "
        f"({pct:.2f}% of window) · "
        f"{output_tokens:,} out[/dim]"
    )


def _stream_response(
    stream_iter,
    model: str,
) -> tuple[types.Content, types.FinishReason | None, int, int]:
    """Consume a streaming response, printing text tokens as they arrive.

    Each chunk is a partial GenerateContentResponse. Text parts are printed
    immediately via sys.stdout (bypassing Rich so there's no buffering delay).
    Function call parts arrive complete in a single chunk — not streamed.

    Returns: (assembled_content, finish_reason, prompt_token_count, output_token_count)
    """
    accumulated_parts: list[types.Part] = []
    finish_reason = None
    prompt_tokens = 0
    output_tokens = 0
    printed_text = False

    for chunk in stream_iter:
        if not chunk.candidates:
            continue

        candidate = chunk.candidates[0]

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.text:
                    sys.stdout.write(part.text)
                    sys.stdout.flush()
                    time.sleep(0.02)
                    printed_text = True
                    accumulated_parts.append(part)
                elif part.function_call:
                    # Function calls arrive whole — no streaming within a call.
                    accumulated_parts.append(part)

        if candidate.finish_reason:
            finish_reason = candidate.finish_reason

        # Usage metadata is only populated on the final chunk.
        if chunk.usage_metadata:
            if chunk.usage_metadata.prompt_token_count:
                prompt_tokens = chunk.usage_metadata.prompt_token_count
            if chunk.usage_metadata.candidates_token_count:
                output_tokens = chunk.usage_metadata.candidates_token_count

    if printed_text:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Guard: the API should never return an empty parts list, but handle it.
    if not accumulated_parts:
        accumulated_parts = [types.Part(text="")]

    return (
        types.Content(role="model", parts=accumulated_parts),
        finish_reason,
        prompt_tokens,
        output_tokens,
    )


def _spawn_subagent(
    client: genai.Client,
    task: str,
    context: str,
    model: str,
    max_tokens: int,
) -> str:
    """Run a self-contained sub-agent with a fresh context window.

    The sub-agent gets only what you pass in context — it does not see the
    parent's message history. This isolation is the whole point: a clean window
    for a focused task.
    """
    messages: list[types.Content] = []

    if context:
        # Seed the sub-agent's context as a prior exchange so it starts oriented.
        messages.append(types.Content(
            role="user",
            parts=[types.Part(text=f"[Context from parent agent]\n{context}")],
        ))
        messages.append(types.Content(
            role="model",
            parts=[types.Part(text="Understood. I have the context.")],
        ))

    messages.append(types.Content(
        role="user",
        parts=[types.Part(text=task)],
    ))

    console.print(f"\n  [bold blue]╔ sub-agent:[/bold blue] {task[:80]}")

    # is_subagent=True ensures this call uses base tools only — no spawn_agent.
    reply, _ = run_turn(client, messages, model, max_tokens, is_subagent=True)

    console.print(f"  [bold blue]╚ sub-agent done[/bold blue]\n")
    return reply or "[sub-agent produced no output]"


def _run_tool_calls(
    content: types.Content,
    client: genai.Client,
    model: str,
    max_tokens: int,
) -> types.Content:
    """Execute every FunctionCall Part in the model's response.
    Returns a single 'user' Content carrying all FunctionResponse Parts.

    All results go in one Content — the Gemini API requires this.
    """
    response_parts = []

    for part in content.parts:
        if not part.function_call:
            continue

        name = part.function_call.name
        args = dict(part.function_call.args)

        args_display = ", ".join(f"{k}={v!r}" for k, v in args.items())
        console.print(
            f"  [bold yellow]▶ {name}[/bold yellow]([cyan]{args_display}[/cyan])"
        )

        if name == "spawn_agent":
            result = _spawn_subagent(
                client=client,
                task=args["task"],
                context=args.get("context", ""),
                model=model,
                max_tokens=max_tokens,
            )
        elif name == "bash" and _is_destructive(args.get("command", "")):
            if not _confirm(args["command"]):
                result = "[blocked: user declined to run this command]"
                console.print("  [dim]blocked[/dim]")
            else:
                result = dispatch(name, args)
        else:
            result = dispatch(name, args)

        preview = result.splitlines()[0][:120] if result else ""
        if len(result) > len(preview):
            preview += " …"
        console.print(f"  [dim]{preview}[/dim]")

        response_parts.append(
            types.Part(
                function_response=types.FunctionResponse(
                    name=name,
                    response={"result": result},
                )
            )
        )

    return types.Content(role="user", parts=response_parts)


def run_turn(
    client: genai.Client,
    messages: list[types.Content],
    model: str,
    max_tokens: int,
    is_subagent: bool = False,
) -> tuple[str, int]:
    """Run one full agent turn. Returns (reply_text, final_prompt_token_count).

    is_subagent=True uses base tools only (no spawn_agent), preventing recursion.
    Text streams to stdout as tokens arrive. The full turn record is appended to messages.
    """
    last_prompt_tokens = 0
    tool_declarations = BASE_DECLARATIONS if is_subagent else _PARENT_DECLARATIONS

    while True:
        stream = client.models.generate_content_stream(
            model=model,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=max_tokens,
                tools=[tool_declarations],
            ),
        )

        content, finish_reason, prompt_tokens, output_tokens = _stream_response(stream, model)
        messages.append(content)

        if prompt_tokens:
            last_prompt_tokens = prompt_tokens
        _show_usage(prompt_tokens, output_tokens, model)

        if finish_reason == types.FinishReason.MAX_TOKENS:
            raise RuntimeError(
                f"Response truncated: hit max_output_tokens={max_tokens}."
            )

        function_calls = [p for p in content.parts if p.function_call]

        if function_calls:
            tool_result_content = _run_tool_calls(content, client, model, max_tokens)
            messages.append(tool_result_content)
            continue

        return _extract_text(content), last_prompt_tokens


def chat(
    client: genai.Client,
    model: str,
    max_tokens: int,
    compact_threshold: float = 0.70,
    initial_messages: list[types.Content] | None = None,
) -> None:
    """REPL: accumulate conversation history across turns."""
    messages: list[types.Content] = list(initial_messages) if initial_messages else []
    ctx_window = CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)

    console.print(
        f"[bold]Loom[/bold]  model=[cyan]{model}[/cyan]  "
        f"max_tokens=[cyan]{max_tokens}[/cyan]  "
        f"compact_at=[cyan]{compact_threshold:.0%}[/cyan]"
    )
    console.print("[dim]Type 'exit' or Ctrl+C to quit.[/dim]\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        start = time.perf_counter()
        try:
            reply, prompt_tokens = run_turn(client, messages, model, max_tokens)
        except RuntimeError as e:
            console.print(f"[red]Error:[/red] {e}")
            if messages and messages[-1].role == "model":
                messages.pop()
            continue

        elapsed = time.perf_counter() - start
        # Text was already streamed to stdout inside run_turn.
        # Just add a blank line for spacing.
        console.print(f"  [dim]elapsed {elapsed:.1f}s[/dim]")
        console.print()

        if prompt_tokens / ctx_window >= compact_threshold:
            messages = compact(client, messages, model)

        save(messages, model, max_tokens, compact_threshold)
