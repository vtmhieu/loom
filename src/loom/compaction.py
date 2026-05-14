from google import genai
from google.genai import types
from rich.console import Console

console = Console()

# Keep this many recent messages verbatim after compaction.
# Everything before this window gets replaced by the summary.
KEEP_LAST = 4  # keep last 4 messages (~2 turns) verbatim; rest goes into summary

_SUMMARY_PROMPT = """\
You are summarizing a coding agent session so the agent can continue with a compressed context.

Write a dense, factual briefing covering:
- The user's original goal and any follow-up requests
- Every file created, modified, or deleted (exact paths + what changed)
- Commands run and their key outcomes or errors
- Decisions made and why
- What is complete and what still needs to be done

Rules:
- Use exact file paths, function names, and error messages — no paraphrasing
- Do not narrate or explain; state facts only
- This summary fully replaces the conversation history, so omit nothing that affects future actions\
"""


def compact(
    client: genai.Client,
    messages: list[types.Content],
    model: str,
) -> list[types.Content]:
    """Summarize old messages and return a shorter history.

    New history structure:
      [user: summary]  [model: ack]  [... last KEEP_LAST messages verbatim ...]

    If history is too short to compact, returns messages unchanged.
    """
    if len(messages) <= KEEP_LAST:
        console.print("  [dim]compaction triggered but history too short to summarize — skipping[/dim]")
        return messages

    to_summarize = messages[:-KEEP_LAST]
    to_keep = messages[-KEEP_LAST:]

    console.print("\n  [bold dim]compacting context…[/bold dim]")

    summary_text = _summarize(client, model, to_summarize)

    # Frame the summary as a user message so the conversation remains valid
    # (Gemini requires alternating user/model roles).
    # The ack message keeps the role alternation intact for to_keep[0].
    summary_msg = types.Content(
        role="user",
        parts=[types.Part(
            text=f"[Summary of earlier conversation — treat as ground truth]\n\n{summary_text}"
        )],
    )
    ack_msg = types.Content(
        role="model",
        parts=[types.Part(text="Understood. Continuing with the context above.")],
    )

    new_messages = [summary_msg, ack_msg] + list(to_keep)

    old_count = len(messages)
    new_count = len(new_messages)
    console.print(
        f"  [dim]compacted {old_count} → {new_count} messages "
        f"(kept last {len(to_keep)})[/dim]\n"
    )

    return new_messages


def _summarize(
    client: genai.Client,
    model: str,
    messages: list[types.Content],
) -> str:
    """Call the model to produce a summary of the given messages."""
    history_text = _format_history(messages)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=f"{_SUMMARY_PROMPT}\n\n---\n\n{history_text}")],
            )
        ],
        # No tools — this is a pure summarization call.
        config=types.GenerateContentConfig(max_output_tokens=2048),
    )
    return response.text or "[summary unavailable]"


def _format_history(messages: list[types.Content]) -> str:
    """Render Content objects as readable text for the summarization prompt.

    Tool results are truncated — the summary prompt doesn't need full bash
    output, just the gist of what happened.
    """
    lines = []
    for msg in messages:
        role = msg.role.upper()
        for part in msg.parts:
            if part.text:
                lines.append(f"{role}: {part.text}")
            elif part.function_call:
                args = dict(part.function_call.args)
                lines.append(f"{role} [tool_call]: {part.function_call.name}({args})")
            elif part.function_response:
                result = part.function_response.response.get("result", "")
                if len(result) > 500:
                    result = result[:500] + "…[truncated]"
                lines.append(
                    f"{role} [tool_result]: {part.function_response.name} → {result}"
                )
    return "\n\n".join(lines)
