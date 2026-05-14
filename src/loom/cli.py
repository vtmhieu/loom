import click
from google import genai
from rich.console import Console

from loom.loop import chat
from loom.state import clear, load, restore

console = Console()


@click.command()
@click.version_option(version="0.1.0")
@click.option(
    "--model",
    default="gemini-3.1-flash-lite",
    show_default=True,
    help="Gemini model to use.",
)
@click.option(
    "--max-tokens",
    default=8096,
    show_default=True,
    help="Max output tokens per model response.",
)
@click.option(
    "--compact-threshold",
    default=0.70,
    show_default=True,
    help="Compact context when this fraction of the window is used. Lower for testing (e.g. 0.05).",
)
@click.option(
    "--fresh",
    is_flag=True,
    default=False,
    help="Ignore any saved session and start fresh.",
)
def main(model: str, max_tokens: int, compact_threshold: float, fresh: bool) -> None:
    """Loom — a minimal coding agent."""
    client = genai.Client()
    initial_messages = None

    if not fresh:
        checkpoint = load()
        if checkpoint:
            updated = checkpoint.get("updated_at", "unknown time")
            msg_count = len(checkpoint.get("messages", []))
            console.print(
                f"[dim]Found session from [bold]{updated}[/bold] "
                f"({msg_count} messages).[/dim]"
            )
            answer = input("Resume? [y/N] ").strip().lower()
            if answer == "y":
                initial_messages = restore(checkpoint)
                console.print(f"[dim]Resumed with {len(initial_messages)} messages.[/dim]\n")
            else:
                clear()

    chat(client, model, max_tokens, compact_threshold, initial_messages)
