"""
Telegram bot for Loom — brainstorm, create issues, run tests, open PRs.

Run:  uv run python telegram_bot.py
Stop: Ctrl+C
"""
import json
import os
import subprocess
import urllib.request
from google import genai
from google.genai import types

TELEGRAM_TOKEN = "8923889673:AAGdahyVrZq2brGIKiAzJrmbFcsQjqj0umI"
CHAT_ID = 1701296613
REPO = "vtmhieu/loom"

client = genai.Client()

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="bash",
        description="Run a shell command. Returns stdout + stderr. Timeout 60s.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "command": types.Schema(type=types.Type.STRING),
            },
            required=["command"],
        ),
    )
])

SYSTEM = f"""You are a coding assistant for the GitHub repository {REPO}.
You receive messages from the owner via Telegram and can run shell commands.
You have gh CLI, git, uv, and pytest available.

You can:
- Brainstorm improvements to the codebase
- Create GitHub issues
- Write code changes, run tests, push branches, open PRs
- Answer questions about the code

Rules:
- Always run `gh auth status` first to confirm auth works before gh commands
- Keep Telegram replies short — bullet points, not essays
- If asked to implement something: do it, run tests, open a PR, report the PR URL
- Never push if tests fail
""".strip()


def _http(url: str, data: dict | None = None) -> dict:
    body = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, body, headers)
    with urllib.request.urlopen(req, timeout=35) as r:
        return json.loads(r.read())


def send(text: str) -> None:
    _http(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        {"chat_id": CHAT_ID, "text": text},
    )


def get_updates(offset: int | None) -> list:
    qs = f"timeout=30{f'&offset={offset}' if offset else ''}"
    data = _http(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates?{qs}")
    return data.get("result", [])


def bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        out = (r.stdout + r.stderr).strip()
        return out[:3000] if out else f"[exit {r.returncode}, no output]"
    except subprocess.TimeoutExpired:
        return "[timed out after 60s]"
    except Exception as e:
        return f"[error: {e}]"


def handle(user_text: str) -> str:
    messages: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_text)])
    ]

    while True:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                tools=[TOOLS],
            ),
        )

        content = response.candidates[0].content
        messages.append(content)

        calls = [p for p in content.parts if p.function_call]

        if not calls:
            return "\n".join(p.text for p in content.parts if p.text).strip()

        result_parts = []
        for part in calls:
            name = part.function_call.name
            args = dict(part.function_call.args)
            print(f"  tool: {name}({list(args.values())[0][:80] if args else ''})")
            result = bash(args["command"]) if name == "bash" else f"unknown tool: {name}"
            result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=name, response={"result": result}
                    )
                )
            )

        messages.append(types.Content(role="user", parts=result_parts))


def main() -> None:
    print(f"Loom bot running. Listening for messages from chat {CHAT_ID}...")
    send("🤖 Loom Bot online!\n\nSend me anything:\n• 'brainstorm ideas'\n• 'add --version flag to cli'\n• 'what's in the trace module?'\n• 'run the tests'")

    offset: int | None = None
    while True:
        try:
            for update in get_updates(offset):
                offset = update["update_id"] + 1
                text = update.get("message", {}).get("text", "")
                if not text:
                    continue
                print(f"← {text}")
                send("⏳ On it...")
                try:
                    reply = handle(text)
                    send(reply)
                    print(f"→ {reply[:100]}")
                except Exception as e:
                    send(f"❌ {e}")
        except KeyboardInterrupt:
            send("👋 Bot stopped.")
            break
        except Exception as e:
            print(f"poll error: {e}")


if __name__ == "__main__":
    main()
