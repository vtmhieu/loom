import os
import subprocess
from google.genai import types

MAX_OUTPUT_CHARS = 10_000


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    dropped = len(text) - MAX_OUTPUT_CHARS
    return text[:half] + f"\n...[{dropped} chars truncated]...\n" + text[-half:]


def _bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "[error: command timed out after 60s]"

    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(result.stderr)
    if result.returncode != 0:
        parts.append(f"[exit code: {result.returncode}]")

    output = "\n".join(parts).strip()
    return _truncate(output) if output else "[no output]"


def _read_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return _truncate(f.read())
    except FileNotFoundError:
        return f"[error: file not found: {path}]"
    except PermissionError:
        return f"[error: permission denied: {path}]"
    except UnicodeDecodeError:
        return f"[error: not valid UTF-8: {path}]"


def _write_file(path: str, content: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    lines = content.count("\n") + 1
    return f"[written {lines} lines ({len(content)} bytes) → {path}]"


def _str_replace(path: str, old_string: str, new_string: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return f"[error: file not found: {path}]"

    count = content.count(old_string)
    if count == 0:
        return (
            "[error: old_string not found in file. "
            "Check for whitespace or indentation differences.]"
        )
    if count > 1:
        return (
            f"[error: old_string appears {count} times — must be unique. "
            "Add more surrounding lines to make it unambiguous.]"
        )

    new_content = content.replace(old_string, new_string, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    old_lines = old_string.count("\n") + 1
    new_lines = new_string.count("\n") + 1
    return f"[replaced {old_lines} line(s) with {new_lines} line(s) in {path}]"


def _grep(pattern: str, path: str = ".") -> str:
    try:
        result = subprocess.run(
            ["grep", "-rnI", pattern, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return _truncate(result.stdout)
        elif result.returncode == 1:
            return f"[no results found for '{pattern}']"
        else:
            return f"[grep error: {result.stderr}]"
    except Exception as e:
        return f"[error: {e}]"


REGISTRY: dict[str, callable] = {
    "bash": lambda args: _bash(args["command"]),
    "read_file": lambda args: _read_file(args["path"]),
    "write_file": lambda args: _write_file(args["path"], args["content"]),
    "str_replace": lambda args: _str_replace(
        args["path"], args["old_string"], args["new_string"]
    ),
    "grep": lambda args: _grep(args["pattern"], args.get("path", ".")),
}


def dispatch(name: str, args: dict) -> str:
    if name not in REGISTRY:
        return f"[error: unknown tool '{name}']"
    try:
        return REGISTRY[name](args)
    except Exception as e:
        return f"[error: {e}]"


DECLARATIONS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="bash",
            description=(
                "Run a shell command in the current working directory. "
                "Returns stdout and stderr combined. Non-zero exit codes are "
                "included in the output. Timeout: 60s."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "command": types.Schema(
                        type=types.Type.STRING,
                        description="The shell command to execute.",
                    )
                },
                required=["command"],
            ),
        ),
        types.FunctionDeclaration(
            name="read_file",
            description=(
                "Read the full contents of a file. "
                "Use this before editing to see the current state."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(
                        type=types.Type.STRING,
                        description="Path to the file, relative or absolute.",
                    )
                },
                required=["path"],
            ),
        ),
        types.FunctionDeclaration(
            name="write_file",
            description=(
                "Create or overwrite a file. "
                "Creates parent directories automatically."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(
                        type=types.Type.STRING,
                        description="Path to the file.",
                    ),
                    "content": types.Schema(
                        type=types.Type.STRING,
                        description="The full content of the file.",
                    ),
                },
                required=["path", "content"],
            ),
        ),
        types.FunctionDeclaration(
            name="str_replace",
            description=(
                "Replace an exact unique string in a file. "
                "old_string must appear exactly once."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(
                        type=types.Type.STRING,
                        description="Path to the file.",
                    ),
                    "old_string": types.Schema(
                        type=types.Type.STRING,
                        description="The string to replace.",
                    ),
                    "new_string": types.Schema(
                        type=types.Type.STRING,
                        description="The replacement string.",
                    ),
                },
                required=["path", "old_string", "new_string"],
            ),
        ),
        types.FunctionDeclaration(
            name="grep",
            description=(
                "Search for a pattern in files recursively."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "pattern": types.Schema(
                        type=types.Type.STRING,
                        description="The grep pattern.",
                    ),
                    "path": types.Schema(
                        type=types.Type.STRING,
                        description="The directory or file to search. Defaults to '.'",
                    ),
                },
                required=["pattern"],
            ),
        ),
    ]
)
