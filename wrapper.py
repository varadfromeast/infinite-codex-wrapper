#!/usr/bin/env python3

import argparse
import json
import os
import re
import select
import shutil
import signal
import sys
import termios
import tty
from pathlib import Path

import pexpect
import tiktoken

DEFAULT_MAX_CONTEXT_TOKENS = int(
    os.environ.get("INFINITE_CODEX_MAX_CONTEXT_TOKENS", "1050000")
)
DEFAULT_TRIGGER_RATIO = float(os.environ.get("INFINITE_CODEX_TRIGGER_RATIO", "0.85"))
DEFAULT_STATE_DIR = Path.home() / ".agent_state"
ENCODING = tiktoken.get_encoding("cl100k_base")
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

HANDOFF_PROMPT = """[System Override: Memory Limit Reached]
Generate a state dump for your successor instance.
1. THE PRIME DIRECTIVE
2. CURRENT STATE
3. RETAINED KNOWLEDGE
4. UNRESOLVED THREADS
5. NEXT ACTION
CRITICAL: You MUST end your response with the exact string: [END OF DUMP]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap Codex in a PTY and auto-handoff near a token limit."
    )
    parser.add_argument("session_name", nargs="?", default="default-agent")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help="Approximate context window used for trigger calculations.",
    )
    parser.add_argument(
        "--trigger-ratio",
        type=float,
        default=DEFAULT_TRIGGER_RATIO,
        help="Trigger handoff when token usage reaches this fraction of the limit.",
    )
    parser.add_argument(
        "--command",
        default="codex",
        help="CLI command to launch inside the wrapper.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="Directory where handoff state and lineage metadata are stored.",
    )
    parser.add_argument(
        "codex_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the wrapped CLI. Prefix them with --.",
    )
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def count_tokens(text: str) -> int:
    cleaned = strip_ansi(text)
    if not cleaned:
        return 0
    return len(ENCODING.encode(cleaned))


def load_generation(meta_file: Path) -> int:
    if not meta_file.exists():
        return 1
    try:
        data = json.loads(meta_file.read_text())
        return max(1, int(data.get("next_generation", 1)))
    except (OSError, ValueError, json.JSONDecodeError):
        return 1


def save_generation(meta_file: Path, generation: int) -> None:
    meta_file.write_text(json.dumps({"next_generation": generation}, indent=2) + "\n")


def announce(message: str) -> None:
    sys.stdout.write(f"\n[SYSTEM] {message}\n")
    sys.stdout.flush()


def resize_child(child: pexpect.spawn) -> None:
    columns, rows = shutil.get_terminal_size((100, 40))
    child.setwinsize(rows, columns)


def handle_input_line(
    line: str,
    session_name: str,
    generation: int,
    infinite_mode: bool,
    current_tokens: int,
) -> tuple[str, int, bool, int]:
    stripped = line.strip()

    if stripped == "INFINITE ON" and not infinite_mode:
        infinite_mode = True
        announce(f"Infinite Context Mode ACTIVATED for {session_name}.{generation}")

    rename_match = re.search(r"^/rename\s+([^\s]+)$", stripped)
    if rename_match:
        session_name = rename_match.group(1)
        announce(
            "Intercepted /rename. "
            f"Next rebirth will save as: {session_name}.{generation}"
        )

    fork_match = re.search(r"^/fork\s+([^\s]+)$", stripped)
    if fork_match:
        session_name = fork_match.group(1)
        generation = 1
        current_tokens = 0
        announce(f"Intercepted /fork. Switched bookkeeping to: {session_name}.1")

    if stripped == "/new":
        current_tokens = 0
        announce("Intercepted /new. Token counter reset to 0.")

    if stripped == "/compact":
        current_tokens = int(current_tokens * 0.5)
        announce("Intercepted /compact. Token counter halved heuristically.")

    if stripped == "/resume":
        current_tokens = 0
        announce("Intercepted /resume. Token counter reset for the resumed thread.")

    if stripped == "/agent":
        current_tokens = 0
        announce("Intercepted /agent. Token counter reset for the selected thread.")

    return session_name, generation, infinite_mode, current_tokens


def inject_previous_state(
    child: pexpect.spawn,
    session_name: str,
    generation: int,
    state_file: Path,
) -> int:
    if not state_file.exists():
        return 0

    previous_state = state_file.read_text()
    if not previous_state.strip():
        return 0

    continuation_prompt = (
        f"CONTINUING FROM {session_name}.{generation - 1}:\n{previous_state}"
    )
    child.sendline(continuation_prompt)
    return count_tokens(continuation_prompt)


def capture_handoff(child: pexpect.spawn, handoff_prompt: str) -> str:
    child.sendline(handoff_prompt)
    child.expect_exact("[END OF DUMP]", timeout=120)
    return strip_ansi(child.before).strip()


def run_agent_lineage(args: argparse.Namespace) -> None:
    args.state_dir.mkdir(parents=True, exist_ok=True)
    session_name = args.session_name

    while True:
        state_file = args.state_dir / f"{session_name}_state.txt"
        meta_file = args.state_dir / f"{session_name}_meta.json"
        generation = load_generation(meta_file)
        current_agent_id = f"{session_name}.{generation}"
        current_tokens = 0
        infinite_mode = state_file.exists()

        announce(f"LAUNCHING {current_agent_id} (PTY Proxy Active)")

        codex_args = list(args.codex_args)
        if codex_args and codex_args[0] == "--":
            codex_args = codex_args[1:]

        child = pexpect.spawn(
            args.command,
            codex_args,
            encoding="utf-8",
            dimensions=(40, 100),
        )
        resize_child(child)

        signal.signal(signal.SIGWINCH, lambda _signum, _frame: resize_child(child))

        current_tokens += inject_previous_state(child, session_name, generation, state_file)

        stdin_fd = sys.stdin.fileno()
        child_fd = child.fileno()
        pending_input = ""
        previous_tty = termios.tcgetattr(stdin_fd)

        try:
            tty.setraw(stdin_fd)

            while True:
                readable, _, _ = select.select([stdin_fd, child_fd], [], [], 0.1)

                if child_fd in readable:
                    chunk = os.read(child_fd, 4096)
                    if not chunk:
                        announce("Session closed manually.")
                        save_generation(meta_file, generation)
                        return

                    os.write(sys.stdout.fileno(), chunk)
                    if infinite_mode:
                        current_tokens += count_tokens(
                            chunk.decode("utf-8", errors="ignore")
                        )

                if stdin_fd in readable:
                    chunk = os.read(stdin_fd, 1024)
                    if not chunk:
                        child.sendeof()
                    else:
                        os.write(child_fd, chunk)
                        decoded = chunk.decode("utf-8", errors="ignore")
                        pending_input += decoded

                        if infinite_mode:
                            current_tokens += count_tokens(decoded)

                        while "\n" in pending_input or "\r" in pending_input:
                            line_break_index = min(
                                idx
                                for idx in (
                                    pending_input.find("\n"),
                                    pending_input.find("\r"),
                                )
                                if idx != -1
                            )
                            line = pending_input[:line_break_index]
                            pending_input = pending_input[line_break_index + 1 :]
                            (
                                session_name,
                                generation,
                                infinite_mode,
                                current_tokens,
                            ) = handle_input_line(
                                line,
                                session_name,
                                generation,
                                infinite_mode,
                                current_tokens,
                            )
                            state_file = args.state_dir / f"{session_name}_state.txt"
                            meta_file = args.state_dir / f"{session_name}_meta.json"
                            current_agent_id = f"{session_name}.{generation}"

                trigger_limit = int(args.max_context_tokens * args.trigger_ratio)
                if infinite_mode and current_tokens >= trigger_limit:
                    announce(
                        f"Approximate usage hit {current_tokens} tokens. "
                        "Injecting handoff prompt."
                    )
                    handoff_state = capture_handoff(child, HANDOFF_PROMPT)
                    state_file.write_text(handoff_state + "\n")
                    save_generation(meta_file, generation + 1)
                    child.terminate(force=True)
                    announce(f"{current_agent_id} terminated. Rebirthing...")
                    break

        except KeyboardInterrupt:
            announce("Wrapper interrupted. Exiting.")
            save_generation(meta_file, generation)
            child.terminate(force=True)
            return
        except pexpect.EOF:
            announce("Session closed manually.")
            save_generation(meta_file, generation)
            return
        finally:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, previous_tty)


if __name__ == "__main__":
    run_agent_lineage(parse_args())
