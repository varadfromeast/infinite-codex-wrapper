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
import time
import tty
from pathlib import Path

import pexpect
import tiktoken

DEFAULT_MAX_CONTEXT_TOKENS = int(
    os.environ.get("INFINITE_CODEX_MAX_CONTEXT_TOKENS", "1050000")
)
DEFAULT_TRIGGER_RATIO = float(os.environ.get("INFINITE_CODEX_TRIGGER_RATIO", "0.85"))
DEFAULT_COMPACT_COOLDOWN_SECONDS = float(
    os.environ.get("INFINITE_CODEX_COMPACT_COOLDOWN_SECONDS", "20")
)
DEFAULT_COMPACT_REDUCTION_RATIO = float(
    os.environ.get("INFINITE_CODEX_COMPACT_REDUCTION_RATIO", "0.5")
)
DEFAULT_REARM_RATIO = float(os.environ.get("INFINITE_CODEX_REARM_RATIO", "0.7"))
DEFAULT_STATE_DIR = Path.home() / ".agent_state"
ENCODING = tiktoken.get_encoding("cl100k_base")
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

CHECKPOINT_PROMPT = """[System Override: Checkpoint Required]
The current session is near its practical context limit. Produce a checkpoint for a successor Codex instance.

This is a lossy checkpoint-resume mechanism, not an exact state transfer. Your job is to preserve the highest-value working context so the next session can continue effectively with minimal drift.

Rules:
1. Output plain text only.
2. Follow the exact headings below in the exact order.
3. Keep every bullet concrete and information-dense.
4. Preserve exact file paths, commands, identifiers, errors, decisions, constraints, and next steps when known.
5. Do not restate generic background unless it is necessary for the next move.
6. If a section has no content, write `- None`.
7. The final line must be exactly `[END OF DUMP]`.
8. Do not add anything after `[END OF DUMP]`.

Use this template exactly:

TASK
- One short paragraph describing the real objective and what "done" looks like.

CURRENT CHECKPOINT
- What is already completed.
- What is partially completed.
- What failed or remains blocked.

DECISIONS THAT MUST NOT BE LOST
- Important decisions already made.
- Why those decisions were made.
- Assumptions or constraints the next agent must keep.

REPO STATE
- Relevant files and what changed in each.
- Relevant commands already run and their outcomes.
- Relevant tool results, errors, or observations.

OPEN PROBLEMS
- Remaining bugs, risks, ambiguities, or unanswered questions.
- For each one, say what the next agent needs to verify or decide.

NEXT BEST ACTIONS
1. The first action the next agent should take.
2. The second action.
3. The third action.

USER PREFERENCES
- Any stated user preferences or workflow constraints.

RESUME NOTE
- A brief note telling the next agent where to re-enter the task.

[END OF DUMP]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap Codex in a PTY and auto-checkpoint near a token limit."
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
        help="Trigger auto-compaction/checkpoint logic at this fraction of the limit.",
    )
    parser.add_argument(
        "--compact-cooldown-seconds",
        type=float,
        default=DEFAULT_COMPACT_COOLDOWN_SECONDS,
        help="How long to wait after injecting /compact before evaluating fallback.",
    )
    parser.add_argument(
        "--compact-reduction-ratio",
        type=float,
        default=DEFAULT_COMPACT_REDUCTION_RATIO,
        help="Heuristic local token reduction applied after auto-compaction.",
    )
    parser.add_argument(
        "--rearm-ratio",
        type=float,
        default=DEFAULT_REARM_RATIO,
        help="Re-arm auto-compaction once tokens fall below this fraction of the trigger point.",
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
        help="Directory where checkpoint state and lineage metadata are stored.",
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
) -> tuple[str, int, bool, int, bool]:
    stripped = line.strip()
    compact_seen = False

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

    if stripped == "/fork":
        generation = 1
        current_tokens = 0
        announce(
            "Intercepted /fork. Token counter reset and lineage restarted for the forked thread."
        )

    if stripped == "/new":
        current_tokens = 0
        announce("Intercepted /new. Token counter reset to 0.")

    if stripped == "/compact":
        compact_seen = True
        current_tokens = int(current_tokens * 0.5)
        announce("Intercepted /compact. Token counter halved heuristically.")

    if stripped == "/resume":
        current_tokens = 0
        announce("Intercepted /resume. Token counter reset for the resumed thread.")

    if stripped == "/agent":
        current_tokens = 0
        announce("Intercepted /agent. Token counter reset for the selected thread.")

    return session_name, generation, infinite_mode, current_tokens, compact_seen


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


def capture_checkpoint(child: pexpect.spawn, checkpoint_prompt: str) -> tuple[str, bool]:
    child.sendline(checkpoint_prompt)
    try:
        child.expect_exact("[END OF DUMP]", timeout=120)
        return strip_ansi(child.before).strip(), True
    except pexpect.TIMEOUT:
        partial_output = strip_ansi(child.before).strip()
        return partial_output, False


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
        auto_compact_attempted = False
        compact_in_flight = False
        compact_started_at = 0.0
        pre_compact_tokens = 0

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
                trigger_limit = int(args.max_context_tokens * args.trigger_ratio)
                rearm_limit = int(trigger_limit * args.rearm_ratio)

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
                                compact_seen,
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
                            if compact_seen:
                                auto_compact_attempted = True

                if current_tokens < rearm_limit:
                    auto_compact_attempted = False

                if compact_in_flight and (
                    time.monotonic() - compact_started_at
                    >= args.compact_cooldown_seconds
                ):
                    compact_in_flight = False
                    reduced_estimate = int(
                        pre_compact_tokens * args.compact_reduction_ratio
                    )
                    current_tokens = min(current_tokens, reduced_estimate)
                    announce(
                        "Auto-compaction cooldown finished. "
                        f"Heuristic token estimate adjusted to {current_tokens}."
                    )

                if infinite_mode and current_tokens >= trigger_limit:
                    if not auto_compact_attempted and not compact_in_flight:
                        announce(
                            f"Approximate usage hit {current_tokens} tokens. "
                            "Injecting /compact before checkpoint fallback."
                        )
                        os.write(child_fd, b"/compact\n")
                        compact_in_flight = True
                        compact_started_at = time.monotonic()
                        pre_compact_tokens = max(current_tokens, 1)
                        auto_compact_attempted = True
                        continue

                    if compact_in_flight:
                        continue

                    announce(
                        f"Approximate usage hit {current_tokens} tokens. "
                        "Injecting checkpoint prompt."
                    )
                    checkpoint_state, completed = capture_checkpoint(
                        child, CHECKPOINT_PROMPT
                    )
                    if checkpoint_state:
                        state_file.write_text(checkpoint_state + "\n")

                    if completed:
                        save_generation(meta_file, generation + 1)
                        child.terminate(force=True)
                        announce(f"{current_agent_id} terminated after checkpoint. Rebirthing...")
                        break

                    save_generation(meta_file, generation)
                    child.terminate(force=True)
                    announce(
                        f"Checkpoint prompt timed out for {current_agent_id}. "
                        "Saved partial state and exiting cleanly."
                    )
                    return

        except KeyboardInterrupt:
            announce("Wrapper interrupted. Exiting.")
            save_generation(meta_file, generation)
            child.terminate(force=True)
            return
        except pexpect.TIMEOUT:
            announce("Unexpected timeout while interacting with Codex. Exiting cleanly.")
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
