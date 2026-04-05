#!/usr/bin/env python3

import argparse
import hashlib
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
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import pexpect
    import tiktoken
except ModuleNotFoundError as exc:
    missing_module = exc.name or "required dependency"
    sys.stderr.write(
        "Missing Python dependency: "
        f"{missing_module}\n"
        "Create a virtual environment and install requirements first:\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
    )
    raise SystemExit(1) from exc

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
DEFAULT_MAX_AUTO_COMPACTS = int(
    os.environ.get("INFINITE_CODEX_MAX_AUTO_COMPACTS", "1")
)
DEFAULT_REARM_RATIO = float(os.environ.get("INFINITE_CODEX_REARM_RATIO", "0.7"))
DEFAULT_IDLE_TIMEOUT_SECONDS = float(
    os.environ.get("INFINITE_CODEX_IDLE_TIMEOUT_SECONDS", "180")
)
DEFAULT_TELEGRAM_POLL_SECONDS = float(
    os.environ.get("INFINITE_CODEX_TELEGRAM_POLL_SECONDS", "5")
)
DEFAULT_STATE_DIR = Path.home() / ".agent_state"
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
ENCODING = None
ENCODING_FAILED = False
INPUT_REQUEST_PATTERNS = [
    re.compile(
        r"(do you want|would you like|which option|choose|select|enter|provide|please reply)",
        re.IGNORECASE,
    ),
    re.compile(r"press enter to continue", re.IGNORECASE),
]

CHECKPOINT_PROMPT = """[System Override: Checkpoint Required]
The current session is near its practical context limit. Produce a checkpoint for a successor Codex instance.

This is a lossy checkpoint-resume mechanism, not an exact state transfer. Your job is to preserve the highest-value working context so the next session can continue effectively with minimal drift.

Priority:
1. Preserve the exact task objective.
2. Preserve irreversible or high-cost reasoning outcomes.
3. Preserve constraints, failed paths, and next actions.
4. Preserve only the context that materially helps the next agent continue.

Rules:
1. Output plain text only.
2. Follow the exact headings below in the exact order.
3. Keep every bullet concrete, specific, and information-dense.
4. Preserve exact file paths, commands, identifiers, errors, decisions, constraints, and next steps when known.
5. Do not restate generic background unless it is necessary for continuation.
6. Do not write motivational language, filler, or vague summaries.
7. If a section has no content, write `- None`.
8. If something is uncertain, mark it explicitly as `Uncertain: ...`.
9. The final line must be exactly `[END OF DUMP]`.
10. Do not add anything after `[END OF DUMP]`.

Use this template exactly:

TASK
- One short paragraph describing the real objective and what "done" looks like.

CURRENT CHECKPOINT
- What is already completed.
- What is partially completed.
- What failed or remains blocked.

HARD DEPENDENCIES
- Design choices or conclusions reached after careful consideration that the next agent should treat as fixed unless there is strong evidence to revisit them.
- Include why each one matters.
- Include any tradeoff already accepted.

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

FAILED OR REJECTED PATHS
- Approaches tried and abandoned.
- Why they failed, were rejected, or should not be retried without new evidence.

NEXT BEST ACTIONS
1. The first action the next agent should take.
2. The second action.
3. The third action.

USER PREFERENCES
- Any stated user preferences, workflow constraints, or communication preferences.

RESUME NOTE
- A brief note telling the next agent exactly where to re-enter the task.

[END OF DUMP]"""


def parse_args() -> argparse.Namespace:
    raw_args = sys.argv[1:]
    codex_args: list[str] = []
    if "--" in raw_args:
        separator_index = raw_args.index("--")
        codex_args = raw_args[separator_index + 1 :]
        raw_args = raw_args[:separator_index]

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
        "--max-auto-compacts",
        type=int,
        default=DEFAULT_MAX_AUTO_COMPACTS,
        help="How many wrapper-injected /compact attempts are allowed before checkpoint fallback.",
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
        "--initial-prompt-file",
        type=Path,
        help="Optional file whose contents are injected as the first prompt in a fresh session.",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT_SECONDS,
        help="Notify Telegram when the agent appears idle for this many seconds.",
    )
    parser.add_argument(
        "--telegram-poll-seconds",
        type=float,
        default=DEFAULT_TELEGRAM_POLL_SECONDS,
        help="How often to poll Telegram for injected messages.",
    )
    parser.add_argument(
        "--telegram-bot-token-env",
        default="TELEGRAM_BOT_TOKEN",
        help="Environment variable containing the Telegram bot token.",
    )
    parser.add_argument(
        "--telegram-chat-id-env",
        default="TELEGRAM_CHAT_ID",
        help="Environment variable containing the authorized Telegram chat id.",
    )
    args = parser.parse_args(raw_args)
    args.max_auto_compacts = max(0, args.max_auto_compacts)
    args.codex_args = codex_args
    return args


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def get_encoding():
    global ENCODING, ENCODING_FAILED
    if ENCODING is not None:
        return ENCODING
    if ENCODING_FAILED:
        return None
    try:
        ENCODING = tiktoken.get_encoding("cl100k_base")
        return ENCODING
    except Exception:
        ENCODING_FAILED = True
        announce(
            "Unable to load tiktoken encoding data. Falling back to a rough token estimate."
        )
        return None


def count_tokens(text: str) -> int:
    cleaned = strip_ansi(text)
    if not cleaned:
        return 0
    encoding = get_encoding()
    if encoding is None:
        return max(1, len(cleaned) // 4)
    return len(encoding.encode(cleaned))


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


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def load_json_dict(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            merged = dict(default)
            merged.update(data)
            return merged
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return dict(default)


def save_json_dict(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_telegram_state(state_file: Path) -> dict:
    return load_json_dict(
        state_file,
        {
            "last_update_id": 0,
            "pending_injections": [],
            "awaiting_input": False,
            "last_input_request_hash": "",
            "last_idle_alert_hash": "",
        },
    )


def save_telegram_state(state_file: Path, state: dict) -> None:
    save_json_dict(state_file, state)


def telegram_enabled(args: argparse.Namespace) -> bool:
    return bool(os.environ.get(args.telegram_bot_token_env)) and bool(
        os.environ.get(args.telegram_chat_id_env)
    )


def telegram_api_request(
    args: argparse.Namespace,
    method: str,
    payload: dict,
) -> dict | None:
    token = os.environ.get(args.telegram_bot_token_env)
    if not token:
        return None
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8", errors="ignore")
        parsed = json.loads(body)
        if isinstance(parsed, dict) and parsed.get("ok"):
            return parsed
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        pass
    return None


def telegram_send_message(
    args: argparse.Namespace,
    text: str,
) -> bool:
    chat_id = os.environ.get(args.telegram_chat_id_env)
    if not chat_id:
        return False
    response = telegram_api_request(
        args,
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text[:4000],
        },
    )
    return response is not None


def poll_telegram_updates(
    args: argparse.Namespace,
    telegram_state: dict,
    session_name: str,
) -> bool:
    chat_id = os.environ.get(args.telegram_chat_id_env)
    if not chat_id:
        return False

    response = telegram_api_request(
        args,
        "getUpdates",
        {
            "timeout": 1,
            "offset": telegram_state["last_update_id"] + 1,
        },
    )
    if response is None:
        return False

    updated = False
    for item in response.get("result", []):
        update_id = int(item.get("update_id", 0))
        telegram_state["last_update_id"] = max(telegram_state["last_update_id"], update_id)
        message = item.get("message") or {}
        if str(message.get("chat", {}).get("id")) != str(chat_id):
            continue
        text = (message.get("text") or "").strip()
        if not text:
            continue

        if text.startswith("/inject"):
            payload = text[len("/inject") :].strip()
            if payload:
                telegram_state["pending_injections"].append(payload)
                updated = True
                telegram_send_message(
                    args,
                    f"[{session_name}] queued injection ({len(telegram_state['pending_injections'])} pending).",
                )
            continue

        if text.startswith("/answer"):
            payload = text[len("/answer") :].strip()
            if payload:
                telegram_state["pending_injections"].append(payload)
                telegram_state["awaiting_input"] = False
                updated = True
                telegram_send_message(args, f"[{session_name}] queued answer.")
            continue

        if text.startswith("/status"):
            pending = len(telegram_state["pending_injections"])
            waiting = "yes" if telegram_state["awaiting_input"] else "no"
            telegram_send_message(
                args,
                f"[{session_name}] pending_injections={pending}, awaiting_input={waiting}",
            )
            continue

        if telegram_state["awaiting_input"]:
            telegram_state["pending_injections"].append(text)
            telegram_state["awaiting_input"] = False
            updated = True
            telegram_send_message(args, f"[{session_name}] queued reply.")

    return updated


def detect_input_request(recent_output: str) -> str | None:
    cleaned = strip_ansi(recent_output).strip()
    if not cleaned:
        return None

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    excerpt = "\n".join(lines[-8:])
    if not excerpt:
        return None

    if any(pattern.search(excerpt) for pattern in INPUT_REQUEST_PATTERNS):
        return excerpt
    if lines and lines[-1].endswith("?"):
        return excerpt
    return None


def notify_input_request(
    args: argparse.Namespace,
    telegram_state: dict,
    telegram_state_file: Path,
    session_name: str,
    current_agent_id: str,
    excerpt: str,
) -> None:
    request_hash = stable_hash(excerpt)
    if telegram_state["last_input_request_hash"] == request_hash:
        return

    telegram_state["awaiting_input"] = True
    telegram_state["last_input_request_hash"] = request_hash
    save_telegram_state(telegram_state_file, telegram_state)
    telegram_send_message(
        args,
        f"[{session_name}] {current_agent_id} is waiting for input.\n\n"
        f"{excerpt}\n\n"
        "Reply with plain text, or use /answer ... or /inject ...",
    )


def notify_idle(
    args: argparse.Namespace,
    telegram_state: dict,
    telegram_state_file: Path,
    session_name: str,
    current_agent_id: str,
    recent_output: str,
) -> None:
    excerpt = strip_ansi(recent_output).strip()
    lines = [line.strip() for line in excerpt.splitlines() if line.strip()]
    tail = "\n".join(lines[-8:]) if lines else "- No recent output -"
    idle_hash = stable_hash(f"{current_agent_id}:{tail}")
    if telegram_state["last_idle_alert_hash"] == idle_hash:
        return

    telegram_state["last_idle_alert_hash"] = idle_hash
    save_telegram_state(telegram_state_file, telegram_state)
    telegram_send_message(
        args,
        f"[{session_name}] {current_agent_id} appears idle.\n\nRecent output:\n{tail}\n\n"
        "Use /inject ... to queue a prompt.",
    )


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
) -> tuple[int, bool]:
    if not state_file.exists():
        return 0, False

    previous_state = state_file.read_text()
    if not previous_state.strip():
        return 0, False

    continuation_prompt = (
        f"CONTINUING FROM {session_name}.{generation - 1}:\n{previous_state}"
    )
    child.sendline(continuation_prompt)
    return count_tokens(continuation_prompt), True


def load_initial_prompt_file(prompt_file: Path | None) -> tuple[str | None, int]:
    if prompt_file is None:
        return None, 0
    if not prompt_file.exists():
        announce(f"Initial prompt file not found: {prompt_file}")
        return None, 0

    prompt_text = prompt_file.read_text()
    if not prompt_text.strip():
        announce(f"Initial prompt file is empty: {prompt_file}")
        return None, 0

    announce(f"Loaded initial prompt from: {prompt_file}")
    return prompt_text, count_tokens(prompt_text)


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
        telegram_state_file = args.state_dir / f"{session_name}_telegram.json"
        generation = load_generation(meta_file)
        current_agent_id = f"{session_name}.{generation}"
        current_tokens = 0
        infinite_mode = state_file.exists()
        auto_compact_count = 0
        compact_in_flight = False
        compact_started_at = 0.0
        pre_compact_tokens = 0
        telegram_state = load_telegram_state(telegram_state_file)
        last_output_at = time.monotonic()
        last_input_at = time.monotonic()
        last_telegram_poll_at = 0.0
        recent_output_buffer = ""

        announce(f"LAUNCHING {current_agent_id} (PTY Proxy Active)")

        codex_args = list(args.codex_args)
        if codex_args and codex_args[0] == "--":
            codex_args = codex_args[1:]

        initial_prompt_text = None
        initial_prompt_tokens = 0
        if generation == 1 and not state_file.exists():
            initial_prompt_text, initial_prompt_tokens = load_initial_prompt_file(
                args.initial_prompt_file
            )
            if initial_prompt_text is not None:
                codex_args.append(initial_prompt_text)

        child = pexpect.spawn(
            args.command,
            codex_args,
            encoding="utf-8",
            dimensions=(40, 100),
        )
        resize_child(child)

        signal.signal(signal.SIGWINCH, lambda _signum, _frame: resize_child(child))

        injected_tokens, resumed_from_state = inject_previous_state(
            child, session_name, generation, state_file
        )
        current_tokens += injected_tokens
        if not resumed_from_state and initial_prompt_text is not None:
            current_tokens += initial_prompt_tokens

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
                    last_output_at = time.monotonic()
                    decoded_chunk = chunk.decode("utf-8", errors="ignore")
                    recent_output_buffer = (recent_output_buffer + decoded_chunk)[-8000:]
                    if telegram_state["last_idle_alert_hash"]:
                        telegram_state["last_idle_alert_hash"] = ""
                        save_telegram_state(telegram_state_file, telegram_state)
                    if infinite_mode:
                        current_tokens += count_tokens(decoded_chunk)

                    if telegram_enabled(args):
                        maybe_request = detect_input_request(recent_output_buffer)
                        if maybe_request:
                            notify_input_request(
                                args,
                                telegram_state,
                                telegram_state_file,
                                session_name,
                                current_agent_id,
                                maybe_request,
                            )

                if stdin_fd in readable:
                    chunk = os.read(stdin_fd, 1024)
                    if not chunk:
                        child.sendeof()
                    else:
                        os.write(child_fd, chunk)
                        last_input_at = time.monotonic()
                        if telegram_state["last_idle_alert_hash"]:
                            telegram_state["last_idle_alert_hash"] = ""
                            save_telegram_state(telegram_state_file, telegram_state)
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
                            new_telegram_state_file = (
                                args.state_dir / f"{session_name}_telegram.json"
                            )
                            if new_telegram_state_file != telegram_state_file:
                                telegram_state_file = new_telegram_state_file
                                telegram_state = load_telegram_state(telegram_state_file)
                            current_agent_id = f"{session_name}.{generation}"
                            if compact_seen and compact_in_flight:
                                auto_compact_count += 1
                                compact_in_flight = False

                now = time.monotonic()
                if telegram_enabled(args) and (
                    now - last_telegram_poll_at >= args.telegram_poll_seconds
                ):
                    if poll_telegram_updates(args, telegram_state, session_name):
                        save_telegram_state(telegram_state_file, telegram_state)
                    last_telegram_poll_at = now

                if (
                    telegram_enabled(args)
                    and telegram_state["pending_injections"]
                    and not compact_in_flight
                ):
                    queued_text = telegram_state["pending_injections"].pop(0)
                    child.send(queued_text)
                    child.send("\n")
                    last_input_at = time.monotonic()
                    if infinite_mode:
                        current_tokens += count_tokens(queued_text) + count_tokens("\n")
                    telegram_state["awaiting_input"] = False
                    save_telegram_state(telegram_state_file, telegram_state)
                    announce(
                        f"Injected queued Telegram prompt ({len(telegram_state['pending_injections'])} remaining)."
                    )

                if (
                    telegram_enabled(args)
                    and not telegram_state["awaiting_input"]
                    and args.idle_timeout_seconds > 0
                    and now - max(last_output_at, last_input_at) >= args.idle_timeout_seconds
                ):
                    notify_idle(
                        args,
                        telegram_state,
                        telegram_state_file,
                        session_name,
                        current_agent_id,
                        recent_output_buffer,
                    )

                if current_tokens < rearm_limit:
                    auto_compact_count = 0

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
                    if (
                        auto_compact_count < args.max_auto_compacts
                        and not compact_in_flight
                    ):
                        announce(
                            f"Approximate usage hit {current_tokens} tokens. "
                            f"Injecting /compact ({auto_compact_count + 1}/"
                            f"{args.max_auto_compacts}) before checkpoint fallback."
                        )
                        os.write(child_fd, b"/compact\n")
                        compact_in_flight = True
                        compact_started_at = time.monotonic()
                        pre_compact_tokens = max(current_tokens, 1)
                        continue

                    if compact_in_flight:
                        continue

                    announce(
                        f"Approximate usage hit {current_tokens} tokens. "
                        f"Auto-compaction budget exhausted after {auto_compact_count} "
                        "attempt(s). Injecting checkpoint prompt."
                    )
                    checkpoint_state, completed = capture_checkpoint(
                        child, CHECKPOINT_PROMPT
                    )
                    if checkpoint_state:
                        state_file.write_text(checkpoint_state + "\n")

                    if completed:
                        save_generation(meta_file, generation + 1)
                        save_telegram_state(telegram_state_file, telegram_state)
                        child.terminate(force=True)
                        announce(f"{current_agent_id} terminated after checkpoint. Rebirthing...")
                        break

                    save_generation(meta_file, generation)
                    save_telegram_state(telegram_state_file, telegram_state)
                    child.terminate(force=True)
                    announce(
                        f"Checkpoint prompt timed out for {current_agent_id}. "
                        "Saved partial state and exiting cleanly."
                    )
                    return

        except KeyboardInterrupt:
            announce("Wrapper interrupted. Exiting.")
            save_generation(meta_file, generation)
            save_telegram_state(telegram_state_file, telegram_state)
            child.terminate(force=True)
            return
        except pexpect.TIMEOUT:
            announce("Unexpected timeout while interacting with Codex. Exiting cleanly.")
            save_generation(meta_file, generation)
            save_telegram_state(telegram_state_file, telegram_state)
            child.terminate(force=True)
            return
        except pexpect.EOF:
            announce("Session closed manually.")
            save_generation(meta_file, generation)
            save_telegram_state(telegram_state_file, telegram_state)
            return
        finally:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, previous_tty)


if __name__ == "__main__":
    run_agent_lineage(parse_args())
