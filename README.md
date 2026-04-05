# Infinite Codex Wrapper

`infinite-codex-wrapper` is a small Python PTY proxy for the local Codex CLI. It sits between your terminal and `codex`, tracks approximate token usage, and falls back to a structured checkpoint plus relaunch near a threshold.

The goal is continuity, not exact state transfer. This wrapper can only estimate tokens from terminal traffic; it cannot see the model's exact server-side context or internal summarization.

It can also act as a Telegram-controlled supervisor: when the agent appears idle or asks for input, the wrapper can notify a Telegram bot, and `/inject` messages from the bot are queued into the live Codex session without disabling the checkpoint rollover flow.

## What It Does

- Preserves a normal interactive TUI by running Codex inside a PTY.
- Tracks a named session lineage such as `my-project.1`, `my-project.2`, and so on.
- Stores successor checkpoint state under `~/.agent_state/` by default.
- Treats `INFINITE ON` as the opt-in switch for token tracking and checkpoint rollover.
- Sniffs typed `/fork`, `/new`, `/resume`, and `/agent` commands as bookkeeping hints, plus `/rename` as an undocumented best-effort extension.

## Important Caveats

- The token count is approximate. It is based on visible terminal input/output after ANSI stripping, not the true model context.
- Current OpenAI docs list a `1,050,000` token context window for `gpt-5.4`, but your Codex CLI session may use a different model or limit. Override `--max-context-tokens` if needed.
- OpenAI's Codex CLI docs currently document interactive slash commands such as `/new`, `/resume`, `/fork`, and `/agent`. `/rename` is not documented there, so the wrapper treats it as a best-effort extension only.
- A forced checkpoint can still fail if the model ignores the dump format or the CLI rendering changes.
- The checkpoint is intentionally lossy. It is designed for practical resume quality, not perfect transcript reconstruction.
- Telegram input/idleness detection is heuristic. The wrapper watches visible PTY output, not hidden model state.

## Install

```bash
git clone <your-repo-url>
cd infinite-codex-wrapper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Start the wrapper with a session name:

```bash
python3 wrapper.py my-backend-api
```

Seed a brand-new session from a file:

```bash
python3 wrapper.py my-backend-api --initial-prompt-file prompts/bootstrap.txt
```

Wrapper options can appear before or after the session name. Only arguments after a literal `--` are forwarded to Codex.

Pass extra arguments to Codex after `--`:

```bash
python3 wrapper.py my-backend-api -- --no-alt-screen --search
```

Opt into tracking from inside the Codex prompt:

```text
INFINITE ON
```

Once enabled, the wrapper writes a checkpoint to `~/.agent_state/my-backend-api_state.txt` and relaunches the next generation when estimated usage reaches `85%` of `--max-context-tokens`.

If `--initial-prompt-file` is provided, the file contents are sent as the first prompt only for a fresh generation-1 session. Checkpoint resume still takes priority on later generations.

## Telegram Control Plane

Set these environment variables before launching the wrapper:

```bash
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
```

Basic Telegram bot setup:

1. Open Telegram and talk to `@BotFather`.
2. Run `/newbot` and follow the prompts.
3. Copy the bot token BotFather gives you and export it as `TELEGRAM_BOT_TOKEN`.
4. Start a chat with your bot and send it any message.
5. Find your numeric chat id by calling:

```bash
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates"
```

6. Read the `message.chat.id` field from the response and export it as `TELEGRAM_CHAT_ID`.
7. Launch the wrapper with those environment variables set.

Then start the wrapper normally. Optional flags:

```bash
python3 wrapper.py my-backend-api \
  --idle-timeout-seconds 180 \
  --telegram-poll-seconds 5
```

Supported Telegram commands:

- `/inject <text>`: queue a prompt into the ongoing agent terminal
- `/answer <text>`: answer a detected input request
- `/status`: show pending queue and waiting state

Behavior:

- If the wrapper detects that the agent is waiting for input, it sends a Telegram alert with recent output.
- If the wrapper detects that the agent is idle for too long, it sends a Telegram idle alert.
- Any queued `/inject` message is persisted under the session state and survives checkpoint rebirth.
- The normal compact/checkpoint handoff flow still runs when context usage crosses the configured threshold.

## Configuration

You can change the trigger behavior with flags:

```bash
python3 wrapper.py my-backend-api \
  --max-context-tokens 1050000 \
  --trigger-ratio 0.85 \
  --initial-prompt-file prompts/bootstrap.txt \
  --idle-timeout-seconds 180 \
  --telegram-poll-seconds 5 \
  -- --no-alt-screen
```

You can also use environment variables:

```bash
export INFINITE_CODEX_MAX_CONTEXT_TOKENS=1050000
export INFINITE_CODEX_TRIGGER_RATIO=0.85
python3 wrapper.py my-backend-api
```

## Files

- `wrapper.py`: PTY proxy and lineage manager.
- `requirements.txt`: Python dependencies.
- `.gitignore`: common local ignores for Python and wrapper state.

## Notes From Review

The original pasted draft had several blocking issues:

- `SESSION_NAME = sys.argv if len(sys.argv) > 1 else "default-agent"` stores the full argv list instead of a single name.
- The Python indentation was broken, so the file would not run.
- The proxy loop only read Codex output; it did not forward keyboard input to the child process.
- Slash-command sniffing was reading the output buffer instead of the user's typed input.
- The README mixed code, prose, and headings in a way that would not render correctly on GitHub.
