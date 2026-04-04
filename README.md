# Infinite Codex Wrapper

`infinite-codex-wrapper` is a small Python PTY proxy for the local Codex CLI. It sits between your terminal and `codex`, tracks approximate token usage, tries native `/compact` first near a configurable threshold, and only falls back to a structured checkpoint plus relaunch when compaction is not enough.

The goal is continuity, not exact state transfer. This wrapper can only estimate tokens from terminal traffic; it cannot see the model's exact server-side context or internal summarization.

## What It Does

- Preserves a normal interactive TUI by running Codex inside a PTY.
- Tracks a named session lineage such as `my-project.1`, `my-project.2`, and so on.
- Stores successor checkpoint state under `~/.agent_state/` by default.
- Treats `INFINITE ON` as the opt-in switch for token tracking and hybrid rollover.
- Sniffs typed `/fork`, `/new`, `/compact`, `/resume`, and `/agent` commands as bookkeeping hints, plus `/rename` as an undocumented best-effort extension.

## Important Caveats

- The token count is approximate. It is based on visible terminal input/output after ANSI stripping, not the true model context.
- Current OpenAI docs list a `1,050,000` token context window for `gpt-5.4`, but your Codex CLI session may use a different model or limit. Override `--max-context-tokens` if needed.
- OpenAI's Codex CLI docs currently document interactive slash commands such as `/compact`, `/new`, `/resume`, `/fork`, and `/agent`. `/rename` is not documented there, so the wrapper treats it as a best-effort extension only.
- The hybrid flow is heuristic: the wrapper injects `/compact`, waits a short cooldown, and adjusts its local estimate before deciding whether to checkpoint.
- A forced checkpoint can still fail if the model ignores the dump format or the CLI rendering changes.
- The checkpoint is intentionally lossy. It is designed for practical resume quality, not perfect transcript reconstruction.

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

Pass extra arguments to Codex after `--`:

```bash
python3 wrapper.py my-backend-api -- --no-alt-screen --search
```

Opt into tracking from inside the Codex prompt:

```text
INFINITE ON
```

Once enabled, the wrapper first injects `/compact` when estimated usage reaches `85%` of `--max-context-tokens`. If the session still appears too full after a short cooldown, it writes a checkpoint to `~/.agent_state/my-backend-api_state.txt` and relaunches the next generation.

## Configuration

You can change the trigger behavior with flags:

```bash
python3 wrapper.py my-backend-api \
  --max-context-tokens 1050000 \
  --trigger-ratio 0.85 \
  --compact-cooldown-seconds 20 \
  --compact-reduction-ratio 0.5 \
  -- --no-alt-screen
```

You can also use environment variables:

```bash
export INFINITE_CODEX_MAX_CONTEXT_TOKENS=1050000
export INFINITE_CODEX_TRIGGER_RATIO=0.85
export INFINITE_CODEX_COMPACT_COOLDOWN_SECONDS=20
export INFINITE_CODEX_COMPACT_REDUCTION_RATIO=0.5
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
