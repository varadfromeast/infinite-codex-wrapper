from pathlib import Path

import wrapper


def test_parse_args_supports_wrapper_options_after_session_name(monkeypatch):
    monkeypatch.setattr(
        wrapper.sys,
        "argv",
        [
            "wrapper.py",
            "my-session",
            "--initial-prompt-file",
            "/tmp/bootstrap.txt",
            "--",
            "--no-alt-screen",
            "--search",
        ],
    )

    args = wrapper.parse_args()

    assert args.session_name == "my-session"
    assert args.initial_prompt_file == Path("/tmp/bootstrap.txt")
    assert args.codex_args == ["--no-alt-screen", "--search"]


def test_load_initial_prompt_file_reads_text_and_counts_tokens(tmp_path, monkeypatch):
    prompt_file = tmp_path / "bootstrap.txt"
    prompt_file.write_text("hello world")
    monkeypatch.setattr(wrapper, "count_tokens", lambda text: 99)

    prompt_text, token_count = wrapper.load_initial_prompt_file(prompt_file)

    assert prompt_text == "hello world"
    assert token_count == 99


def test_load_initial_prompt_file_handles_missing_file(tmp_path):
    prompt_text, token_count = wrapper.load_initial_prompt_file(tmp_path / "missing.txt")

    assert prompt_text is None
    assert token_count == 0


def test_detect_input_request_matches_question_prompt():
    output = """
    Assistant: Do you want to continue with option 1 or option 2?
    Please reply with your choice.
    """

    excerpt = wrapper.detect_input_request(output)

    assert excerpt is not None
    assert "Do you want to continue" in excerpt


def test_detect_input_request_returns_none_for_regular_output():
    output = """
    Running tests...
    12 passed in 0.42s
    Wrote docs/progress/current-status.md
    """

    assert wrapper.detect_input_request(output) is None


def test_detect_input_request_matches_tui_confirmation_choice():
    output = """
    $ rm -rf /Users/varad/V/repo/infinite-codex-wrapper/.build

    › 1. Yes, proceed (y)
      2. No, cancel (n)
    """

    excerpt = wrapper.detect_input_request(output)

    assert excerpt is not None
    assert "1. Yes, proceed (y)" in excerpt


def test_telegram_state_round_trip(tmp_path):
    state_file = tmp_path / "telegram.json"
    state = wrapper.load_telegram_state(state_file)
    state["pending_injections"].append("continue")
    state["awaiting_input"] = True
    wrapper.save_telegram_state(state_file, state)

    loaded = wrapper.load_telegram_state(state_file)

    assert loaded["pending_injections"] == ["continue"]
    assert loaded["awaiting_input"] is True
    assert loaded["last_update_id"] == 0


def test_process_telegram_updates_queues_injection(monkeypatch):
    sent_messages = []
    monkeypatch.setattr(wrapper.os, "environ", {
        "TELEGRAM_BOT_TOKEN": "token",
        "TELEGRAM_CHAT_ID": "123",
    })
    monkeypatch.setattr(
        wrapper,
        "submit_telegram_send",
        lambda executor, args, text: sent_messages.append(text),
    )
    args = wrapper.parse_args.__globals__["argparse"].Namespace(
        telegram_bot_token_env="TELEGRAM_BOT_TOKEN",
        telegram_chat_id_env="TELEGRAM_CHAT_ID",
    )
    state = wrapper.load_telegram_state(Path("/tmp/nonexistent-telegram-state.json"))
    response = {
        "result": [
            {
                "update_id": 7,
                "message": {
                    "chat": {"id": 123},
                    "text": "/inject continue working",
                },
            }
        ]
    }

    updated = wrapper.process_telegram_updates(args, state, "session", response, None)

    assert updated is True
    assert state["pending_injections"] == ["continue working"]
    assert state["last_update_id"] == 7
    assert sent_messages


def test_process_telegram_updates_queues_plain_reply_when_waiting(monkeypatch):
    monkeypatch.setattr(wrapper.os, "environ", {
        "TELEGRAM_BOT_TOKEN": "token",
        "TELEGRAM_CHAT_ID": "123",
    })
    monkeypatch.setattr(wrapper, "submit_telegram_send", lambda executor, args, text: None)
    args = wrapper.parse_args.__globals__["argparse"].Namespace(
        telegram_bot_token_env="TELEGRAM_BOT_TOKEN",
        telegram_chat_id_env="TELEGRAM_CHAT_ID",
    )
    state = wrapper.load_telegram_state(Path("/tmp/nonexistent-telegram-state.json"))
    state["awaiting_input"] = True
    response = {
        "result": [
            {
                "update_id": 8,
                "message": {
                    "chat": {"id": 123},
                    "text": "1",
                },
            }
        ]
    }

    updated = wrapper.process_telegram_updates(args, state, "session", response, None)

    assert updated is True
    assert state["pending_injections"] == ["1"]
    assert state["awaiting_input"] is False


def test_count_tokens_falls_back_when_encoding_is_unavailable(monkeypatch):
    monkeypatch.setattr(wrapper, "get_encoding", lambda: None)

    assert wrapper.count_tokens("abcd") == 1
    assert wrapper.count_tokens("abcdefgh") == 2
