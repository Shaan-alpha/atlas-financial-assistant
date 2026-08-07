from atlas.engine.prompt import build_system_prompt


def test_includes_known_profile_and_facts():
    prompt = build_system_prompt(
        {
            "name": "Shaan",
            "role": "equity analyst",
            "briefing_time": "08:30",
            "timezone": "Asia/Kolkata",
            "onboarding_state": "done",
        },
        [{"fact": "Covers semiconductors", "category": "focus"}],
    )

    assert "Shaan" in prompt
    assert "equity analyst" in prompt
    assert "Covers semiconductors" in prompt


def test_new_user_prompt_directs_onboarding():
    prompt = build_system_prompt(
        {
            "name": None,
            "role": None,
            "briefing_time": None,
            "timezone": "UTC",
            "onboarding_state": "new",
        },
        [],
    )

    assert "nothing yet" in prompt.lower()
    assert "one question at a time" in prompt.lower()


def test_prompt_forbids_command_surface():
    prompt = build_system_prompt(
        {
            "name": "A",
            "role": None,
            "briefing_time": None,
            "timezone": "UTC",
            "onboarding_state": "done",
        },
        [],
    )

    assert "slash command" in prompt.lower()
    assert "button" in prompt.lower()
