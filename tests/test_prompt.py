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



def test_only_the_newest_facts_ride_along():
    from atlas.engine import prompt

    facts = [{"category": "general", "fact": f"fact {i}"} for i in range(100)]

    text = prompt.build_system_prompt({}, facts)

    lines = text.splitlines()
    assert "- [general] fact 0" in lines
    assert f"- [general] fact {prompt.MAX_FACTS_IN_PROMPT - 1}" in lines
    assert f"- [general] fact {prompt.MAX_FACTS_IN_PROMPT}" not in lines


def test_the_prompt_forbids_assuming_dollars():
    from atlas.engine import prompt

    assert "Never assume dollars" in prompt.BASE
