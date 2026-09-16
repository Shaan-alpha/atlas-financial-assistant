"""main() wiring.

main() had no coverage at all, which is how drop_pending_updates=False and
one-at-a-time update processing survived unnoticed. Nothing here starts a bot:
the port bind and run_polling are the only things that reach outside the
process, and both are replaced.
"""

import sys
import threading
from types import SimpleNamespace

import pytest
from telegram.ext import Application

import atlas.main as main
from atlas.config import get_settings
from atlas.ingress import handlers
from atlas.memory import store

pytestmark = pytest.mark.usefixtures("fresh_db")


@pytest.fixture
def run_main(monkeypatch):
    """Run main() up to the point it would start polling; hand back what it built."""
    seen = {"order": []}
    real_init_db = main.init_db

    def _run_polling(self, **kwargs):
        seen["app"] = self
        seen["polling"] = kwargs

    def _health(port, host="0.0.0.0"):
        seen["order"].append("health")
        seen["port"], seen["host"] = port, host

    def _init_db():
        seen["order"].append("init_db")
        real_init_db()

    def _exit(code):
        # main() ends in os._exit so a dead poller cannot linger as a zombie.
        # Unpatched, that would take pytest itself down with it.
        seen["exit_code"] = code

    monkeypatch.setattr(Application, "run_polling", _run_polling)
    monkeypatch.setattr(main.os, "_exit", _exit)
    monkeypatch.setattr(main, "mark_polling_stopped", lambda: seen.__setitem__("marked", True))
    monkeypatch.setattr(main, "start_health_server", _health)
    monkeypatch.setattr(main, "init_db", _init_db)
    monkeypatch.setattr(main, "_configure_logging", lambda level: None)

    def _run():
        main.main()
        return seen

    return _run


def test_updates_are_processed_concurrently(run_main):
    app = run_main()["app"]

    assert app.concurrent_updates == main.MAX_CONCURRENT_UPDATES
    # Bounded on purpose: concurrent_updates(True) silently means 256, which is
    # the whole connection pool and far past what a 5-rpm model quota can serve.
    assert 1 < main.MAX_CONCURRENT_UPDATES <= 64


def test_pending_updates_are_dropped_on_startup(run_main):
    """Telegram queues messages while the bot is down. Replaying them after a
    restart means answering questions the user asked hours ago, all at once."""
    assert run_main()["polling"]["drop_pending_updates"] is True


def test_every_message_kind_still_reaches_a_handler(run_main):
    app = run_main()["app"]
    callbacks = {h.callback for group in app.handlers.values() for h in group}

    assert callbacks == {
        handlers.start,
        handlers.on_text,
        handlers.on_voice,
        handlers.on_photo,
        handlers.on_document,
    }
    assert main._on_error in app.error_handlers


def test_the_port_is_bound_before_the_database(run_main):
    """main.py explains why: a host kills a web service that never opens a port,
    so binding second turns one database problem into two misleading errors."""
    assert run_main()["order"] == ["health", "init_db"]


def test_the_background_jobs_are_installed(run_main, monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "http://atlas.test/")
    get_settings.cache_clear()
    uid = store.get_or_create_user(42, "Shaan")
    store.set_profile(uid, briefing_time="08:30", timezone="Asia/Kolkata")

    names = sorted(j.name for j in run_main()["app"].job_queue.jobs())

    assert names == ["_job", "_keepalive", "_resync", f"briefing:{uid}"]


def test_no_public_url_means_no_self_ping(run_main):
    """The keep-alive would otherwise hammer whatever PUBLIC_URL happens to be."""
    names = {j.name for j in run_main()["app"].job_queue.jobs()}

    assert "_keepalive" not in names


def test_the_watchdog_spots_a_running_app_with_a_dead_poller():
    """The zombie, exactly: Application up, job queue ticking, polling task
    finished. Both flags drop together on a real shutdown, so only this shape
    means the bot is answering nobody."""
    done = SimpleNamespace(done=lambda: True)
    alive = SimpleNamespace(done=lambda: False)

    def app(running, task):
        return SimpleNamespace(
            running=running, updater=SimpleNamespace(_Updater__polling_task=task)
        )

    assert main._polling_is_dead(app(True, done)) is True
    # Healthy: still polling.
    assert main._polling_is_dead(app(True, alive)) is False
    # A genuine shutdown drops both — not a zombie, do not force-exit.
    assert main._polling_is_dead(app(False, done)) is False
    # No updater at all (job-queue-less builds) must not trip it.
    assert main._polling_is_dead(SimpleNamespace(running=True, updater=None)) is False


def test_the_watchdog_survives_a_renamed_internal():
    """It reads a private PTB attribute. If that name ever changes the watchdog
    must go quiet, not crash the bot it exists to protect."""
    app = SimpleNamespace(running=True, updater=SimpleNamespace())

    assert main._polling_is_dead(app) is False


def test_a_requested_stop_exits_cleanly(run_main, caplog):
    """run_polling only returns once a stop signal arrives: PTB catches SIGTERM
    itself and shuts down in order. Every `systemctl restart` used to log an ERROR
    and exit 1, so a deploy looked exactly like a crash in the journal."""
    with caplog.at_level("INFO", logger="atlas.main"):
        seen = run_main()

    assert seen["exit_code"] == 0
    assert "ERROR" not in [r.levelname for r in caplog.records if r.name == "atlas.main"]


def test_a_crashed_poller_takes_the_process_down(run_main, monkeypatch):
    """A poller that raises (a revoked token, a network that never comes back
    during bootstrap) must exit non-zero so the host restarts it and says why."""

    def _crash(self, **kwargs):
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(Application, "run_polling", _crash)

    seen = run_main()

    assert seen["exit_code"] == 1
    assert seen.get("marked") is True


async def test_polling_network_blips_are_one_line_warnings(caplog):
    """Telegram answers getUpdates with Bad Gateway some nights; PTB retries on
    its own. A full traceback at ERROR for each one buries real faults."""
    from telegram.error import NetworkError

    context = SimpleNamespace(error=NetworkError("Bad Gateway"))

    with caplog.at_level("DEBUG", logger="atlas.main"):
        await main._on_error(None, context)

    records = [r for r in caplog.records if r.name == "atlas.main"]
    assert [r.levelname for r in records] == ["WARNING"]
    assert records[0].exc_info is None


def test_sdk_chatter_is_filtered_but_real_warnings_are_not(caplog):
    """google-genai logs an INFO and a WARNING on every single model call."""
    import logging

    main._quiet_sdk_chatter()
    sdk = logging.getLogger("google_genai.models")

    with caplog.at_level("INFO", logger="google_genai.models"):
        sdk.info("AFC is enabled with max remote calls: 8.")
        sdk.warning(
            "Direct use of automatic function calling (AFC) in AsyncModels.generate_content "
            "is not recommended."
        )
        sdk.warning("Model gemini-x is deprecated")

    assert [r.getMessage() for r in caplog.records] == ["Model gemini-x is deprecated"]


def test_startup_does_not_import_yfinance(run_main):
    """Canary: no test may pay the 12s import. If this ever fails, something
    started importing yfinance at module scope instead of inside a call."""
    run_main()

    assert "yfinance" not in sys.modules


def test_health_answers_on_loopback_unless_a_host_must_reach_it(run_main, monkeypatch):
    """/diag spends provider quota; on the VM nothing outside needs the port."""
    assert run_main()["host"] == "127.0.0.1"

    monkeypatch.setenv("PUBLIC_URL", "https://atlas.example")
    get_settings.cache_clear()
    assert run_main()["host"] == "0.0.0.0"
