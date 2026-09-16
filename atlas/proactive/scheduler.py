"""Scheduling briefings.

Users pick a local time, so each job is registered at the UTC equivalent. The
roster is re-read periodically rather than only at boot, so someone who sets a
briefing time mid-conversation gets tomorrow's briefing without a restart.
"""

import datetime as dt
import logging

from atlas.memory import store
from atlas.proactive import briefing

log = logging.getLogger(__name__)

# Short enough that a briefing time set mid-conversation takes effect the same
# day rather than tomorrow. The resync is a database read, so it is cheap.
RESYNC_INTERVAL = dt.timedelta(minutes=5)
JOB_PREFIX = "briefing:"
# APScheduler's default grace is one second: any stall on the event loop at the
# briefing minute (a slow turn, a restart) silently skipped that day's briefing.
MISFIRE_GRACE = int(dt.timedelta(minutes=15).total_seconds())


async def _run(context) -> None:
    payload = context.job.data
    sent = await briefing.send_to(
        context.bot, payload["user_id"], payload["telegram_id"], payload["timezone"]
    )
    log.info("briefing for %s: %s", payload["user_id"], "sent" if sent else "silent")


def sync_jobs(job_queue) -> int:
    """Register or refresh a briefing job per opted-in user. Returns the count."""
    # Roster first. Removing jobs before a read that then failed (Postgres
    # restarting under the resync) left nobody scheduled until the next success.
    try:
        users = store.users_with_briefings()
    except Exception:
        log.exception("could not read the briefing roster; keeping existing jobs")
        return 0

    for job in job_queue.jobs():
        if job.name and job.name.startswith(JOB_PREFIX):
            job.schedule_removal()

    registered = 0
    for user in users:
        try:
            when = briefing.utc_time_for(user["briefing_time"], user["timezone"])
        except Exception:
            log.warning(
                "user %s has an unusable briefing time %r",
                user["user_id"],
                user["briefing_time"],
            )
            continue

        job_queue.run_daily(
            _run,
            time=when,
            name=f"{JOB_PREFIX}{user['user_id']}",
            data=user,
            job_kwargs={"misfire_grace_time": MISFIRE_GRACE},
        )
        registered += 1

    log.info("scheduled %d briefing(s)", registered)
    return registered


async def _resync(context) -> None:
    sync_jobs(context.job_queue)


def install(job_queue) -> None:
    """Schedule existing users now, and keep the roster fresh as people opt in."""
    sync_jobs(job_queue)
    job_queue.run_repeating(_resync, interval=RESYNC_INTERVAL, first=RESYNC_INTERVAL)
