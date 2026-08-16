"""One turn at a time, per user.

Updates run concurrently so one slow turn does not stall everyone. But two turns
for the same person must not overlap: respond() reads the history before it
appends to it, so overlapping turns would each answer a prompt the other has
already invalidated, and would interleave their rows in the message log. Because
history is ordered by row id, a tangled pair stays tangled for the next twenty
turns.

What this guarantees is mutual exclusion: a user's turn runs start to finish
before their next one begins, so every reply is stored directly after the
question it answers.

What it does NOT guarantee is arrival order. The lock is taken inside respond(),
and each handler does variable-latency work before it gets there — the user
lookup, the typing indicator, and for voice or documents a download. Two messages
sent within a few hundred milliseconds can therefore reach the lock in either
order. They will be answered cleanly, just possibly the second one first. Fixing
that would mean locking at handler entry and holding it across multi-second file
downloads, which trades a rare reordering for a common stall.
"""

import asyncio
import weakref
from contextlib import asynccontextmanager

# Weak values: the only strong references are the frames inside `async with`, so
# an entry lives exactly as long as someone holds or waits on it. A bot with ten
# thousand users never accumulates ten thousand locks.
_locks: "weakref.WeakValueDictionary[int, asyncio.Lock]" = (
    weakref.WeakValueDictionary()
)


@asynccontextmanager
async def user_turn(user_id: int):
    """Serialize everything that reads-then-writes one user's message log.

    Event loop only. asyncio.Lock is not thread-safe, so this must never be
    entered from a tool running under asyncio.to_thread.
    """
    lock = _locks.get(user_id)
    if lock is None:
        # No await between the miss and the store, so no second task can wedge
        # a rival lock in between.
        lock = asyncio.Lock()
        _locks[user_id] = lock
    async with lock:
        yield
