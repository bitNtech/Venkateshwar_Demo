"""Graceful shutdown for the per-call worker tasks in main.py's WS handler.

Its own module so the shutdown policy is unit-testable without a websocket,
a model, or a running call - the same reason barge_in.py holds the "is the
agent currently talking" state instead of main.py.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging

logger = logging.getLogger("aica.tasks")

# How long a disconnecting call gives its workers to finish what they are
# holding before they are cancelled outright. The tail of a turn - its last
# clause, its agent_speaking_end, a closing call_control - is written by the
# very last statements of that turn, so cancelling the instant the socket
# drops truncated the call log at exactly the point worth recording. Short
# enough that a wedged model call cannot noticeably delay teardown.
WORKER_SHUTDOWN_GRACE_SECONDS = 2.0


async def shutdown_worker(task: asyncio.Task, name: str) -> None:
    """Let `task` finish its current item, then cancel it if it will not.

    Callers put a None sentinel on the task's queue first; this gives the task
    the brief window it needs to observe that sentinel and unwind normally,
    which is what lets an in-flight turn record its own ending. Cancellation
    remains the backstop for a worker genuinely stuck on a model call.
    """
    try:
        # shield() so the timeout cancels only this wait, not the worker -
        # otherwise wait_for would kill the task it is meant to let finish.
        await asyncio.wait_for(asyncio.shield(task), timeout=WORKER_SHUTDOWN_GRACE_SECONDS)
        return
    except asyncio.TimeoutError:
        logger.info("%s did not stop within the grace period; cancelling", name)
    except asyncio.CancelledError:
        # The task ended by cancellation of its own accord - barge-in leaves a
        # cancelled speak task behind. Nothing further to wait for.
        return
    except Exception:
        # A worker that died on its own already logged why; this is teardown.
        return

    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task
