"""Detection lead-time calculation.

See AI_CONTEXT.md Section 12 (event-level anomaly detection), Section 13
(detection lead time), and Section 19 (`evaluation/lead_time.py`
contract: first valid alert, failure cycle, lead time, missed-detection
handling).

Definition:

    lead_time = failure_cycle - first_valid_alert_cycle

A positive lead time means the alert fired before failure (the useful,
intended case). The "first valid alert" is not simply the first cycle
where the score crosses the threshold — AI_CONTEXT.md Section 12
explicitly requires a persistence / consecutive-alert rule so that a
single noisy crossing does not count as an early-warning event. That
rule is never invented by this module; the caller must pass
`persistence` explicitly (see Section 12: "Do not invent a persistence
value unless the experiment specifies one.").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class LeadTimeResult:
    """Structured result of a lead-time computation for one engine.

    Attributes:
        first_alert_cycle: Cycle of the first *valid* (persistence-rule
            satisfying) alert, or None if no valid alert occurred.
        failure_cycle: The engine's actual (or provided ground-truth)
            failure cycle.
        lead_time: `failure_cycle - first_alert_cycle`, or None if there
            was no valid alert (missed detection).
        detected: Whether a valid alert occurred at all.
    """

    first_alert_cycle: Optional[int]
    failure_cycle: int
    lead_time: Optional[float]
    detected: bool


def first_valid_alert_cycle(
    cycles: Sequence[int],
    alerts: Sequence[bool],
    persistence: int,
) -> Optional[int]:
    """Find the cycle of the first alert that satisfies the persistence rule.

    A "valid" alert is the *start* of the first run of `persistence` or
    more consecutive True values in `alerts` (in the order given by
    `cycles`, which must already be sorted ascending for the trajectory).

    Args:
        cycles: Cycle indices for a single engine, ascending order.
        alerts: Boolean alert flag per cycle (same length as `cycles`).
        persistence: Minimum number of consecutive True alerts required
            for the run to count as a valid early-warning event. Must be
            a positive integer; `persistence=1` reduces to "first
            threshold crossing".

    Returns:
        The cycle value at which the first valid alert run begins, or
        None if no run of length >= `persistence` exists.

    Raises:
        ValueError: If `cycles`/`alerts` are mismatched in length, empty,
            or `persistence` is not a positive integer.
    """
    if len(cycles) != len(alerts):
        raise ValueError("cycles and alerts must have the same length")
    if len(cycles) == 0:
        raise ValueError("cycles/alerts must be non-empty")
    if not isinstance(persistence, int) or persistence <= 0:
        raise ValueError(f"persistence must be a positive integer, got {persistence!r}")

    alerts_arr = np.asarray(alerts, dtype=bool)
    run_length = 0
    for i, is_alert in enumerate(alerts_arr):
        run_length = run_length + 1 if is_alert else 0
        if run_length >= persistence:
            start_index = i - persistence + 1
            return int(cycles[start_index])
    return None


def detection_lead_time(
    cycles: Sequence[int],
    alerts: Sequence[bool],
    failure_cycle: int,
    persistence: int,
) -> LeadTimeResult:
    """Compute the detection lead time for a single engine trajectory.

    Args:
        cycles: Cycle indices for the engine, ascending order.
        alerts: Boolean alert flag per cycle (same length as `cycles`).
        failure_cycle: The actual (or ground-truth-provided) failure
            cycle for this engine. Must not be derived from `alerts`.
        persistence: Consecutive-alert rule passed through to
            `first_valid_alert_cycle` — see that function's docstring.

    Returns:
        A `LeadTimeResult`. If no valid alert occurred, `detected` is
        False, and `first_alert_cycle`/`lead_time` are both None
        (explicit missed-detection handling, per AI_CONTEXT.md
        Section 13) rather than being silently coerced to 0 or NaN.
    """
    alert_cycle = first_valid_alert_cycle(cycles, alerts, persistence)
    if alert_cycle is None:
        return LeadTimeResult(
            first_alert_cycle=None,
            failure_cycle=failure_cycle,
            lead_time=None,
            detected=False,
        )

    return LeadTimeResult(
        first_alert_cycle=alert_cycle,
        failure_cycle=failure_cycle,
        lead_time=float(failure_cycle - alert_cycle),
        detected=True,
    )


__all__ = ["LeadTimeResult", "first_valid_alert_cycle", "detection_lead_time"]
