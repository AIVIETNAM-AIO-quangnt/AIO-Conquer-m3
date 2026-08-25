"""PaySim ``step`` -> wall-clock timestamp, in integer microseconds.

PaySim has no real timestamp: ``step`` is a 1-based hour index over a 744-hour
(31-day) simulation, and averages ~8,500 rows per step. Using ``step`` alone as the
event time therefore produces massive ties, and any window function ordering by it
gets a non-deterministic row order -- which silently changes what "the previous
transaction" is.

So each step's rows are spread evenly across that step's hour.

**Integer microseconds with floor division, never float seconds.** This formula is
reproduced in three places (Python here, DuckDB SQL in the silver transform, and
pandas in the Colab notebook). Float arithmetic would diverge in the last ulp
between them and reorder tied transactions; ``//`` is chosen precisely because
DuckDB's integer division matches Python's exactly.
"""

from __future__ import annotations

from typing import Final

# 2024-01-01T00:00:00Z. Arbitrary but fixed: PaySim is a simulation with no real
# calendar. Overridable via C3_SIM_EPOCH_ISO, but changing it invalidates every
# derived event_ts_us already in the warehouse.
SIM_EPOCH_US: Final[int] = 1_704_067_200_000_000

US_PER_HOUR: Final[int] = 3_600_000_000
US_PER_SECOND: Final[int] = 1_000_000
HOURS_PER_DAY: Final[int] = 24

# PaySim spans 744 steps (31 simulated days).
MAX_STEP: Final[int] = 744


def derive_event_ts_us(step: int, intra_step_seq: int, step_cardinality: int) -> int:
    """Map a PaySim row to a unique, deterministic microsecond timestamp.

    Args:
        step: 1-based hour index from the dataset.
        intra_step_seq: 1-based rank of this row *within* its step, ordered by the
            source row order (``row_number() over (partition by step order by row_id)``).
        step_cardinality: total row count for this step (``count(*) over (partition by step)``).

    Returns:
        Microseconds since the Unix epoch.
    """
    if step < 1:
        raise ValueError(f"step is 1-based, got {step}")
    if intra_step_seq < 1:
        raise ValueError(f"intra_step_seq is 1-based, got {intra_step_seq}")
    offset = (intra_step_seq - 1) * US_PER_HOUR // max(step_cardinality, 1)
    return SIM_EPOCH_US + (step - 1) * US_PER_HOUR + offset


def derive_step_from_ts_us(ts_us: int) -> int:
    """Inverse of :func:`derive_event_ts_us` at hour granularity.

    Used for live requests, which carry a real timestamp but no ``step``.
    """
    return 1 + max(0, (ts_us - SIM_EPOCH_US) // US_PER_HOUR)


def sim_day(step: int) -> int:
    """0-based simulated day for a 1-based step."""
    return (step - 1) // HOURS_PER_DAY


def hour_of_day(step: int) -> int:
    """0-23 hour within the simulated day."""
    return (step - 1) % HOURS_PER_DAY


def day_of_week(step: int) -> int:
    """0-6 day-of-week index, taking simulated day 0 as day 0."""
    return sim_day(step) % 7
